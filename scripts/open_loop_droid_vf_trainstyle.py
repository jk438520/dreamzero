#!/usr/bin/env python3
"""Run inference that mirrors the training forward path as closely as possible.

This script builds noisy latents/actions using the action head scheduler (FlowMatchScheduler),
calls the diffusion model with those noisy inputs (the same call used during training),
and reconstructs predicted x0 for the action channels so results are comparable to training.

Limitations: this reproduces the training forward call but does not run the full optimizer
or return training losses. It attempts to reuse the exact scheduler/add_noise logic.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from tianshou.data import Batch

from open_loop_droid_vf_common import (
    make_droid_dataset,
    build_obs,
    get_gt_action_dict,
    get_dataset_prompt,
    make_policy,
    make_arg_parser,
    ACTION_KEY_ORDER,
    ACTION_SLICES,
    save_plots,
)


def run_trainstyle(args, context_length: int = 33):
    device = args.device
    policy = make_policy(args.model_path, device)
    model = policy.trained_model
    action_head = model.action_head

    try:
        dataset = make_droid_dataset(args.dataset_path)
    except FileNotFoundError as e:
        print("Error loading dataset:", e)
        print("You can create or point to a compatible DROID dataset, or run with --dataset_path pointing to an available dataset.")
        return
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    num = min(args.num_samples, len(dataset))
    times = []
    # Prepare containers for per-key predictions to generate plots like open_loop_yam
    preds_per_key = {k: [] for k in ACTION_KEY_ORDER}
    gts_per_key = {k: [] for k in ACTION_KEY_ORDER}

    print(f"Running training-style inference {num} samples, context={context_length} ...")

    for i in range(num):
        idx = args.start_idx + i
        prompt = args.prompt
        if args.use_dataset_prompt:
            prompt = get_dataset_prompt(dataset, idx)
        obs = build_obs(dataset, idx, prompt, context_length=context_length)
        gt = get_gt_action_dict(dataset, idx)

        # Add action GT into obs so transforms produce action tensors (first timestep filled)
        # Build action tensor: shape (1, action_horizon, action_dim) with GT in first step
        action_horizon = getattr(action_head, "action_horizon", model.config.action_horizon)
        action_dim = getattr(action_head, "action_dim", model.config.action_dim)
        # Extract full action vector and place at first horizon step
        action_array = np.zeros((1, action_horizon, action_dim), dtype=np.float32)
        # Build flat GT action vector according to ACTION_SLICES
        gt_flat_vec = np.zeros((action_dim,), dtype=np.float32)
        for key in ACTION_KEY_ORDER:
            if key in gt:
                start, end = ACTION_SLICES[key]
                vals = np.asarray(gt[key], dtype=np.float32).reshape(-1)
                copy_len = min(end - start, vals.shape[0])
                gt_flat_vec[start:start+copy_len] = vals[:copy_len]
        action_array[0, 0, :] = gt_flat_vec

        obs["action"] = action_array  # Tianshou/batch transform expects this key

        # Run through policy.apply to get normalized inputs
        batch = Batch(obs=obs)
        batch = policy.apply(batch)
        normalized = batch.normalized_obs

        # Prepare model inputs via VLA.prepare_input to ensure device/dtype
        backbone_inputs, action_inputs = model.prepare_input(normalized)

        # Move everything to device
        def to_dev(x):
            if torch.is_tensor(x):
                return x.to(device)
            return x

        backbone_inputs = {k: to_dev(v) for k, v in backbone_inputs.items()}
        action_inputs = {k: to_dev(v) for k, v in action_inputs.items()}

        # Follow training forward steps to build latents + noisy versions
        with torch.inference_mode():
            # Encode video -> latents
            videos = action_inputs["images"]  # expected shape (B, T, H, W, C) uint8 or float
            # Reuse action_head.encode_video and encode_image helpers
            latents = action_head.encode_video(videos, tiled=action_head.tiled, tile_size=(action_head.tile_size_height, action_head.tile_size_width), tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width))
            _, _, num_frames, height, width = videos.shape
            image = videos[:, :, :1].transpose(1, 2)
            clip_feas, ys, _ = action_head.encode_image(image, num_frames, height, width)

            latents = latents.to(action_head.device)
            clip_feas = clip_feas.to(action_head.device)
            ys = ys.to(action_head.device)

            # Prepare noise and actions
            noise = torch.randn_like(latents)
            actions = torch.from_numpy(action_array).to(device=action_head.device, dtype=latents.dtype)
            noise_action = torch.randn_like(actions)

            # Timestep sampling following training heuristics (uniform/beta)
            if action_head.config.decouple_video_action_noise:
                video_noise_ratio = action_head.video_beta_dist.sample([noise.shape[0], noise.shape[1]]).to(action_head.device)
                timestep_id = ((1.0 - video_noise_ratio) * action_head.scheduler.num_train_timesteps).long().to(action_head.device)
                noise_mode = "DECOUPLED"
            elif action_head.config.use_high_noise_emphasis:
                noise_ratio = action_head.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]]).to(action_head.device)
                timestep_id = ((1.0 - noise_ratio) * action_head.scheduler.num_train_timesteps).long().to(action_head.device)
                noise_mode = "HIGH_NOISE_EMPHASIS"
            else:
                timestep_id = torch.randint(0, action_head.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]), device=action_head.device)
                noise_mode = "STANDARD"

            timestep_id_block = timestep_id[:, 1:].reshape(timestep_id.shape[0], -1, action_head.num_frame_per_block)
            timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
            timestep_id_block = timestep_id_block.reshape(timestep_id.shape[0], -1)
            timestep_id = torch.cat([timestep_id[:, :1], timestep_id_block], dim=1)

            timestep = action_head.scheduler.timesteps[timestep_id].to(action_head.device)
            tokens_per_frame = (height // 2) * (width // 2)
            seq_len = num_frames * tokens_per_frame

            noisy_latents = action_head.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))

            # Action timestep sampling
            if action_head.config.decouple_video_action_noise:
                timestep_action_id = torch.randint(0, action_head.scheduler.num_train_timesteps, (actions.shape[0], actions.shape[1]), device=action_head.device)
            else:
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1)) if (noise.shape[1]-1)>0 else torch.zeros((actions.shape[0], actions.shape[1]), device=action_head.device, dtype=torch.int64)
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)

            timestep_action = action_head.scheduler.timesteps[timestep_action_id].to(action_head.device)
            noisy_actions = action_head.scheduler.add_noise(actions.flatten(0,1), noise_action.flatten(0,1), timestep_action.flatten(0,1)).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))

            # Call the diffusion model similarly to training forward
            video_noise_pred, action_noise_pred = action_head.model(
                noisy_latents.transpose(1, 2),
                timestep=timestep,
                clip_feature=clip_feas,
                y=ys,
                context=None,
                seq_len=seq_len,
                state=action_inputs.get("state", None),
                embodiment_id=action_inputs.get("embodiment_id", None),
                action=noisy_actions,
                timestep_action=timestep_action,
                clean_x=latents.transpose(1, 2),
            )

            # Reconstruct predicted x0 for actions: x0 = noisy_actions - sigma * action_noise_pred
            # Map timestep_action_id -> sigma
            timestep_action_id_for_sigma = torch.argmin((action_head.scheduler.timesteps.unsqueeze(1) - timestep_action.flatten(0,1).unsqueeze(0)).abs(), dim=0)
            sigma_action = action_head.scheduler.sigmas[timestep_action_id_for_sigma].to(device=action_head.device, dtype=noisy_actions.dtype).unflatten(0, (noisy_actions.shape[0], noisy_actions.shape[1]))
            while len(sigma_action.shape) < len(noisy_actions.shape):
                sigma_action = sigma_action.unsqueeze(-1)

            action_x0_pred = noisy_actions - sigma_action * action_noise_pred

            # Extract first horizon step prediction vector and GT flat vector
            pred_vec = action_x0_pred[0, 0].detach().cpu().numpy()
            gt_vec = action_array[0, 0]

            # Split into per-key slices and record
            gt_dict = get_gt_action_dict(dataset, idx)
            for key in ACTION_KEY_ORDER:
                start, end = ACTION_SLICES[key]
                pred_slice = pred_vec[start:end]
                gt_slice = gt_dict[key]
                preds_per_key[key].append(pred_slice)
                gts_per_key[key].append(gt_slice)

        # Record timing and a small preview
        print(f"[{i+1}/{num}] idx={idx} pred_first[:5]={pred_vec[:5]} gt_first[:5]={gt_vec[:5]}")

    # After loop: summarize and save plots
    if sum(len(v) for v in preds_per_key.values()) == 0:
        print("No predictions recorded.")
        return

    valid_keys = [k for k in ACTION_KEY_ORDER if len(preds_per_key[k]) > 0]
    stacked_preds = {k: np.stack(preds_per_key[k]) for k in valid_keys}
    stacked_gts = {k: np.stack(gts_per_key[k]) for k in valid_keys}

    pred_all = np.concatenate([stacked_preds[k] for k in valid_keys], axis=-1)
    gt_all = np.concatenate([stacked_gts[k] for k in valid_keys], axis=-1)
    overall_mse = float(np.mean((pred_all - gt_all) ** 2))

    print(f"\n{'='*60}")
    print(f"Overall MSE: {overall_mse:.6f}  |  Avg inference time: {np.mean(times) if times else 0.0:.4f}s")
    for key in valid_keys:
        key_mse = float(np.mean((stacked_preds[key] - stacked_gts[key]) ** 2))
        print(f"  {key}: MSE={key_mse:.6f}")
    print(f"{'='*60}")

    mse_dim, overall_mse_dim = save_plots(stacked_preds, stacked_gts, valid_keys, args.output_dir)

    with open(os.path.join(args.output_dir, "mse.txt"), "w") as f:
        f.write(f"overall_mse,{overall_mse}\n")
        for key in valid_keys:
            key_mse = float(np.mean((stacked_preds[key] - stacked_gts[key]) ** 2))
            f.write(f"{key},{key_mse}\n")
        for dim_idx, value in enumerate(mse_dim):
            f.write(f"dim_{dim_idx},{value}\n")

    print(f"Results saved to {os.path.abspath(args.output_dir)}/")


def main():
    parser = make_arg_parser(default_output_dir="./open_loop_trainstyle_out", include_context_length=True)
    args = parser.parse_args()
    run_trainstyle(args, context_length=args.context_length if hasattr(args, 'context_length') else 33)


if __name__ == "__main__":
    main()
