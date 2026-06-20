#!/usr/bin/env python3
"""Shared helpers for DROID value-function open-loop evaluation scripts."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.dataset.lerobot import CachedLeRobotSingleDataset
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


VIDEO_CAMERAS = {
    "observation/exterior_image_0_left": "video.exterior_image_1_left",
    "observation/exterior_image_1_left": "video.exterior_image_2_left",
    "observation/wrist_image_left": "video.wrist_image_left",
}

STATE_SLICES = {
    "state.joint_position": (0, 7),
    "state.gripper_position": (7, 8),
}

ACTION_SLICES = {
    "action.joint_position": (14, 21),
    "action.gripper_position": (12, 13),
    "action.value_function": (28, 29),
}

ACTION_KEY_ORDER = [
    "action.joint_position",
    "action.gripper_position",
    "action.value_function",
]

DEFAULT_PLACEHOLDER_FRAME_SHAPE = (256, 256, 3)

def build_droid_modality_configs() -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            eval_delta_indices=[0],
            modality_keys=[
                "video.exterior_image_1_left",
                "video.exterior_image_2_left",
                "video.wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.joint_position", "state.gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=["action.joint_position", "action.gripper_position", "action.value_function"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ],
        ),
    }


def make_droid_dataset(dataset_path: str):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"DROID dataset path '{dataset_path}' not found.\n"
            "Expected a dataset directory containing 'data/', 'videos/', and 'meta/'.\n"
            "If you moved or renamed the dataset, pass the correct --dataset_path or restore the dataset.\n"
            "Example: --dataset_path ./data/droid_lerobot_1_traj_with_value_function"
        )
    return CachedLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=build_droid_modality_configs(),
        embodiment_tag=EmbodimentTag.OXE_DROID,
        use_global_metadata=False,
        video_backend="decord",
        transforms=ComposedModalityTransform(transforms=[]),
        discard_bad_trajectories=True,
        relative_action=False,
        relative_action_keys=["joint_position"],
        relative_action_per_horizon=False,
    )


def build_obs(dataset, idx: int, prompt: str, context_length: int = 1) -> dict:
    """Build an obs dict using the LeRobot dataset implementation."""
    trajectory_id, base_index = dataset.all_steps[idx]
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    dataset.curr_traj_id = trajectory_id

    obs: dict = {}
    step_indices = np.arange(base_index - context_length + 1, base_index + 1)

    for key in VIDEO_CAMERAS.values():
        frames = dataset.get_video(trajectory_id, key, step_indices)
        # Ensure a minimal temporal length so downstream VAE/conv3d kernels are valid
        min_frames = 4
        if frames.shape[0] < min_frames:
            pad_count = min_frames - frames.shape[0]
            last = np.repeat(frames[-1:][np.newaxis, ...], pad_count, axis=0)
            frames = np.concatenate([frames, last], axis=0)
        obs[key] = frames.astype(np.uint8)

    for key in STATE_SLICES:
        values = dataset.get_data_by_modality(trajectory_id, "state", key, np.array([base_index]))
        obs[key] = np.asarray(values)

    for key in [
        "annotation.language.language_instruction",
        "annotation.language.language_instruction_2",
        "annotation.language.language_instruction_3",
    ]:
        obs[key] = [prompt]

    return obs


def get_gt_action_dict(dataset, idx: int) -> dict:
    """Split the flat GT action vector into per-key arrays."""
    trajectory_id, base_index = dataset.all_steps[idx]
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    dataset.curr_traj_id = trajectory_id
    gt = {}
    for key in ACTION_KEY_ORDER:
        values = dataset.get_data_by_modality(trajectory_id, "action", key, np.array([base_index]))
        gt[key] = np.asarray(values)[0]
    return gt


def get_dataset_prompt(dataset, idx: int) -> str:
    trajectory_id, base_index = dataset.all_steps[idx]
    traj_data = dataset.get_trajectory_data(trajectory_id)
    if "annotation.task" in traj_data.columns:
        value = traj_data["annotation.task"].iloc[base_index]
        if isinstance(value, str) and value.strip():
            return value
    if "task_index" in traj_data.columns:
        try:
            task_index = int(traj_data["task_index"].iloc[base_index])
            if task_index in dataset.tasks.index:
                value = dataset.tasks.loc[task_index]["task"]
                if isinstance(value, str) and value.strip():
                    return value
        except Exception:
            pass
    return "pick up the object"


def save_plots(all_preds, all_gts, key_names, output_dir):
    """Plot pred vs gt for each action dimension across all keys."""
    pred_flat = np.concatenate([all_preds[k] for k in key_names], axis=-1)
    gt_flat = np.concatenate([all_gts[k] for k in key_names], axis=-1)
    dim_count = pred_flat.shape[1]
    mse_dim = np.mean((pred_flat - gt_flat) ** 2, axis=0)

    for dim_idx in range(dim_count):
        plt.figure(figsize=(10, 4))
        plt.plot(gt_flat[:, dim_idx], label="gt", alpha=0.8, marker="o", markersize=3)
        plt.plot(pred_flat[:, dim_idx], label="pred", alpha=0.8, marker="o", markersize=3)
        plt.title(f"Action dim {dim_idx}  (MSE={mse_dim[dim_idx]:.6f})")
        plt.xlabel("sample index")
        plt.ylabel("value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"action_dim_{dim_idx}.png"), dpi=150)
        plt.close()

    ncols = 4
    nrows = (dim_count + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    overall_mse = float(np.mean(mse_dim))
    fig.suptitle(f"All action dims  (overall MSE={overall_mse:.6f})", fontsize=14)
    for dim_idx in range(dim_count):
        ax = axes[dim_idx // ncols][dim_idx % ncols]
        ax.plot(gt_flat[:, dim_idx], label="gt", alpha=0.7, lw=0.8, marker="o", markersize=2)
        ax.plot(pred_flat[:, dim_idx], label="pred", alpha=0.7, lw=0.8, marker="o", markersize=2)
        ax.set_title(f"dim {dim_idx} (MSE={mse_dim[dim_idx]:.4f})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if dim_idx == 0:
            ax.legend(fontsize=7)
    for dim_idx in range(dim_count, nrows * ncols):
        axes[dim_idx // ncols][dim_idx % ncols].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(output_dir, "all_action_dims.png"), dpi=200)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, len(key_names), figsize=(5 * len(key_names), 4), squeeze=False)
    for key_index, key_name in enumerate(key_names):
        ax = axes2[0][key_index]
        pred_vals, gt_vals = all_preds[key_name], all_gts[key_name]
        for dim_idx in range(pred_vals.shape[1]):
            ax.plot(gt_vals[:, dim_idx], "--", alpha=0.5, lw=0.8, marker="o", markersize=2)
            ax.plot(pred_vals[:, dim_idx], alpha=0.7, lw=0.8, marker="o", markersize=2)
        key_mse = float(np.mean((pred_vals - gt_vals) ** 2))
        ax.set_title(f"{key_name}\nMSE={key_mse:.6f}", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    fig2.suptitle("Per-key pred (solid) vs gt (dashed)", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    fig2.savefig(os.path.join(output_dir, "per_key_summary.png"), dpi=200)
    plt.close(fig2)

    return mse_dim, overall_mse


def make_policy(model_path: str, device: str):
    return GrootSimPolicy(
        embodiment_tag=EmbodimentTag.OXE_DROID,
        model_path=model_path,
        device=device,
    )


def evaluate(args, context_length: int = 1):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    print(f"Loading model from {args.model_path} ...")
    policy = make_policy(args.model_path, args.device)
    print("Model loaded.")

    dataset = make_droid_dataset(args.dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)

    num = min(args.num_samples, len(dataset))
    preds_per_key = {k: [] for k in ACTION_KEY_ORDER}
    gts_per_key = {k: [] for k in ACTION_KEY_ORDER}
    times = []

    print(f"\nEvaluating {num} samples (start={args.start_idx}, context_length={context_length}) ...")
    print("-" * 60)

    for i in range(num):
        idx = args.start_idx + i

        prompt = args.prompt
        if args.use_dataset_prompt:
            prompt = get_dataset_prompt(dataset, idx)

        obs = build_obs(dataset, idx, prompt, context_length=context_length)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        gt = get_gt_action_dict(dataset, idx)

        for key in ACTION_KEY_ORDER:
            if key in result.act:
                pred_val = result.act[key]
                if isinstance(pred_val, torch.Tensor):
                    pred_val = pred_val.cpu().numpy()
                # Policy output is horizon-shaped; compare the first predicted step here.
                pred_val = np.atleast_1d(pred_val[0]).flatten()
                preds_per_key[key].append(pred_val)
                gts_per_key[key].append(gt[key])

        if i % args.log_every == 0:
            if i == 0:
                print(f"  Action keys in output: {list(result.act.keys())}")
                for key in ACTION_KEY_ORDER:
                    if key in result.act:
                        value = result.act[key]
                        shape = value.shape if hasattr(value, "shape") else "?"
                        print(f"    {key}: pred_shape={shape}, gt_shape={gt[key].shape}")
            print(f"  [{i:>5d}/{num}] idx={idx} infer={elapsed:.3f}s prompt={repr(prompt)[:60]}")

        if hasattr(policy.trained_model, "action_head") and hasattr(policy.trained_model.action_head, "clear_kv_cache"):
            policy.trained_model.action_head.clear_kv_cache()

    valid_keys = [k for k in ACTION_KEY_ORDER if len(preds_per_key[k]) > 0]
    if not valid_keys:
        print("No predictions!")
        return

    stacked_preds = {k: np.stack(preds_per_key[k]) for k in valid_keys}
    stacked_gts = {k: np.stack(gts_per_key[k]) for k in valid_keys}

    pred_all = np.concatenate([stacked_preds[k] for k in valid_keys], axis=-1)
    gt_all = np.concatenate([stacked_gts[k] for k in valid_keys], axis=-1)
    overall_mse = float(np.mean((pred_all - gt_all) ** 2))

    print(f"\n{'='*60}")
    print(f"Overall MSE: {overall_mse:.6f}  |  Avg inference time: {np.mean(times):.4f}s")
    for key in valid_keys:
        key_mse = float(np.mean((stacked_preds[key] - stacked_gts[key]) ** 2))
        print(f"  {key}: MSE={key_mse:.6f}")
    print(f"{'='*60}")

    mse_dim, _ = save_plots(stacked_preds, stacked_gts, valid_keys, args.output_dir)

    with open(os.path.join(args.output_dir, "mse.txt"), "w") as f:
        f.write(f"overall_mse,{overall_mse}\n")
        for key in valid_keys:
            key_mse = float(np.mean((stacked_preds[key] - stacked_gts[key]) ** 2))
            f.write(f"{key},{key_mse}\n")
        for dim_idx, value in enumerate(mse_dim):
            f.write(f"dim_{dim_idx},{value}\n")

    print(f"Results saved to {os.path.abspath(args.output_dir)}/")


def make_arg_parser(default_output_dir: str, include_context_length: bool = False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model checkpoint dir (contains config.json, model.safetensors, experiment_cfg/)",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Root of the DROID dataset (contains data/, videos/, meta/)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument(
        "--use_dataset_prompt",
        action="store_true",
        help="Read task annotation from parquet instead of --prompt",
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--log_every", type=int, default=10)
    if include_context_length:
        parser.add_argument(
            "--context_length",
            type=int,
            default=33,
            help="Number of frames to include per observation window for the long-context variant.",
        )
    return parser
