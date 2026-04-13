#!/usr/bin/env python3
"""Offline validation for DreamZero value-function predictions.

The script loads a checkpoint, reads a LeRobot-format dataset, runs inference
through the model's causal path, and writes one plot per episode comparing the
ground-truth and predicted value functions.

For a compact episode-level diagnostic, the value-function horizon is reduced to
a scalar by averaging across the horizon and feature dimensions at each sampled
step. Raw per-step scalar values are also written to CSV.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch._dynamo
from omegaconf import OmegaConf
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

from groot.vla.data.dataset.lerobot import LeRobotSingleDataset
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


torch._dynamo.config.disable = True


def _init_distributed() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def _create_device_mesh():
    if not dist.is_initialized():
        _init_distributed()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if world_size <= 1:
        return None

    return init_device_mesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )


def _infer_embodiment_tag(train_cfg, explicit_tag: str | None) -> EmbodimentTag:
    if explicit_tag is not None:
        return EmbodimentTag(explicit_tag)

    transform_keys = list(getattr(train_cfg, "transforms", {}).keys())
    if len(transform_keys) == 1:
        return EmbodimentTag(transform_keys[0])

    modality_keys = list(getattr(train_cfg, "modality_configs", {}).keys())
    if len(modality_keys) == 1:
        return EmbodimentTag(modality_keys[0])

    raise ValueError(
        "Could not infer embodiment tag from the checkpoint config. "
        "Pass --embodiment_tag explicitly."
    )


def _unsqueeze_values(data: dict) -> dict:
    batched = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            batched[key] = value.unsqueeze(0)
        elif isinstance(value, np.ndarray):
            batched[key] = np.expand_dims(value, axis=0)
        else:
            batched[key] = value
    return batched


def _is_batched(data: dict) -> bool:
    for value in data.values():
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return value.ndim > 0 and value.shape[0] == 1
    return False


def _collapse_value(value) -> float:
    if torch.is_tensor(value):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    return float(array.mean())


def _prepare_model_input(policy: GrootSimPolicy, obs: dict) -> dict:
    batch = Batch(obs=_unsqueeze_values(obs) if not _is_batched(obs) else obs)
    batch = policy.apply(batch)
    normalized_input = batch.normalized_obs
    if isinstance(normalized_input, Batch):
        normalized_input = normalized_input.__getstate__()

    if policy.eval_bf16:
        for key, value in list(normalized_input.items()):
            if torch.is_tensor(value) and value.dtype == torch.float32:
                normalized_input[key] = value.to(dtype=torch.bfloat16)

    return normalized_input


def _reset_causal_state(policy: GrootSimPolicy) -> None:
    action_head = policy.trained_model.action_head
    for attr in (
        "current_start_frame",
        "kv_cache1",
        "kv_cache_neg",
        "crossattn_cache",
        "crossattn_cache_neg",
        "clip_feas",
        "ys",
    ):
        if hasattr(action_head, attr):
            setattr(action_head, attr, 0 if attr == "current_start_frame" else None)


def _plot_episode(df: pd.DataFrame, output_dir: Path, episode_id: int) -> None:
    episode_dir = output_dir / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(episode_dir / "value_function_samples.csv", index=False)

    mse = float(np.mean((df["predicted_value"].to_numpy() - df["target_value"].to_numpy()) ** 2))
    mae = float(np.mean(np.abs(df["predicted_value"].to_numpy() - df["target_value"].to_numpy())))

    plt.figure(figsize=(12, 4))
    plt.plot(df["step_index"], df["target_value"], label="target", linewidth=1.5)
    plt.plot(df["step_index"], df["predicted_value"], label="prediction", linewidth=1.5)
    plt.title(f"Episode {episode_id:06d}  MSE={mse:.6f}  MAE={mae:.6f}")
    plt.xlabel("step index")
    plt.ylabel("collapsed value function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(episode_dir / "value_function_plot.png", dpi=160)
    plt.close()


def evaluate(args) -> None:
    _init_distributed()
    device_mesh = _create_device_mesh()
    rank = dist.get_rank() if dist.is_initialized() else 0

    train_cfg_path = Path(args.model_path) / "experiment_cfg" / "conf.yaml"
    train_cfg = OmegaConf.load(train_cfg_path)
    embodiment_tag = _infer_embodiment_tag(train_cfg, args.embodiment_tag)

    policy = GrootSimPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        device_mesh=device_mesh,
    )

    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_configs,
        embodiment_tag=policy.embodiment_tag,
        video_backend=args.video_backend,
        transforms=None,
        use_global_metadata=False,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path) / "value_function_validation"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    action_horizon = int(policy.trained_model.action_head.action_horizon)
    step_stride = args.step_stride if args.step_stride is not None else action_horizon
    num_frame_per_block = int(policy.trained_model.action_head.num_frame_per_block)

    episode_ids = list(dataset.trajectory_ids)
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    summary_rows = []

    for episode_id in episode_ids:
        traj_index = dataset.get_trajectory_index(int(episode_id))
        traj_len = int(dataset.trajectory_lengths[traj_index])
        latent_video = None
        _reset_causal_state(policy)

        rows = []
        for step_index in range(0, traj_len, step_stride):
            indices = {
                key: np.clip(delta_indices + step_index, 0, traj_len - 1)
                for key, delta_indices in dataset.delta_indices.items()
            }
            sample = dataset.get_step_data(int(episode_id), indices)
            normalized_input = _prepare_model_input(policy, sample)

            with torch.inference_mode():
                model_pred = policy.trained_model.lazy_joint_video_action_causal(
                    normalized_input,
                    latent_video=latent_video,
                )

            predicted_value = _collapse_value(model_pred["value_function_pred"])
            target_value = _collapse_value(sample["value_function"])
            rows.append(
                {
                    "step_index": int(step_index),
                    "target_value": target_value,
                    "predicted_value": predicted_value,
                    "abs_error": abs(predicted_value - target_value),
                }
            )

            video_pred = model_pred.get("video_pred")
            if torch.is_tensor(video_pred):
                latent_video = video_pred[:, :, -num_frame_per_block:]
            else:
                latent_video = None

        episode_df = pd.DataFrame(rows)
        if rank == 0:
            _plot_episode(episode_df, output_dir, int(episode_id))

        summary_rows.append(
            {
                "episode_id": int(episode_id),
                "num_samples": len(rows),
                "mse": float(np.mean((episode_df["predicted_value"] - episode_df["target_value"]) ** 2)),
                "mae": float(np.mean(np.abs(episode_df["predicted_value"] - episode_df["target_value"]))),
            }
        )

        if rank == 0:
            print(
                f"episode {int(episode_id):06d}: samples={len(rows)} "
                f"mse={summary_rows[-1]['mse']:.6f} mae={summary_rows[-1]['mae']:.6f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    if rank == 0:
        summary_df.to_csv(output_dir / "summary.csv", index=False)

    if rank == 0 and not summary_df.empty:
        print(
            f"overall: episodes={len(summary_df)} "
            f"mse={summary_df['mse'].mean():.6f} mae={summary_df['mae'].mean():.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DreamZero value-function predictions offline.")
    parser.add_argument("--model_path", required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--dataset_path", required=True, help="Path to the LeRobot-format dataset.")
    parser.add_argument("--output_dir", default=None, help="Directory for plots and CSV files.")
    parser.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--step_stride", type=int, default=None, help="Step spacing between inference calls. Defaults to the action horizon.")
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional cap on the number of episodes to evaluate.")
    parser.add_argument("--embodiment_tag", default=None, help="Explicit embodiment tag if it cannot be inferred from the checkpoint.")
    parser.add_argument("--video_backend", default="ffmpeg", help="Video backend used by the dataset loader.")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()