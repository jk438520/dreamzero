#!/usr/bin/env python3
"""Evaluate value-function predictions via websocket inference server.

This script:
1. Reads LeRobot metadata and episodes.
2. Streams episode frames to the websocket policy server with the same schedule as test_client_AR.py.
3. Extracts predicted value_function (from action.value_function or action tail).
4. Saves per-episode CSV + plot and a global summary CSV.

Usage:
  python scripts/inference/eval_value_function.py \
      --dataset-path ./data/droid_lerobot_value \
      --host 127.0.0.1 \
      --port 8000 \
      --output-dir ./logs/value_function_validation
"""

import argparse
import json
import logging
import os
import uuid
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_utils.policy_client import WebsocketClientPolicy
from eval_utils.policy_server import PolicyServerConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Matches test_client_AR.py and AR inference schedule.
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24

OBS_TO_VIDEO_KEY = {
    "observation/exterior_image_0_left": "video.exterior_image_1_left",
    "observation/exterior_image_1_left": "video.exterior_image_2_left",
    "observation/wrist_image_left": "video.wrist_image_left",
}


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_task_map(dataset_path: Path) -> dict[int, str]:
    """Load task_index -> task text mapping from meta/tasks.jsonl."""
    task_map: dict[int, str] = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return task_map
    for row in read_jsonl(tasks_path):
        if "task_index" not in row or "task" not in row:
            continue
        try:
            task_idx = int(row["task_index"])
        except (TypeError, ValueError):
            continue
        task_text = str(row["task"]).strip()
        if task_text:
            task_map[task_idx] = task_text
    log.info("Loaded %d task mappings from %s", len(task_map), tasks_path)
    return task_map


def get_episode_indices(dataset_path: Path, info: dict) -> list[int]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        rows = read_jsonl(episodes_path)
        indices = sorted({int(row["episode_index"]) for row in rows if "episode_index" in row})
        if indices:
            return indices
    return list(range(int(info["total_episodes"])))


def get_parquet_path(dataset_path: Path, info: dict, episode_index: int) -> Path:
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    return dataset_path / pattern.format(episode_chunk=episode_chunk, episode_index=episode_index)


def get_video_path(
    dataset_path: Path,
    info: dict,
    modality: dict,
    episode_index: int,
    droid_video_key: str,
) -> Path:
    video_pattern = info["video_path"]
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size

    subkey = droid_video_key.replace("video.", "")
    meta = modality["video"][subkey]
    original_key = meta.get("original_key") or subkey

    return dataset_path / video_pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=original_key,
    )


def load_all_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def build_frame_schedule(total_frames: int, max_chunks: int | None = None) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23
    while True:
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        current_frame += ACTION_HORIZON
    return chunks


def build_observation(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    obs: dict = {}
    for obs_key, frames in camera_frames.items():
        selected = frames[frame_indices]
        if len(frame_indices) == 1:
            selected = selected[0]
        obs[obs_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def extract_gt_value(df: pd.DataFrame, modality: dict) -> np.ndarray:
    action_meta = modality.get("action", {})
    value_meta = action_meta.get("value_function")

    if value_meta is not None:
        original_key = value_meta.get("original_key", "action")
        start = int(value_meta["start"])
        end = int(value_meta["end"])
        if original_key in df.columns:
            action_arr = np.stack(df[original_key].values)
            value = action_arr[:, start:end]
            return value.reshape(len(df))

    if "frame_index" in df.columns:
        frame_index = df["frame_index"].to_numpy(dtype=np.float64)
        total = float(max(len(df), 1))
        return frame_index / total

    return np.arange(len(df), dtype=np.float64) / float(max(len(df), 1))


def infer_episode_prompt(
    df: pd.DataFrame,
    modality: dict,
    task_map: dict[int, str],
    fallback_prompt: str = "",
) -> str:
    """Infer a language prompt from annotation metadata/columns for one episode."""
    # Preferred path: dataset provides task_index and we resolve via meta/tasks.jsonl.
    if "task_index" in df.columns and len(task_map) > 0:
        task_indices = pd.to_numeric(df["task_index"], errors="coerce").dropna()
        if not task_indices.empty:
            first_task_idx = int(task_indices.iloc[0])
            mapped = task_map.get(first_task_idx, "").strip()
            if mapped:
                log.info("Inferred prompt from task_index=%d: %s", first_task_idx, mapped)
                return mapped

    candidates: list[str] = []

    # Prefer annotation keys declared in modality metadata.
    annotation_meta = modality.get("annotation", {})
    for _, meta in annotation_meta.items():
        original_key = meta.get("original_key")
        if isinstance(original_key, str) and original_key in df.columns:
            candidates.append(original_key)

    # Heuristic fallback over likely language/task columns.
    for col in df.columns:
        col_lower = str(col).lower()
        if (
            col_lower.startswith("annotation")
            or "task" in col_lower
            or "language" in col_lower
            or "instruction" in col_lower
            or "prompt" in col_lower
        ):
            candidates.append(str(col))

    # Deduplicate while preserving order.
    seen = set()
    uniq_candidates: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq_candidates.append(c)

    for col in uniq_candidates:
        series = df[col].dropna()
        if series.empty:
            continue
        # Use the first non-empty string in the episode.
        for val in series.tolist():
            text = str(val).strip()
            if text:
                log.info("Inferred prompt from column %s: %s", col, text)
                return text

    if fallback_prompt:
        log.info("Using fallback prompt: %s", fallback_prompt)
    return fallback_prompt


def extract_pred_value(response: object) -> np.ndarray:
    # Case 1: dict with explicit action.value_function key.
    if isinstance(response, dict):
        if "action.value_function" in response:
            arr = np.asarray(response["action.value_function"], dtype=np.float64)
            return arr.reshape(-1)
        if "action" in response:
            arr = np.asarray(response["action"], dtype=np.float64)
            if arr.ndim == 1 and arr.shape[0] >= 9:
                return np.asarray([arr[8]], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 9:
                return arr[:, 8]

    # Case 2: plain action array from roboarena wrapper.
    arr = np.asarray(response, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] >= 9:
        return np.asarray([arr[8]], dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 9:
        return arr[:, 8]

    raise ValueError(
        "Could not extract value_function from server response. "
        "Expected action.value_function key or >=9 action dims."
    )


def save_episode_artifacts(
    episode_dir: Path,
    episode_index: int,
    anchors: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
) -> tuple[float, float]:
    episode_dir.mkdir(parents=True, exist_ok=True)

    abs_err = np.abs(gt - pred)
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(abs_err))

    df = pd.DataFrame(
        {
            "episode_index": episode_index,
            "query_index": np.arange(len(anchors), dtype=np.int64),
            "anchor_frame": anchors.astype(np.int64),
            "ground_truth_value": gt,
            "predicted_value": pred,
            "abs_error": abs_err,
        }
    )
    df.to_csv(episode_dir / "value_function_samples.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(anchors, gt, label="Ground Truth", lw=1.5)
    plt.plot(anchors, pred, label="Predicted", lw=1.5)
    plt.title(f"Episode {episode_index:06d} Value Function (MSE={mse:.6f}, MAE={mae:.6f})")
    plt.xlabel("Anchor Frame")
    plt.ylabel("Value Function")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(episode_dir / "value_function_plot.png", dpi=180)
    plt.close()

    return mse, mae


def evaluate_episode(
    dataset_path: Path,
    info: dict,
    modality: dict,
    client: WebsocketClientPolicy,
    server_config: PolicyServerConfig,
    task_map: dict[int, str],
    episode_index: int,
    output_dir: Path,
    fallback_prompt: str,
    max_chunks: int | None,
) -> dict:
    parquet_path = get_parquet_path(dataset_path, info, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found for episode {episode_index}: {parquet_path}")

    log.info("Episode %06d parquet: %s", episode_index, parquet_path)

    episode_df = pd.read_parquet(parquet_path)
    gt_value = extract_gt_value(episode_df, modality)
    prompt = infer_episode_prompt(
        episode_df,
        modality,
        task_map,
        fallback_prompt=fallback_prompt,
    )
    if not prompt:
        raise RuntimeError(
            f"Could not infer prompt from dataset for episode {episode_index}. "
            "Provide --fallback-prompt if your dataset has no language/task column."
        )
    log.info("Episode %06d prompt: %s", episode_index, prompt)

    camera_frames: dict[str, np.ndarray] = {}
    for obs_key, droid_key in OBS_TO_VIDEO_KEY.items():
        if obs_key.startswith("observation/wrist") and not server_config.needs_wrist_camera:
            continue
        path = get_video_path(dataset_path, info, modality, episode_index, droid_key)
        log.info("Episode %06d video %s -> %s", episode_index, obs_key, path)
        camera_frames[obs_key] = load_all_frames(path)

    total_frames = min(fr.shape[0] for fr in camera_frames.values())
    if total_frames == 0:
        raise RuntimeError(f"No video frames for episode {episode_index}")
    log.info("Episode %06d total_frames=%d", episode_index, total_frames)

    anchors: list[int] = [0]
    for ids in build_frame_schedule(total_frames, max_chunks=max_chunks):
        anchors.append(ids[-1])
    log.info("Episode %06d anchor frames: %s", episode_index, anchors)

    session_id = str(uuid.uuid4())
    pred_points: list[float] = []

    # Initial single-frame call.
    init_obs = build_observation(camera_frames, [0], prompt=prompt, session_id=session_id)
    init_resp = client.infer(init_obs)
    pred_points.append(float(extract_pred_value(init_resp)[0]))
    log.info("Episode %06d initial predicted value=%.6f", episode_index, pred_points[-1])

    # Chunked multi-frame calls.
    for frame_ids in build_frame_schedule(total_frames, max_chunks=max_chunks):
        obs = build_observation(camera_frames, frame_ids, prompt=prompt, session_id=session_id)
        resp = client.infer(obs)
        pred_val = float(extract_pred_value(resp)[0])
        pred_points.append(pred_val)
        log.info("Episode %06d chunk frames=%s predicted value=%.6f", episode_index, frame_ids, pred_val)

    # End current episode session.
    client.reset({})

    anchors_arr = np.asarray(anchors, dtype=np.int64)
    gt_arr = gt_value[anchors_arr]
    pred_arr = np.asarray(pred_points, dtype=np.float64)

    ep_dir = output_dir / f"episode_{episode_index:06d}"
    mse, mae = save_episode_artifacts(ep_dir, episode_index, anchors_arr, gt_arr, pred_arr)
    log.info("Episode %06d saved to %s", episode_index, ep_dir)

    return {
        "episode_index": episode_index,
        "num_points": int(len(pred_arr)),
        "mse": mse,
        "mae": mae,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate value_function predictions from websocket server.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="LeRobot dataset root path.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Inference server host.")
    parser.add_argument("--port", type=int, default=8000, help="Inference server port.")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/value_function_validation"))
    parser.add_argument(
        "--fallback-prompt",
        type=str,
        default="",
        help="Optional fallback prompt used only when dataset has no language/task text.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes.")
    parser.add_argument("--episode-start", type=int, default=0, help="Start index into sorted episode list.")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit chunked calls per episode after initial frame. Useful for fast smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = read_json(dataset_path / "meta" / "info.json")
    modality = read_json(dataset_path / "meta" / "modality.json")
    task_map = load_task_map(dataset_path)

    episode_indices = get_episode_indices(dataset_path, info)
    if args.episode_start > 0:
        episode_indices = episode_indices[args.episode_start :]
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    if not episode_indices:
        raise RuntimeError("No episodes found for evaluation")

    log.info("Connecting to server %s:%d", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    server_config = PolicyServerConfig(**metadata)
    log.info("Server config: %s", server_config)

    summary_rows: list[dict] = []
    for i, ep_idx in enumerate(episode_indices):
        log.info("[%d/%d] Evaluating episode %06d", i + 1, len(episode_indices), ep_idx)
        try:
            row = evaluate_episode(
                dataset_path=dataset_path,
                info=info,
                modality=modality,
                client=client,
                server_config=server_config,
                task_map=task_map,
                episode_index=ep_idx,
                output_dir=output_dir,
                fallback_prompt=args.fallback_prompt,
                max_chunks=args.max_chunks,
            )
            summary_rows.append(row)
            log.info(
                "Episode %06d done: points=%d mse=%.6f mae=%.6f",
                ep_idx,
                row["num_points"],
                row["mse"],
                row["mae"],
            )
        except Exception as e:
            log.exception("Failed episode %06d: %s", ep_idx, e)

    if not summary_rows:
        raise RuntimeError("All episodes failed; no summary generated")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    log.info("Saved summary: %s", output_dir / "summary.csv")
    log.info(
        "Aggregate: episodes=%d mean_mse=%.6f mean_mae=%.6f",
        len(summary_df),
        float(summary_df["mse"].mean()),
        float(summary_df["mae"].mean()),
    )


if __name__ == "__main__":
    main()
