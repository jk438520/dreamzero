#!/usr/bin/env python3
"""Create a DROID LeRobot dataset that yields exactly one train sample.

This keeps the first episode parquet/video files and writes a step filter that
allows only one step index from that episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_first_episode(episodes_path: Path) -> dict:
    with episodes_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"No episodes found in {episodes_path}")
    return json.loads(first_line)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_episode_parquet(source_dir: Path, target_dir: Path, episode_index: int) -> None:
    ep_name = f"episode_{episode_index:06d}.parquet"
    matches = list(source_dir.glob(f"data/chunk-*/{ep_name}"))
    if not matches:
        raise FileNotFoundError(f"Could not find parquet for {ep_name} under {source_dir / 'data'}")
    src = matches[0]
    rel = src.relative_to(source_dir)
    dst = target_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_episode_videos(source_dir: Path, target_dir: Path, episode_index: int) -> None:
    video_name = f"episode_{episode_index:06d}.mp4"
    for src in source_dir.glob(f"videos/chunk-*/**/{video_name}"):
        rel = src.relative_to(source_dir)
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_step_filter(target_meta_dir: Path, episode_index: int, episode_length: int, keep_step: int) -> None:
    if keep_step < 0 or keep_step >= episode_length:
        raise ValueError(
            f"keep-step {keep_step} is out of range for episode length {episode_length}"
        )
    filtered_indices = [i for i in range(episode_length) if i != keep_step]
    payload = {
        "episode_index": episode_index,
        "step_indices": filtered_indices,
    }
    with (target_meta_dir / "step_filter.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create single-batch DROID LeRobot dataset")
    parser.add_argument("--source", default="data/droid_lerobot", help="Source LeRobot dataset path")
    parser.add_argument(
        "--target",
        default="data/droid_lerobot_single_batch",
        help="Target path for one-batch dataset",
    )
    parser.add_argument(
        "--keep-step",
        type=int,
        default=0,
        help="Step index to keep from first episode (all other steps filtered out)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    source_meta = source_dir / "meta"
    target_meta = target_dir / "meta"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if not source_meta.exists():
        raise FileNotFoundError(f"Missing meta directory: {source_meta}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_meta.mkdir(parents=True, exist_ok=True)

    first_episode = read_first_episode(source_meta / "episodes.jsonl")
    episode_index = int(first_episode["episode_index"])
    episode_length = int(first_episode["length"])

    # Keep first episode metadata only.
    with (target_meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(first_episode) + "\n")

    # Copy all other meta files; update info.json totals for clarity.
    for meta_file in source_meta.iterdir():
        if not meta_file.is_file() or meta_file.name == "episodes.jsonl":
            continue
        dst = target_meta / meta_file.name
        if meta_file.name == "info.json":
            with meta_file.open("r", encoding="utf-8") as f:
                info = json.load(f)
            info["total_episodes"] = 1
            info["total_frames"] = episode_length
            info["total_chunks"] = 1
            with dst.open("w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
        else:
            shutil.copy2(meta_file, dst)

    copy_episode_parquet(source_dir, target_dir, episode_index)
    copy_episode_videos(source_dir, target_dir, episode_index)
    write_step_filter(target_meta, episode_index, episode_length, args.keep_step)

    print(f"Created single-batch dataset at: {target_dir}")
    print(f"Episode index: {episode_index}, length: {episode_length}, kept step: {args.keep_step}")


if __name__ == "__main__":
    main()
