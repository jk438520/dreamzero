#!/usr/bin/env python3
"""
Adds `action.value_function` to a LeRobot dataset by modifying the parquet files.

Usage:
  python scripts/data/add_value_function.py \\
      --dataset-path ./Dataset/oxe_droid \\
      --output-path ./Dataset/oxe_droid_value
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def get_hardcoded_value_function_stats() -> dict[str, float]:
    """Return placeholder stats for value_function; can be replaced by recomputation later."""
    return {
        "mean": 0.5,
        "std": 0.288675,  # ~sqrt(1/12)
        "min": 0.0,
        "max": 1.0,
        "q01": 0.01,
        "q99": 0.99,
    }


def load_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        log.error("meta/info.json not found at %s", info_path)
        sys.exit(1)
    with open(info_path) as f:
        return json.load(f)


def save_info(info: dict, dataset_path: Path):
    with open(dataset_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)


def get_parquet_paths(dataset_path: Path, info: dict) -> list[Path]:
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    paths = []
    for ep_idx in range(total_episodes):
        chunk_idx = ep_idx // chunks_size
        p = dataset_path / pattern.format(episode_chunk=chunk_idx, episode_index=ep_idx)
        if p.exists():
            paths.append(p)
    return sorted(paths)


def _is_existing_value_function_tail(action_data: np.ndarray) -> bool:
    """Detect whether the last action dim already matches expected t/T progress."""
    if action_data.ndim != 2 or action_data.shape[0] == 0 or action_data.shape[1] == 0:
        return False
    num_frames = action_data.shape[0]
    expected = np.arange(num_frames, dtype=np.float64) / float(num_frames)
    return np.allclose(action_data[:, -1].astype(np.float64), expected, rtol=1e-6, atol=1e-6)


def process_parquet(path: Path):
    """Adds a value_function to the action array of the parquet file."""
    df = pd.read_parquet(path)
    
    if "action" not in df.columns:
        log.warning(f"No 'action' column in {path}")
        return
    
    # action is usually a column of lists / arrays
    action_data = np.stack(df["action"].values)
    if action_data.ndim == 1:
        action_data = action_data.reshape(-1, 1)
        
    num_frames, action_dim = action_data.shape
    
    # Idempotency guard: if the tail already looks like t/T, skip this parquet.
    if _is_existing_value_function_tail(action_data):
        log.info("Skipping already-processed parquet: %s", path)
        return action_dim - 1

    # Calculate value function as t/T where t in [0, T-1].
    value_function = (np.arange(num_frames, dtype=np.float64) / float(num_frames)).reshape(-1, 1)
    value_function = value_function.astype(action_data.dtype, copy=False)
        
    # Append value_function as the last dimension
    new_action_data = np.concatenate([action_data, value_function], axis=1)
    
    # Replace action column
    df["action"] = list(new_action_data)
    
    # Save back to parquet
    df.to_parquet(path, index=False)
    
    return action_dim

def update_metadata(meta_dir: Path, original_action_dim: int):
    # 1. Update info.json
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
            
        if "action" in info.get("features", {}):
            old_shape = info["features"]["action"].get("shape", [original_action_dim])
            if old_shape[0] == original_action_dim:
                info["features"]["action"]["shape"] = [original_action_dim + 1]
                
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=4)
                log.info(f"Updated info.json action shape to [{original_action_dim + 1}]")

    # 2. Update modality.json
    modality_path = meta_dir / "modality.json"
    if modality_path.exists():
        with open(modality_path) as f:
            modality = json.load(f)
            
        if "action" in modality:
            if "value_function" in modality["action"]:
                log.info("modality.json already contains action.value_function. Skipping modality update.")
            else:
                modality["action"]["value_function"] = {
                    "original_key": "action",
                    "start": original_action_dim,
                    "end": original_action_dim + 1,
                    "rotation_type": None,
                    "absolute": True,
                    "dtype": "float64",
                    "range": None
                }
                with open(modality_path, "w") as f:
                    json.dump(modality, f, indent=4)
                log.info("Updated modality.json to include action.value_function")

    # 3. Update stats.json (simply append uniform stats for value_function)
    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
            
        if "action" in stats:
            value_stats = get_hardcoded_value_function_stats()
            
            for key, val in zip(
                ["mean", "std", "min", "max", "q01", "q99"],
                [
                    value_stats["mean"],
                    value_stats["std"],
                    value_stats["min"],
                    value_stats["max"],
                    value_stats["q01"],
                    value_stats["q99"],
                ]
            ):
                if isinstance(stats["action"][key], list):
                    # Append if it's a list
                    if len(stats["action"][key]) == original_action_dim:
                        stats["action"][key].append(val)
                elif isinstance(stats["action"][key], (float, int)):
                    pass # Handled differently if not a list
                    
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
            log.info("Updated stats.json for action dimensionality increase.")


def main():
    parser = argparse.ArgumentParser(description="Add value_function to dataset actions.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output-path", type=str, required=True, help="Output dataset path")
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copy/symlink and process an already-copied dataset in --output-path.",
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path).resolve()
    output_path = Path(args.output_path).resolve()
    
    if not dataset_path.exists():
        log.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)
        
    if args.skip_copy:
        if not output_path.exists():
            log.error("--skip-copy was set, but output path does not exist: %s", output_path)
            sys.exit(1)
        log.info("Skipping copy step and operating on existing output dataset.")
    else:
        if output_path.exists():
            log.error("Output path already exists. Please point to a new path to avoid accidental mutations.")
            sys.exit(1)

        output_path.mkdir(parents=True)

        # Symlink videos dir to save space/time, copy everything else
        for item in os.listdir(dataset_path):
            src = dataset_path / item
            dst = output_path / item
            if item == "videos":
                os.symlink(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        log.info("Base dataset copy/symlinked successfully.")
    
    info = load_info(output_path)
    parquet_paths = get_parquet_paths(output_path, info)
    
    if not parquet_paths:
        log.error("No parquet files found in dataset.")
        sys.exit(1)

    modality_path = output_path / "meta" / "modality.json"
    if modality_path.exists():
        with open(modality_path) as f:
            modality = json.load(f)
        if "action" in modality and "value_function" in modality["action"]:
            log.info("Dataset already contains action.value_function in metadata. Nothing to do.")
            return
        
    log.info(f"Processing {len(parquet_paths)} parquet files to add value function...")
    
    original_dims = set()
    for p in tqdm(parquet_paths, desc="Updating parquets"):
        dim = process_parquet(p)
        if dim is not None:
            original_dims.add(dim)
            
    if len(original_dims) == 0:
        log.error("No valid action shapes found in parquet files.")
        sys.exit(1)
    elif len(original_dims) > 1:
        log.warning(f"Multiple original action dimensions found: {original_dims}. This is unusual.")
        
    original_action_dim = list(original_dims)[0]
    log.info(f"Found original action dimensionality: {original_action_dim}. Updating metadata...")
    
    update_metadata(output_path / "meta", original_action_dim)
    
    log.info("Success! Dataset is ready with value_function.")

if __name__ == "__main__":
    main()
