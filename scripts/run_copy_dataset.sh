#!/bin/sh

# Script to copy the DROID dataset with value function annotations
# Usage: bash scripts/run_copy_dataset.sh
# This will create a new dataset at ./data/droid_lerobot_1_traj
# with value function annotations added by scripts/data/add_value_function.py

# Create new dataset directory
SOURCE_DIR="./data/droid_lerobot_1_traj"
DEST_DIR="./data/droid_lerobot_1_traj_with_zeros"

python scripts/data/add_value_function.py \
    --dataset-path "$SOURCE_DIR" \
    --output-path "$DEST_DIR"