#!/bin/bash
# Script to run the Value Function evaluation on DROID 10_traj dataset

LOG_FILE="eval_value_$(date +%Y%m%d_%H%M%S).log"

echo "Starting evaluation... Logging to $LOG_FILE"
# Run the evaluation script
CUDA_VISIBLE_DEVICES=7 python scripts/eval_value_function.py \
    --model_path checkpoints/dreamzero_droid_vfh_finetune_1_traj \
    --dataset_path ./data/droid_lerobot_1_traj \
    --num_episodes 1 2>&1 | tee "$LOG_FILE"

echo "Evaluation completed."
