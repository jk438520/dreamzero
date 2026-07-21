#!/usr/bin/env bash

if [ -n "${TRAIN_LOG_FILE:-}" ]; then
	mkdir -p "$(dirname "$TRAIN_LOG_FILE")"
	exec > >(tee -a "$TRAIN_LOG_FILE") 2>&1
fi

export DROID_DATA_ROOT="./data/droid_lerobot_1_traj_with_value_function"
export OUTPUT_DIR="./checkpoints/dreamzero_droid_finetune_full_1_traj_with_value_function_full"
export NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=5,6,7

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Longer low-LR tail profile
export LEARNING_RATE=8e-5
export MAX_STEPS=2000
export WARMUP_RATIO=0.01
export LR_SCHEDULER_TYPE=polynomial
export LR_END=1e-6
export LR_POWER=3.0
export SAVE_STEPS=400

# Optional: start from your latest run checkpoint/model dir
# export PRETRAINED_MODEL_PATH="./checkpoints/dreamzero_droid_finetune_lora_10_traj_with_value_function"

./scripts/train/droid_training_value_finetune_full.sh