#!/bin/sh

export DROID_DATA_ROOT="./data/droid_lerobot_1_traj_with_value_function"
export OUTPUT_DIR="./checkpoints/dreamzero_droid_finetune_lora_1_traj_with_value_function"
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=4,7


./scripts/train/droid_training_value_finetune.sh