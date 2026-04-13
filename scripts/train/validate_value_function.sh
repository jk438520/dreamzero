#!/bin/bash
# DreamZero Value-Function Validation Script
#
# Usage:
#   bash scripts/train/validate_value_function.sh
#
# Prerequisites:
#   - DreamZero checkpoint at MODEL_PATH
#   - LeRobot-format dataset at DATASET_PATH
#   - Run on 1 or more GPUs via torchrun; 2 GPUs is recommended for efficient model parallelism

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
MODEL_PATH=${MODEL_PATH:-"./checkpoints/dreamzero_droid_finetune_lora"}
DATASET_PATH=${DATASET_PATH:-"./data/droid_lerobot_10_traj_with_progress"}
OUTPUT_DIR=${OUTPUT_DIR:-"./logs/value_function_validation"}

# Number of GPUs to use (default: all visible GPUs, falling back to 2)
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-2}

DEVICE=${DEVICE:-"cuda"}
STEP_STRIDE=${STEP_STRIDE:-}
MAX_EPISODES=${MAX_EPISODES:-}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-}
VIDEO_BACKEND=${VIDEO_BACKEND:-"ffmpeg"}
# ===========================================

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: checkpoint directory not found at $MODEL_PATH"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: dataset directory not found at $DATASET_PATH"
    exit 1
fi

CMD=(torchrun --nproc_per_node "$NUM_GPUS" --standalone scripts/validate_value_function.py
    --model_path "$MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --output_dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --video_backend "$VIDEO_BACKEND"
)

if [ -n "$STEP_STRIDE" ]; then
    CMD+=(--step_stride "$STEP_STRIDE")
fi

if [ -n "$MAX_EPISODES" ]; then
    CMD+=(--max_episodes "$MAX_EPISODES")
fi

if [ -n "$EMBODIMENT_TAG" ]; then
    CMD+=(--embodiment_tag "$EMBODIMENT_TAG")
fi

"${CMD[@]}"
