#!/bin/sh

set -eu

MODEL_PATH="${MODEL_PATH:-./checkpoints/replace_with_your_checkpoint}"
DATASET_PATH="${DATASET_PATH:-./data/replace_with_your_yam_dataset}"
DEVICE="${DEVICE:-cuda:0}"
PROMPT="${PROMPT:-pick up the object}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_yam}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
START_IDX="${START_IDX:-0}"
LOG_EVERY="${LOG_EVERY:-10}"

python scripts/open_loop_yam.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --device "$DEVICE" \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --start_idx "$START_IDX" \
    --log_every "$LOG_EVERY"