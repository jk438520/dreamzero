#!/bin/sh

set -eu

MODEL_PATH="${MODEL_PATH:-./checkpoints/dreamzero_droid_finetune_full_1_traj_with_value_function}"
DATASET_PATH="${DATASET_PATH:-./data/droid_lerobot_1_traj_with_value_function}"
DEVICE="${DEVICE:-cuda:7}"
PROMPT="${PROMPT:-pick up the object}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_droid_vf_long_context}"
# Use a large default; the Python eval loop clamps to the dataset length,
# so this evaluates the full episode unless you override NUM_SAMPLES.
NUM_SAMPLES="${NUM_SAMPLES:-999}"
START_IDX="${START_IDX:-0}"
LOG_EVERY="${LOG_EVERY:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-33}"

python scripts/open_loop_droid_vf_long_context.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --device "$DEVICE" \
    --use_dataset_prompt \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --start_idx "$START_IDX" \
    --log_every "$LOG_EVERY" \
    --context_length "$CONTEXT_LENGTH"