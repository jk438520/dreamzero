#!/bin/sh

set -eu

MODEL_PATH="${MODEL_PATH:-./checkpoints/dreamzero_droid_finetune_full_1_traj_with_value_function_v3}"
DATASET_PATH="${DATASET_PATH:-./data/droid_lerobot_1_traj_with_value_function}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
DEVICE="${DEVICE:-cuda}"
PROMPT="${PROMPT:-pick up the object}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_droid_vf_long_context_big_lora}"
# Use a large default; the Python eval loop clamps to the dataset length,
# so this evaluates the full episode unless you override NUM_SAMPLES.
NUM_SAMPLES="${NUM_SAMPLES:-999}"
START_IDX="${START_IDX:-0}"
LOG_EVERY="${LOG_EVERY:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-33}"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"

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