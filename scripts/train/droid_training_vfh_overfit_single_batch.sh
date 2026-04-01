#!/bin/bash
# Overfit DreamZero VFH on a single batch from DROID LeRobot dataset.

set -euo pipefail

export HYDRA_FULL_ERROR=1

# Source full dataset and generated tiny dataset paths.
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"./data/droid_lerobot"}
OVERFIT_DATA_ROOT=${OVERFIT_DATA_ROOT:-"./data/droid_lerobot_single_batch"}

PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-DROID"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid_vfh_overfit_single_batch"}
NUM_GPUS=${NUM_GPUS:-1}
TRAIN_ARCHITECTURE=${TRAIN_ARCHITECTURE:-"vfh_only"}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

# Overfit controls.
KEEP_STEP=${KEEP_STEP:-0}
MAX_STEPS=${MAX_STEPS:-200}

if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: source DROID dataset not found at $DROID_DATA_ROOT"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "ERROR: pretrained checkpoint not found at $PRETRAINED_MODEL_PATH"
    exit 1
fi

if [ "$TRAIN_ARCHITECTURE" != "lora_vfh" ] && [ "$TRAIN_ARCHITECTURE" != "vfh_only" ]; then
    echo "ERROR: TRAIN_ARCHITECTURE must be 'lora_vfh' or 'vfh_only'"
    exit 1
fi

python scripts/data/create_droid_single_batch_dataset.py \
    --source "$DROID_DATA_ROOT" \
    --target "$OVERFIT_DATA_ROOT" \
    --keep-step "$KEEP_STEP"

echo "Starting overfit training on one sample from $OVERFIT_DATA_ROOT"

torchrun --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture="$TRAIN_ARCHITECTURE" \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=3e-4 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=50 \
    training_args.warmup_ratio=0.0 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps="$MAX_STEPS" \
    weight_decay=0.0 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    droid_data_root="$OVERFIT_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL_PATH" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
