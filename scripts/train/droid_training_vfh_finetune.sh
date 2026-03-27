#!/bin/bash
# DreamZero DROID VFH Fine-Tuning Script
#
# Loads an older checkpoint (without VFH), then trains:
#   - VFH (value head) fully, and
#   - the rest either with LoRA (default) or fully frozen.
#
# Usage:
#   bash scripts/train/droid_training_vfh_finetune.sh
#
# Optional:
#   TRAIN_ARCHITECTURE=lora_vfh   # default, LoRA + full VFH
#   TRAIN_ARCHITECTURE=vfh_only   # only full VFH, everything else frozen

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# Dataset path (DROID in LeRobot format)
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"./data/droid_lerobot"}

# Path to old pretrained checkpoint (without VFH)
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-DROID"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid_vfh_finetune"}

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-2}

# New training modes:
#   lora_vfh: LoRA adapters + full value head (recommended)
#   vfh_only: full value head only, rest frozen
TRAIN_ARCHITECTURE=${TRAIN_ARCHITECTURE:-"lora_vfh"}

# Model weight paths (download from HuggingFace if not already present)
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    echo "Download with: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir $DROID_DATA_ROOT"
    exit 1
fi

# Validate pretrained checkpoint exists
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "ERROR: pretrained checkpoint not found at $PRETRAINED_MODEL_PATH"
    exit 1
fi

# Validate training architecture
if [ "$TRAIN_ARCHITECTURE" != "lora_vfh" ] && [ "$TRAIN_ARCHITECTURE" != "vfh_only" ]; then
    echo "ERROR: TRAIN_ARCHITECTURE must be 'lora_vfh' or 'vfh_only'"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=$TRAIN_ARCHITECTURE \
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
    training_args.learning_rate=1e-4 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=10 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL_PATH \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
