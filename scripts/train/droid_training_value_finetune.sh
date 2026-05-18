#!/bin/bash
# DreamZero DROID Value-Function LoRA Fine-Tuning Script
#
# Usage:
#   # Set your value-augmented dataset path and output directory, then run:
#   bash scripts/train/droid_training_value_finetune.sh
#
# Prerequisites:
#   - Value-augmented DROID dataset in LeRobot format at DROID_DATA_ROOT
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded)
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded)
#   - Existing DreamZero checkpoint for fine-tuning (e.g. DreamZero-DROID)

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# Dataset path (DROID in LeRobot format with action.value_function)
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"./data/droid_lerobot_value"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid_value_finetune"}

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-8}

# Training schedule knobs (override via env from wrapper script)
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MAX_STEPS=${MAX_STEPS:-400}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
# LoRA knobs
LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-4}
# For polynomial scheduler: quick early decay and flat low-LR tail near the end.
LR_END=${LR_END:-1e-6}
LR_POWER=${LR_POWER:-3.0}
SAVE_STEPS=${SAVE_STEPS:-50}

# Model weight paths (download from HuggingFace if not already present)
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

# Pretrained checkpoint used as initialization for LoRA fine-tuning
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-DROID"}
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
    echo "ERROR: value-augmented DROID dataset not found at $DROID_DATA_ROOT"
    echo "Create it first with: scripts/data/add_value_function.py"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "ERROR: pretrained checkpoint not found at $PRETRAINED_MODEL_PATH"
    echo "Set PRETRAINED_MODEL_PATH to an existing DreamZero checkpoint directory"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/droid_value_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
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
    training_args.learning_rate=$LEARNING_RATE \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=$WARMUP_RATIO \
    training_args.lr_scheduler_type=$LR_SCHEDULER_TYPE \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
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
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL_PATH \
    ++action_head_cfg.config.use_value_reconstruction_loss=true \
    ++action_head_cfg.config.value_reconstruction_loss_weight=1 \
    ++action_head_cfg.config.value_reconstruction_index=-1 \
    ++action_head_cfg.config.value_reconstruction_huber_delta=0.01 \
    ++action_head_cfg.config.lora_rank=$LORA_RANK \
    ++action_head_cfg.config.lora_alpha=$LORA_ALPHA \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    ++training_args.lr_scheduler_kwargs.lr_end=$LR_END \
    ++training_args.lr_scheduler_kwargs.power=$LR_POWER 
