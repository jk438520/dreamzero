# Configure paths (override defaults as needed)
export DROID_DATA_ROOT="./data/droid_lerobot_1_traj_with_value_function"
export OUTPUT_DIR="./checkpoints/dreamzero_droid_finetune_lora_1_traj_with_value_function"
export NUM_GPUS=2

export CUDA_VISIBLE_DEVICES=6,7

# Point to your downloaded model weights (if not using default paths)
export WAN_CKPT_DIR="./checkpoints/Wan2.1-I2V-14B-480P"
export TOKENIZER_DIR="./checkpoints/umt5-xxl"

export PRETRAINED_MODEL_PATH="./checkpoints/DreamZero-DROID"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch training
bash scripts/train/droid_training_finetune.sh