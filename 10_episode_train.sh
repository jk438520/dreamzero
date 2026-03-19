./scripts/train/droid_training_vfh_finetune.sh \
    DROID_DATA_ROOT="./data/droid_lerobot_10_traj" \
    PRETRAINED_MODEL_PATH="./checkpoints/DreamZero-DROID" \
    OUTPUT_DIR="./checkpoints/dreamzero_droid_vfh_finetune_10_traj" \
    NUM_GPUS=1 \
    TRAIN_ARCHITECTURE="lora_vfh"