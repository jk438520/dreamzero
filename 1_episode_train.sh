export DROID_DATA_ROOT="./data/droid_lerobot_1_traj"
export PRETRAINED_MODEL_PATH="./checkpoints/DreamZero-DROID"
export OUTPUT_DIR="./checkpoints/dreamzero_droid_vfh_finetune_1_traj"
export NUM_GPUS=2
export TRAIN_ARCHITECTURE="lora_vfh"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=6,7
mkdir -p logs
./scripts/train/droid_training_vfh_finetune.sh 2>&1 | tee logs/train.log
