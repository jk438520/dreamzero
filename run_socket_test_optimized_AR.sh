#!/bin/sh
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 5000 \
    --enable-dit-cache \
    --model-path checkpoints/dreamzero_droid_finetune_lora_1_traj_with_value_function