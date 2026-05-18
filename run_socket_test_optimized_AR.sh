#!/bin/sh
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 5000 \
    --enable-dit-cache \
    --model-path checkpoints/dreamzero_droid_finetune_full_1_traj_with_value_function 