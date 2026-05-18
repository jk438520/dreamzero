#!/bin/sh

python eval_value_function.py \
    --dataset-path ./data/droid_lerobot_1_traj_with_value_function \
    --output-dir ./logs/value_function_validation \
    --host 0.0.0.0 \
    --port 5000