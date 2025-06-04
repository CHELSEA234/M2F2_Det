#!/bin/bash
source ~/.bashrc
conda activate M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

python scripts/merge_lora_weights_deepfake.py \
    --model-base ./checkpoints/llava-v1.5-7b-deepfake-stage-2 \
    --model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-8400 \
    --save-model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-3