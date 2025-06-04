#!/bin/bash
source ~/.bashrc
conda activate M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python scripts/merge_lora_weights_deepfake_random.py \
    --model-path /user/guoxia11/cvlshare/cvl-guoxia11/huggingface/hub/llava-v1.5-7b \
    --save-model-path ./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1