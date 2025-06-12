#!/bin/bash
source ~/.bashrc
conda activate M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python scripts/merge_lora_weights_deepfake.py \
    --model-base /user/guoxia11/cvlshare/cvl-guoxia11/huggingface/hub/llava-v1.5-7b \
    --model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-2-proj \
    --save-model-path ./checkpoints/llava-v1.5-7b-deepfake-stage-2

bash scripts/finetune_stage_3.sh

CUDA_VISIBLE_DEVICES=$CUDA_NUM python scripts/merge_lora_weights_deepfake.py \
    --model-base ./checkpoints/llava-v1.5-7b-deepfake-stage-2 \
    --model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-2124 \
    --save-model-path ./checkpoints/llava-v1.5-7b-M2F2-Det