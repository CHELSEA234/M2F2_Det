#!/bin/bash
source ~/.bashrc
conda activate M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python -m llava.serve.cli_DDVQA_exp \
    --model-path ./checkpoints/llava-v1.5-7b-M2F2-Det