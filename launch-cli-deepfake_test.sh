#!/bin/bash
source ~/.bashrc
conda activate M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=4
CUDA_VISIBLE_DEVICES=$CUDA_NUM python -m llava.serve.cli_deepfake_test \
    --model-path /user/guoxia11/cvlshare/cvl-guoxia11/M2F2_Det/llava-1.5-7b-densenet121-deepfake