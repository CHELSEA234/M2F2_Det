#!/bin/bash
source ~/.bashrc
conda activate M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"

CUDA_NUM=1,2,3,4,5,7
# CUDA_NUM=1,2,3,4
MODEL_VERSION="./checkpoints/llava-v1.5-7b-deepfake-stage-2"
DATA_PATH="./utils/DDVQA_split/c40/train_DDVQA_format.json"
IMG_FOLDER="./utils/DDVQA_images/c40/train"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta"
DEEPFAKE_CKPT_PATH="./utils/weights/M2F2_Det_densenet121.pth"
VISION_TOWER="openai/clip-vit-large-patch14-336"

deepspeed --include localhost:$CUDA_NUM llava/train/train_deepfake.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_VERSION \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMG_FOLDER \
    --vision_tower $VISION_TOWER \
    --deepfake_ckpt_path $DEEPFAKE_CKPT_PATH \
    --tune_mm_mlp_adapter True \
    --tune_deepfake_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to wandb
