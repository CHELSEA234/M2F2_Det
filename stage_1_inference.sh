source ~/.bashrc
conda activate M2F2_det

CUDA_NUM=0
CUDA_VISIBLE_DEVICES=$CUDA_NUM python stage_1_detection_inference.py
