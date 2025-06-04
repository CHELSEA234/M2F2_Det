source ~/.bashrc
conda activate M2F2_det

CUDA_NUM=0,1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python run_detector.py