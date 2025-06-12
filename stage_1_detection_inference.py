# coding: utf-8
# author: Xiao Guo and Xiufeng Song
import json
import os
import csv
import numpy as np
import subprocess
import logging
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import models, utils
from sklearn.metrics import roc_auc_score
from dataset import ImageFolderH5Dataset_inference, get_dataloader
from sequence.models.M2F2_Det.models.model import M2F2Det
from sequence.runjobs_utils import init_logger,torch_load_model

def get_val_transformation_cfg():
    cfg = {
        'post': {
            'blur': {
                'prob': 0.0,
                'sig': [0.0, 3.0]
            },
            'jpeg': {
                'prob': 0.0,
                'method': ['cv2', 'pil'],
                'qual': [30, 100]
            },
            'noise':{
                'prob': 0.0,
                'var': [0.01]
            }
        },
        'flip': False,    # set false when testing
    }
    return cfg

def parse_auc_score(result_csv):
    '''compute the auc score and the best accuracy'''
    df = pd.read_csv(result_csv)
    last_image_id = ""
    pred_lst, gt_lst = [], []
    pred_sample_lst = []
    for index, row in df.iterrows():
        image_id = "_".join(row['image_id'].split('_')[:-1])
        cur_image_id = image_id
        if cur_image_id != last_image_id:
            last_image_id = cur_image_id
            if len(pred_sample_lst) > 0:
                sampled_pred_sample_lst = pred_sample_lst[::1]
                sampled_pred_sample_lst.sort()
                ## remove frames with highest and lowest 10 scores.
                sampled_pred_sample_lst = sampled_pred_sample_lst[10:-10]
                pred_lst.append(np.mean(sampled_pred_sample_lst))
                gt_lst.append(row['ground_truth'] )
            pred_sample_lst = []
        pred_sample_lst.append(row['prediction'])
    auc_score = roc_auc_score(gt_lst, pred_lst)

    accuracy_lst = []
    for threshold in np.arange(0, 1.1, 0.0001):
        binary_preds = [1 if pred >= threshold else 0 for pred in pred_lst]
        accuracy = sum(1 for x, y in zip(binary_preds, gt_lst) if x == y) / len(gt_lst)
        accuracy_lst.append(accuracy)
    
    return auc_score, max(accuracy_lst)

starting_time = datetime.datetime.now()

## Deterministic training
_seed_id = 100
torch.backends.cudnn.deterministic = True
torch.manual_seed(_seed_id)

exp_name = 'stage_1'
model_name = exp_name
model_path = './checkpoints'
model_path = os.path.join(model_path, model_name)

# Create the model path if doesn't exists
if not os.path.exists(model_path):
    subprocess.call(f"mkdir -p {model_path}", shell=True)
    
gpus = 1

# logger for training
logger = init_logger(__name__)
logger.setLevel(logging.INFO)
out_handler = logging.FileHandler(filename=os.path.join(model_path, 'test.log'))
out_handler.setLevel(level=logging.INFO)
logger.addHandler(out_handler)


## Hyper-params #######################
hparams = {
            'epochs': 200, 'batch_size': 300, 'basic_lr': 1e-3, 'fine_tune': True, 'use_laplacian': True, 'step_factor': 0.4, 
            'patience': 6, 'weight_decay': 1e-06, 'lr_gamma': 2.0, 'use_magic_loss': True, 'feat_dim': 2048, 'drop_rate': 0.2, 
            'skip_valid': False, 'rnn_type': 'LSTM', 'rnn_hidden_size': 256, 'num_rnn_layers': 1, 'rnn_drop_rate': 0.2, 
            'bidir': True, 'merge_mode': 'concat', 'perc_margin_1': 0.95, 'perc_margin_2': 0.95, 'soft_boundary': False, 
            'dist_p': 2, 'radius_param': 0.84, 'strat_sampling': True, 'normalize': True, 'window_size': 10, 'hop': 1, 
            'valid_step': 1000, 'display_step': 10, 'use_sched_monitor': True, 'level': 'video', 'save_epoch': 20
            }
batch_size = hparams['batch_size']
basic_lr = hparams['basic_lr']
fine_tune = hparams['fine_tune']
use_laplacian = hparams['use_laplacian']
step_factor = hparams['step_factor']
patience = hparams['patience']
weight_decay = hparams['weight_decay']
lr_gamma = hparams['lr_gamma']
use_magic_loss = hparams['use_magic_loss']
feat_dim = hparams['feat_dim']
drop_rate = hparams['drop_rate']
rnn_type = hparams['rnn_type']
rnn_hidden_size = hparams['rnn_hidden_size']
num_rnn_layers = hparams['num_rnn_layers']
rnn_drop_rate = hparams['rnn_drop_rate']
bidir = hparams['bidir']
merge_mode = hparams['merge_mode']
perc_margin_1 = hparams['perc_margin_1']
perc_margin_2 = hparams['perc_margin_2']
dist_p = hparams['dist_p']
radius_param = hparams['radius_param']
strat_sampling = hparams['strat_sampling']
normalize = hparams['normalize']
window_size = hparams['window_size']
hop = hparams['hop']
soft_boundary = hparams['soft_boundary']
use_sched_monitor = hparams['use_sched_monitor']
level = hparams['level']    # 'frame' or 'video'
valid_step = hparams['valid_step']
valid_step = 50

logger.info(hparams)
########################################

accum_grad_loop = batch_size
workers_per_gpu = 12

h5_dataset_root = "/user/guoxia11/cvlshare/cvl-guoxia11/FaceForensics_HiFiNet"  ## put your data root here
h5_dataset_test_split_fn = './utils/FFPP_split/test.json'
# h5_dataset_test_split_fn = './utils/FFPP_split/train.json'
test_transformation_cfg = get_val_transformation_cfg()
test_dataset = ImageFolderH5Dataset_inference(data_root=h5_dataset_root, transform_cfg=test_transformation_cfg, split_fn=h5_dataset_test_split_fn)
test_generator = get_dataloader(dataset=test_dataset, mode='test', bs=batch_size, workers=workers_per_gpu * gpus)

## Model definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = M2F2Det(
    clip_text_encoder_name="openai/clip-vit-large-patch14-336",
    clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
    deepfake_encoder_name='efficientnet_b4',
    hidden_size=1792,
)

llava_vision_tower = torch.load('./utils/weights/vision_tower.pth', weights_only=True)
vision_tower_dict = {}
for k, v in llava_vision_tower.items():
    vision_tower_dict[k.replace("vision_tower.", "")] = v
if model.clip_vision_encoder is not None:   
    model.clip_vision_encoder.model.load_state_dict(vision_tower_dict, strict=True)
    print('Load Llava Vision Tower.')

# logger.info(model)
model = torch.nn.DataParallel(model)
model = model.cuda()

## Attention!!! Only the parameters that is declared in the models "assign_lr_dict_list()" will be optimized.
params_dict_list = model.module.assign_lr_dict_list(lr=basic_lr)
optimizer = torch.optim.Adam(params_dict_list, weight_decay=weight_decay)

load_model_path = os.path.join(model_path,'current_model_180.pth')
logger.info(f'Loading weights, optimizer and scheduler from {load_model_path}...')
_ = torch_load_model(model, optimizer, load_model_path)

model.eval()
pred_lst, gt_lst, img_id_lst = [], [], []
with torch.no_grad():
    for idx, val_batch in tqdm(enumerate(test_generator, 1), total=len(test_generator), desc='valid'):
        val_img_batch, val_true_labels, val_img_id = val_batch
        
        val_true_labels = val_true_labels.long().cuda()
        val_img_batch = val_img_batch.float().cuda()
        val_out = model(val_img_batch, return_dict=True)
        val_preds = val_out['pred']
        frame_log_probs = F.softmax(val_preds, dim=-1)
        pred_lst.extend(frame_log_probs[:,0].tolist())
        frame_fixed_labels = 1 - val_true_labels
        gt_lst.extend(frame_fixed_labels[:].tolist())
        img_id_lst.extend(val_img_id)

model_name = load_model_path.split('/')[-1].replace('.pth','')
result_csv = f'./result_{model_name}.csv'
with open(result_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'ground_truth', 'prediction'])
    for img_id, gt, pred in zip(img_id_lst, gt_lst, pred_lst):
        writer.writerow([img_id, gt, pred])
logger.info(f'Results saved to {result_csv}')

auc_score, accuracy = parse_auc_score(result_csv)
logger.info(f'Test AUC-ROC: {auc_score:.4f}')
logger.info(f'Test accuracy: {accuracy:.4f}')