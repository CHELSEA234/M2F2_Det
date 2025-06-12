# coding: utf-8
# author: Xiao Guo and Xiufeng Song
import json
import os
import numpy as np
import subprocess
import logging
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, utils
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from dataset import ImageFolderH5Dataset, get_dataloader, ImageFolderDataset, random_split_dataset

from sequence.models.M2F2_Det.models.model import M2F2Det
from sequence.torch_utils import eval_model,display_eval_tb,train_logging,get_lr_blocks,associate_param_with_lr,lrSched_monitor, step_train_logging, Metrics
from sequence.runjobs_utils import init_logger,Saver,DataConfig,torch_load_model,get_iter,get_data_to_copy_str

def get_train_transformation_cfg():
    cfg = {
        'post': {
            'blur': {
                'prob': 0.1,
                'sig': [0.0, 3.0]
            },
            'jpeg': {
                'prob': 0.1,
                'method': ['cv2', 'pil'],
                'qual': [30, 100]
            },
            'noise':{
                'prob': 0.0,
                'var': [0.01]
            }
        },
        'flip': True,    # set false when testing
    }
    return cfg

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
out_handler = logging.FileHandler(filename=os.path.join(model_path, 'train.log'))
out_handler.setLevel(level=logging.INFO)
logger.addHandler(out_handler)


## Hyper-params #######################
hparams = {
            'epochs': 200, 'batch_size': 80, 'basic_lr': 1e-3, 'fine_tune': True, 'use_laplacian': True, 'step_factor': 0.4, 
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
h5_dataset_train_split_fn = './utils/FFPP_split/train.json'
h5_dataset_val_split_fn = './utils/FFPP_split/val.json'
h5_dataset_test_split_fn = './utils/FFPP_split/splits/test.json'

train_transformation_cfg = get_train_transformation_cfg()
val_transformation_cfg = get_val_transformation_cfg()

train_dataset = ImageFolderH5Dataset(data_root=h5_dataset_root, transform_cfg=train_transformation_cfg, split_fn=h5_dataset_train_split_fn)
val_dataset = ImageFolderH5Dataset(data_root=h5_dataset_root, transform_cfg=train_transformation_cfg, split_fn=h5_dataset_val_split_fn)
train_generator = get_dataloader(dataset=train_dataset, mode='train', bs=batch_size, drop_last=True, workers=workers_per_gpu * gpus)
val_generator = get_dataloader(dataset=val_dataset, mode='test', bs=1, workers=workers_per_gpu * gpus)

## Model definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DenseNet_Deepfake()
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
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=step_factor, min_lr=1e-08, patience=patience, cooldown=5, verbose=True)
criterion = nn.CrossEntropyLoss()
mask_criterion = nn.BCELoss()

## Re-loading the model in case
epoch_init=epoch=ib=ib_off=before_train=0
val_loss = np.inf

## Saver object and data config
data_config = DataConfig(model_path, model_name)
saver = Saver(model, optimizer, lr_scheduler, data_config, starting_time, hours_limit=23, mins_limit=0)

# print('Skip Loading the pre-trained weights!!!!!')
epoch_init=epoch=ib=ib_off=before_train=0
load_model_path = os.path.join(model_path,'current_model_180.pth')
val_loss = np.inf
if os.path.exists(load_model_path):
    logger.info(f'Loading weights, optimizer and scheduler from {load_model_path}...')
    ib_off, epoch_init, scheduler, val_loss = torch_load_model(model, optimizer, load_model_path)
epoch_init=epoch=ib=ib_off=before_train=0

## Writer summary for tb
tb_folder = os.path.join(model_path, 'tb_logs',model_name)
writer = SummaryWriter(tb_folder)
log_string_config = '  '.join([k+':'+str(v) for k,v in hparams.items()])
writer.add_text('config : %s' % model_name, log_string_config, 0)

if epoch_init == 0:
    model.zero_grad()

## Start training
tot_iter = 0
for epoch in range(epoch_init,hparams['epochs']):
    logger.info(f'Epoch ############: {epoch}')
    total_loss = 0
    total_accu = 0

    for ib, (img_batch, true_labels) in enumerate(train_generator, 1):
        original = img_batch
        original = original.float().cuda()
        B, C, H, W = original.shape
        img_batch = original
        
        true_labels = true_labels.long().cuda()
        optimizer.zero_grad()
        pred_out = model(img_batch, return_dict=True)
        pred_labels = pred_out['pred']

        cls_loss = criterion(pred_labels, true_labels)
        train_loss = cls_loss # + mask_loss
        log_probs = F.softmax(pred_labels, dim=-1)
            
        res_probs = torch.argmax(log_probs, dim=-1)
        summation = torch.sum(res_probs == true_labels)
        accu = summation / true_labels.shape[0]
        total_accu += accu
        total_loss += cls_loss.item()

        train_loss.backward()
        optimizer.step()

        if tot_iter % hparams['display_step'] == 0:
            train_logging(
                        'loss/train_loss_iter', writer, logger, epoch, saver, 
                        tot_iter, total_loss/hparams['display_step'], 
                        total_accu/hparams['display_step'], lr_scheduler
                        )
            total_loss = 0
            total_accu = 0
        tot_iter += 1

        if (tot_iter + 1) % hparams['valid_step'] == 0:
            model.eval()
            with torch.no_grad():
                frame_metrics = Metrics()
                for idx, val_batch in tqdm(enumerate(val_generator, 1), total=len(val_generator), desc='valid'):
                    val_img_batch, val_true_labels = val_batch
                    val_true_labels = val_true_labels.long().cuda()
                    if isinstance(val_img_batch, tuple) or isinstance(val_img_batch, list):
                        B, C, H, W = val_img_batch[0].shape
                    else:
                        B, C, H, W = val_img_batch.shape
                    
                    val_img_batch = val_img_batch.float().cuda()
                    
                    val_out = model(val_img_batch, return_dict=True)
                    val_preds = val_out['pred']
                    # val_sim_scores = val_out['sim_scores']
                    frame_val_loss = criterion(val_preds, val_true_labels)
                    frame_log_probs = F.softmax(val_preds, dim=-1)
                    frame_res = torch.argmax(frame_log_probs, dim=-1)
                    frame_samples = frame_res.shape[0]
                    
                    frame_matching_num = (frame_res == val_true_labels).sum().item()
                    frame_metrics.roc.predictions.extend(frame_res.tolist())
                    frame_metrics.roc.pred_proba.extend(frame_log_probs[:,0].tolist())
                    frame_fixed_labels = 1 - val_true_labels
                    frame_metrics.roc.gt.extend(frame_fixed_labels[:].tolist())
                    frame_metrics.update(frame_matching_num, frame_val_loss.item(), frame_samples)

            ## Setting the model back to train mode
            model.train()
            video_val_loss = frame_metrics.get_avg_loss()
            frame_metrics.roc.eval()
            frame_metrics.roc.get_average_precision()
            lr_scheduler.step(video_val_loss)
            writer.add_scalar('loss/val_loss_iter', video_val_loss, tot_iter)
            logger.info(f'Val loss: {video_val_loss}')
            logger.info(f'Val auc: {frame_metrics.roc.auc_proba}')
            logger.info(f'Val ap: {frame_metrics.roc.ap}')
            logger.info(f'Patience: {lr_scheduler.num_bad_epochs} / {patience}')

            if frame_metrics.roc.auc_proba > 0.93:
                saver.save_model(epoch,tot_iter,total_loss,before_train,force_saving=True)

    if epoch % hparams['save_epoch'] == 0:
        saver.save_model(epoch,tot_iter,total_loss,before_train,force_saving=True)