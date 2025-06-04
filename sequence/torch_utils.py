import math
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from .runjobs_utils import init_logger
import logging
import torch.nn.functional as F
import os
from einops import rearrange

logger = init_logger(__name__)
logger.setLevel(logging.INFO)


class ROC(object):
    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.auc = None
        self.scores = None
        self.fpr_proba = None
        self.tpr_proba = None
        self.auc_proba = None
        self.scores_proba = None
        self.ap_0 = None
        self.ap_1 = None
        self.weighted_ap = None
        self.ap = None
        
        self.predictions = []
        self.gt = []
        self.pred_proba = []
    
    def get_tpr_at_fpr(self,fpr_value):
        abs_fpr = np.absolute(self.fpr_proba - fpr_value)
        idx_min = np.argmin(abs_fpr)
        fpr_value_target = self.fpr_proba[idx_min]
        idx = np.max(np.where(self.fpr_proba == fpr_value_target))
        return self.tpr_proba[idx], self.scores_proba[idx]
        
    def eval(self):
        self.fpr_proba, self.tpr_proba, self.scores_proba = metrics.roc_curve(self.gt,self.pred_proba,drop_intermediate=True)
        self.auc_proba = metrics.auc(self.fpr_proba,self.tpr_proba)

    def compute_best_accuracy(self,n_samples=200):
        '''find the best threshold for the accuracy.'''
        acc_thrs = []
        min_thr = min(self.pred_proba)
        max_thr = max(self.pred_proba)
        all_thrs = np.linspace(min_thr,max_thr,n_samples).tolist()
        for t in all_thrs:
            acc = self.compute_acc(self.pred_proba,self.gt,t)
            acc_thrs.append((t,acc))
        acc_thrs_arr = np.array(acc_thrs)
        idx_max = acc_thrs_arr[:,1].argmax()
        best_thr = acc_thrs_arr[idx_max,0]
        best_acc = acc_thrs_arr[idx_max,1]
        return best_thr, best_acc

    def compute_acc(self,list_scores,list_labels,thr):
        labels = np.array(list_labels)
        scores_th = (np.array(list_scores) >= thr).astype(np.int32)
        acc = (scores_th==labels).sum()/labels.size
        return acc
    
    def get_precision(self,criterion,thr):
        '''compute the best precision'''
        pred_labels = []
        for d in self.pred_proba:
            if (d < thr):
                pred_labels.append(0)
            elif (d >= thr):
                pred_labels.append(1)
        self.ap_0 = metrics.precision_score(self.gt, pred_labels, average='binary', pos_label=0)
        self.ap_1 = metrics.precision_score(self.gt, pred_labels, average='binary', pos_label=1)
        self.weighted_ap = metrics.precision_score(self.gt, pred_labels, average='weighted')
    
    def get_average_precision(self):
        self.ap = metrics.average_precision_score(self.gt, self.pred_proba, average='macro')

class Metrics(object):
    def __init__(self):
        self.tp = 0
        self.tot_samples = 0
        self.loss = 0.0
        self.loss_samples = 0
        self.roc = ROC()
        
        self.best_valid_acc = 0.0
        self.best_valid_thr = 0.0

        self.tuned_acc_thrs = (0,0)
        
    def update(self,tp,loss_value,samples):
        self.tp+=tp
        self.tot_samples+=samples
        self.loss+=loss_value
        ## for the loss we sum +1
        ## because we get the averaged
        ## value over the batch (a single scalar)
        self.loss_samples+=1

    def get_acc(self):
        if self.tot_samples == 0:
            raise ZeroDivisionError('not enough sample to compute accuracy')
        return self.tp/self.tot_samples

    def get_avg_loss(self):
        if self.loss_samples == 0:
            raise ZeroDivisionError('not enough sample to avg loss')
        return self.loss/self.loss_samples

def count_matching_samples(preds,true_labels,criterion,use_magic_loss=True):
    acc = 0
    if use_magic_loss:
        for l,d in zip(true_labels,preds):
            if (l == criterion.class_label and d < criterion.R) \
            or (l != criterion.class_label and d >= criterion.R):
                acc += 1
    else:
        matching_idx = (preds.argmax(dim=1)==true_labels)
        acc = matching_idx.sum().item()
    return acc
        
def get_dist(criterion,feat):
    return criterion.pdist(feat, criterion.c)

def _convert2Proba(d,m1,m2,label):
    # Probablity of being FAKE
    if label == 1:
        if d > m2:
            return 1.0
        elif d < m1:
            return 0.0
        else:
            return abs(d-m1)/abs(m2-m1)
    # Probablity of being REAL
    elif label == 0:
        if d > m2:
            return 0.0
        elif d < m1:
            return 1.0
        else:
            return abs(d-m2)/abs(m2-m1)
    
def get_proba(dists,criterion,label):
    m1 = criterion.R - (criterion.R*criterion.perc_margin_1)
    m2 = criterion.R + (criterion.R*criterion.perc_margin_2)
    pred_proba = list(map(lambda x:_convert2Proba(x,m1,m2,label),dists))
    return pred_proba


def eval_frame_model(model, valid_generator, criterion, device, desc='valid'):
    with torch.no_grad():
        frame_metrics = Metrics()
        video_metrics = Metrics()
        for idx, val_batch in tqdm(enumerate(valid_generator, 1), total=len(valid_generator), desc=desc):

            if idx%10!=0:
                continue

            val_img_batch, val_true_labels = val_batch
            video_true_labels = val_true_labels.long().to(device)
            if isinstance(val_img_batch, tuple) or isinstance(val_img_batch, list):
                B, L, C, H, W = val_img_batch[0].shape
            elif len(val_img_batch.shape) == 5:
                B, L, C, H, W = val_img_batch.shape
            else:
                B, C, H, W = val_img_batch.shape
                L = 1
            frame_true_labels = video_true_labels.repeat_interleave(L, dim=0)
            video_samples = video_true_labels.shape[0]
            frame_samples = frame_true_labels.shape[0]
            frame_preds = model(val_img_batch)
            frame_val_loss = criterion(frame_preds, frame_true_labels)
            frame_log_probs = F.softmax(frame_preds, dim=-1)
            frame_res = torch.argmax(frame_log_probs, dim=-1)
            
            video_preds = frame_preds.reshape(B, L, -1)    # [B, L, 2]
            for b in range(B):    # do video level prediction on frame level results per video
                single_video_preds = video_preds[b]    # [L, 2]
                single_video_labels = video_true_labels[b].unsqueeze(0)
                frame_means = torch.mean(single_video_preds, dim=0)
                frame_stds = torch.std(single_video_preds, dim=0)
                inliner_filter = torch.abs(single_video_preds[:, 0] - frame_means[0]) <= 3 * frame_stds[0]    # 3 sigmas for outlier exclusion
                single_video_preds = single_video_preds[inliner_filter, :]
                single_video_preds = torch.mean(single_video_preds, dim=0, keepdim=True)
                single_video_val_loss = criterion(single_video_preds, single_video_labels)
                single_video_log_probs = F.softmax(single_video_preds, dim=-1)
                single_video_res = torch.argmax(single_video_log_probs, dim=-1)
                
                single_video_samples = 1
                single_video_matching_num = (single_video_res == single_video_labels).sum().item()
                video_metrics.roc.predictions.extend(single_video_res.tolist())
                video_metrics.roc.pred_proba.extend(single_video_log_probs[:,0].tolist())
                single_video_fixed_labels = 1 - single_video_labels
                video_metrics.roc.gt.extend(single_video_fixed_labels[:].tolist())
                video_metrics.update(single_video_matching_num, single_video_val_loss.item(), single_video_samples)
            
            frame_matching_num = (frame_res == frame_true_labels).sum().item()
            frame_metrics.roc.predictions.extend(frame_res.tolist())
            frame_metrics.roc.pred_proba.extend(frame_log_probs[:,0].tolist())
            frame_fixed_labels = 1 - frame_true_labels
            frame_metrics.roc.gt.extend(frame_fixed_labels[:].tolist())
            frame_metrics.update(frame_matching_num, frame_val_loss.item(), frame_samples)
            
        ## Setting the model back to train mode
        model.train()
    return frame_metrics, video_metrics
            

def eval_video_model(model, valid_generator, criterion, window_size, device, desc='valid'):
    with torch.no_grad():
        frame_metrics = Metrics()
        nonoverlapped_video_metrics = Metrics()
        for idx, val_batch in tqdm(enumerate(valid_generator, 1), total=len(valid_generator), desc=desc):
            val_img_batch, val_true_labels = val_batch
            val_true_labels = val_true_labels.long().to(device)
            if isinstance(val_img_batch, tuple) or isinstance(val_img_batch, list):
                B, L, C, H, W = val_img_batch[0].shape
            else:
                B, L, C, H, W = val_img_batch.shape
            for index in range(L - window_size + 1):
                if isinstance(val_img_batch, tuple) or isinstance(val_img_batch, list):
                    clip = val_img_batch[0][:, index: index + window_size, :, :, :], val_img_batch[1][:, index: index + window_size, :, :, :]
                else:
                    clip = val_img_batch[:, index: index + window_size, :, :, :]
                val_preds = model(clip)
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
                
                if index % window_size == 0 or index == L - window_size:
                    nonoverlapped_video_metrics.roc.predictions.extend(frame_res.tolist())
                    nonoverlapped_video_metrics.roc.pred_proba.extend(frame_log_probs[:,0].tolist())
                    nonoverlapped_video_metrics.roc.gt.extend(frame_fixed_labels[:].tolist())
                    nonoverlapped_video_metrics.update(frame_matching_num, frame_val_loss.item(), frame_samples)
        
        ## Setting the model back to train mode
        model.train()
    return frame_metrics, nonoverlapped_video_metrics

         
def eval_model(model,valid_joined_generator,criterion,
               device,desc='valid',
               debug_mode=False, level='video', window_size=None):
    model.eval()
    print("with the eval model, without bn.")
    isValid = False
    if desc == 'valid':
        isValid = True
    assert(level in ['frame', 'video'])
    if level == 'frame':
        print('Eval model on frame level')
        return eval_frame_model(model, valid_joined_generator, criterion, device, desc)
    else:
        assert(window_size is not None)
        print('Eval model on video level')
        return eval_video_model(model, valid_joined_generator, criterion, window_size, device, desc)


def display_eval_tb(writer,metrics,tot_iter,desc='valid',old_metrics=False):
    avg_loss = metrics.get_avg_loss()
    acc = metrics.get_acc()
    ## in case of test we report the accuracy
    ## with the best thrs from the validation
    if desc != 'valid':
        acc = metrics.tuned_acc_thrs[0]
        thrs = metrics.tuned_acc_thrs[1]        
    # auc = metrics.roc.auc
    auc = metrics.roc.auc_proba
    writer.add_scalar('%s/loss'%desc, avg_loss, tot_iter)
    writer.add_scalar('%s/acc'%desc, acc, tot_iter)                      
    writer.add_scalar('%s/auc'%desc, auc, tot_iter)

    fpr_values = [0.1,0.01]    
    for fpr_value in fpr_values:
        tpr_fpr, score_for_tpr_fpr = metrics.roc.get_tpr_at_fpr(fpr_value)
        writer.add_scalar('%s/tpr_fpr_%.0f'%(desc,(fpr_value*100.0)), tpr_fpr, tot_iter)
    
def train_logging(string, writer, logger, epoch, saver, tot_iter, loss, accu, lr_scheduler):
    _, hours, mins = saver.check_time()
    logger.info("[Epoch %d] | h:%d m:%d | iteration: %d, loss: %f, accu: %f", epoch, hours, mins, tot_iter,
                loss, accu)
    
    writer.add_scalar(string, loss, tot_iter )
    for count, gp in enumerate(lr_scheduler.optimizer.param_groups,1):
        writer.add_scalar('progress/lr_%d'%count, gp['lr'], tot_iter)
    writer.add_scalar('progress/epoch', epoch, tot_iter)
    writer.add_scalar('progress/curr_patience',lr_scheduler.num_bad_epochs,tot_iter)
    writer.add_scalar('progress/patience',lr_scheduler.patience,tot_iter)

def step_train_logging(string, writer, logger, epoch, saver, tot_iter, loss, accu, lr_scheduler):
    _, hours, mins = saver.check_time()
    logger.info("[Epoch %d] | h:%d m:%d | iteration: %d, loss: %f, accu: %f", epoch, hours, mins, tot_iter,
                loss, accu)
    
    writer.add_scalar(string, loss, tot_iter )
    for count, gp in enumerate(lr_scheduler.optimizer.param_groups,1):
        writer.add_scalar('progress/lr_%d'%count, gp['lr'], tot_iter)
    writer.add_scalar('progress/epoch', epoch, tot_iter)
    # writer.add_scalar('progress/curr_patience',lr_scheduler.num_bad_epochs,tot_iter)
    # writer.add_scalar('progress/patience',lr_scheduler.patience,tot_iter)
    
    
def get_lr_blocks(lr_basic=2e-05,gamma=2.0):
    ## These values specify the indexing for Densenet 121
    ## it will access the first conv block, then each DenseBlock+Transition.
    ## In total we have 5 blocks (the classification layer is outside of this)
    ## Note: this is model specific. We might need a dictionary with the model name
    ## if we want to do it model agnostic
    idx_blocks = [[0,4],[4,6],[6,8],[8,10],[10,12]]
    lr_list = [None]*(len(idx_blocks)+1)
    for count, l in enumerate(reversed(range(len(idx_blocks)+1))):
        scale_factor = 1/(gamma**l)
        lr_list[count] = 0.5*lr_basic*scale_factor
    lr_list.append(lr_basic)
    return idx_blocks, lr_list

def associate_param_with_lr(model_lp,idx_blocks,lr_list,
                            offset=6,lp_lr_multiplier=1.0):
    count = 0
    params_dict_list = []
    if torch.cuda.device_count() > 1:
        ## LP branch ########################################################
        ## Optimizing fast the LP branch
        params_dict_list.append({'params' : model_lp.module.lp_branch.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [laplacian] c_lr: %.10f'% (count,lr_list[-1])) 
        # print(model_lp.module.lp_branch)
        ## Optimizing fast merging of features
        params_dict_list.append({'params' : model_lp.module.conv_1x1_merge.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [conv_1x1_merge] c_lr: %.10f'% (count,lr_list[-1])) 
        # print(model_lp.module.conv_1x1_merge)
        ## RGB branch ########################################################
        ## Optimizing 1st conv layer very very slowly
        params_dict_list.append({'params' : model_lp.module.rgb_branch[0][:4].parameters(), 'lr' : lr_list[0]})
        print('******* lr block %d, [rgb_conv] c_lr: %.10f'% (count,lr_list[0])) 
        # print(model_lp.module.rgb_branch[0][:4])
        ## Optimizing 1st densnet block very slowly
        params_dict_list.append({'params' : model_lp.module.rgb_branch[0][4:6].parameters(), 'lr' : lr_list[1]})
        print('******* lr block %d, [rgb_dense_block] c_lr: %.10f'% (count,lr_list[1])) 
        # print(model_lp.module.rgb_branch[0][4:6])
        ## Now deceding the optimizer in the backbone
        mod_feat = model_lp.module.backbone
        for count, (idx, c_lr) in enumerate(zip(idx_blocks[2:],lr_list[2:])):
            print('******* lr block %d, [%d,%d] c_lr: %.10f'% (count,idx[0],idx[1],c_lr))    
            sliced_model = mod_feat[idx[0]-offset:idx[-1]-offset]                    
            # print(sliced_model)
            param_dict = {'params' : sliced_model.parameters(), 'lr' : c_lr}
            params_dict_list.append(param_dict)
        ## Adding the flatten  
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+1,'flatten',c_lr)) 
        # print(model_lp.module.flatten)       
        params_dict_list.append({'params' : model_lp.module.flatten.parameters(), 'lr' : c_lr})

        ## Finally adding the RNN
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+2,'RNN',c_lr)) 
        # print(model_lp.module.rnn)       
        params_dict_list.append({'params' : model_lp.module.rnn.parameters(), 'lr' : c_lr})
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+2,'output',c_lr)) 
        # print(model_lp.module.output)       
        params_dict_list.append({'params' : model_lp.module.output.parameters(), 'lr' : c_lr})
    else:
        ## LP branch ########################################################
        ## Optimizing fast the LP branch
        params_dict_list.append({'params' : model_lp.lp_branch.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [laplacian] c_lr: %.10f'% (count,lr_list[-1])) 
        print(model_lp.lp_branch)
        ## Optimizing fast merging of features
        params_dict_list.append({'params' : model_lp.conv_1x1_merge.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [conv_1x1_merge] c_lr: %.10f'% (count,lr_list[-1])) 
        print(model_lp.conv_1x1_merge)
        ## RGB branch ########################################################
        ## Optimizing 1st conv layer very very slowly
        params_dict_list.append({'params' : model_lp.rgb_branch[0][:4].parameters(), 'lr' : lr_list[0]})
        print('******* lr block %d, [rgb_conv] c_lr: %.10f'% (count,lr_list[0])) 
        print(model_lp.rgb_branch[0][:4])
        ## Optimizing 1st densnet block very slowly
        params_dict_list.append({'params' : model_lp.rgb_branch[0][4:6].parameters(), 'lr' : lr_list[1]})
        print('******* lr block %d, [rgb_dense_block] c_lr: %.10f'% (count,lr_list[1])) 
        print(model_lp.rgb_branch[0][4:6])
        ## Now deceding the optimizer in the backbone
        mod_feat = model_lp.backbone
        for count, (idx, c_lr) in enumerate(zip(idx_blocks[2:],lr_list[2:])):
            print('******* lr block %d, [%d,%d] c_lr: %.10f'% (count,idx[0],idx[1],c_lr))    
            sliced_model = mod_feat[idx[0]-offset:idx[-1]-offset]                    
            print(sliced_model)
            param_dict = {'params' : sliced_model.parameters(), 'lr' : c_lr}
            params_dict_list.append(param_dict)
        ## Adding the flatten  
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+1,'flatten',c_lr)) 
        print(model_lp.flatten)       
        params_dict_list.append({'params' : model_lp.flatten.parameters(), 'lr' : c_lr})

        ## Finally adding the RNN
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+2,'RNN',c_lr)) 
        print(model_lp.rnn)       
        params_dict_list.append({'params' : model_lp.rnn.parameters(), 'lr' : c_lr})
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+2,'output',c_lr)) 
        print(model_lp.output)       
        params_dict_list.append({'params' : model_lp.output.parameters(), 'lr' : c_lr})
    return params_dict_list

class lrSched_monitor(object):
    """
    This class is used to monitor the learning rate scheduler's behavior
    during training. If the learning rate decreases then this class re-initializes
    the last best state of the model and starts training from that point of time.
    
    Parameters
    ----------
    model : torch model
    scheduler : learning rate scheduler object from training
    data_config : this object holds model_path and model_name, used to load the last best model.
    """
    def __init__(self, model, scheduler, data_config):
        self.model = model
        self.scheduler = scheduler
        self.model_name = data_config.model_name
        self.model_path = data_config.model_path
        self._last_lr = [0]*len(scheduler.optimizer.param_groups)
        self.prev_lr_mean = self.get_lr_mean()
    
    ## Get the current mean learning rate from the optimizer
    def get_lr_mean(self):
        lr_mean = 0
        for i, grp in enumerate(self.scheduler.optimizer.param_groups):
            if 'lr' in grp.keys():
                lr_mean += grp['lr']
                self._last_lr[i] = grp['lr']
        return lr_mean/(i+1)       
        
    ## This is the function that is to be called right after lr_scheduler.step(val_loss)    
    def monitor(self):
        ## When self.num_bad_epochs > self.patience, lr will be decreased
        if self.scheduler.num_bad_epochs == self.scheduler.patience:
            self.prev_lr_mean = self.get_lr_mean()      ## to keep the best lr in the last time
        elif self.get_lr_mean() < self.prev_lr_mean:    ## this means scheduler/ReduceLROnPlateau effects, please load the last best model
            self.load_best_model()        ## lr is reduced one, but the rest is the last best one.
            self.prev_lr_mean = self.get_lr_mean()
    
    ## This function loads the last best model once the learning rate decreases
    def load_best_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'))
            self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
            self.scheduler.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print(f'Loading the best model from {self.model_path}')
            if device.type == 'cpu':
                ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'), map_location='cpu')
            else:
                ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'))
            ## Model State Dict
            state_dict = ckpt['model_state_dict']
            ## Since the model files are saved on dataparallel we use the below hack to load the weights on a model in cpu or a model on single gpu.
            keys = state_dict.keys()
            values = state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key.replace('module.','')    # remove the 'module.'
                new_keys.append(new_key)

            new_state_dict = OrderedDict(list(zip(new_keys, values))) # create a new OrderedDict with (key, value) pairs
            self.model.load_state_dict(new_state_dict, strict=True)
            # self.model.load_state_dict(new_state_dict, strict=False)

            ## Optimizer State Dict
            optim_state_dict = ckpt['optimizer_state_dict']
            # Since the model files are saved on dataparallel we use the below hack to load the optimizer state in cpu or a model on single gpu.
            keys = optim_state_dict.keys()
            values = optim_state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key.replace('module.','')    # remove the 'module.'
                new_keys.append(new_key)

            new_optim_state_dict = OrderedDict(list(zip(new_keys, values))) # create a new OrderedDict with (key, value) pairs
            self.scheduler.optimizer.load_state_dict(new_optim_state_dict)
        
        ## Reduce the learning rate
        for i, grp in enumerate(self.scheduler.optimizer.param_groups):
            grp['lr'] = self._last_lr[i]
            # self._last_lr[i] is the new one, decreased by ReduceLROnPlateau