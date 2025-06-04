import datetime
import logging
import sys
import torch
import os
import datetime

def init_logger(name):
    # print(f"the log works on {name}.")
    logger = logging.getLogger(name)
    h = logging.StreamHandler(sys.stdout)
    h.flush = sys.stdout.flush
    logger.addHandler(h)
    return logger

logger = init_logger(__name__)
logger.setLevel(logging.INFO)

def torch_load_model(model, optimizer, load_model_path,strict=True):
    loaded_file = torch.load(load_model_path)
    model.load_state_dict(loaded_file['model_state_dict'], strict=strict)
    # model.load_state_dict(loaded_file['model_state_dict'], strict=False)
    iteration = loaded_file['iter']
    scheduler = loaded_file['scheduler']
    epoch = loaded_file['epoch']
    val_loss = 1.0
    if 'val_loss' in loaded_file:
        val_loss = loaded_file['val_loss']
    # optimizer.load_state_dict(loaded_file['optimizer_state_dict'])    
    return iteration, epoch, scheduler, val_loss

class DataConfig(object):
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name

class Saver(object):
    def __init__(self, model, optimizer, scheduler, data_config,
                 starting_time, hours_limit=23, mins_limit=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_loss = sys.maxsize
        self.data_config = data_config
        
        self.hours_limit = hours_limit
        self.mins_limit = mins_limit
        self.starting_time = starting_time

    def save_model(self,epoch,ib,val_loss,before_train,best_only=False,force_saving=False):
        if (val_loss  <= self.best_val_loss and not(before_train)) or force_saving:
            
            ## preserving best_loss
            if val_loss  <= self.best_val_loss:
                self.best_val_loss = val_loss
                
            saving_str = f'{self.data_config.model_path}/model-checkpoint-Train_{self.data_config.model_name}-epoch_{epoch}-val_loss_{self.best_val_loss}.pth'
            ## Save with the full name and with just best
            ## so we know always the best model
            if best_only:
                saving_list = [os.path.join(os.path.dirname(saving_str),'best_model.pth')]
            else:
                saving_list = [saving_str,os.path.join(os.path.dirname(saving_str),'best_model.pth')]

            if force_saving:
                saving_list = [os.path.join(os.path.dirname(saving_str),f'current_model_{epoch}.pth')]
                
            for ss in saving_list:
                # print("save...")
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict':
                            self.optimizer.state_dict() if self.optimizer is not None else None,
                            'iter' : ib,
                            'scheduler' : self.scheduler,
                            'val_loss' : val_loss,
                            },
                           ss
                )
           
    def check_time(self):
        this_time = datetime.datetime.now()
        days, hours, mins = self.days_hours_minutes(this_time - self.starting_time)
        return days, hours, mins

    def resubmit_if_time(self,epoch,tot_iter,val_loss,before_train):
        ''''deprecated function.'''
        days, hours, mins = self.check_time()
        if hours == self.hours_limit and mins > self.mins_limit:
            logger.info('Time Limit Reached h:%d,m:%d - Saving and Resubmitting' % (hours,mins) )
            logger.info('Saving current model...')
            self.best_val_loss = self.save_model(epoch,tot_iter,val_loss,
                                                 before_train,
                                                 best_only=True,
                                                 force_saving=True)
            ## Exit 99 is the code used by queue to automatically resubmit your job
            exit(99)

    def days_hours_minutes(self,td):
        return td.days, td.seconds//3600, (td.seconds//60)%60

def get_iter(train_joined_dataset,epoch,epoch_init,ib,ib_off,batch_size):
    tot_iter = ((len(train_joined_dataset)//batch_size)*epoch) + ib + ib_off
    return tot_iter

def get_data_to_copy_str(img_path,datasets,ctype):
    list_string_data = [f'{img_path}FFPP*{dd}*{ctype}*.h5' for dd in datasets]
    data_copy_str = ''
    for l in list_string_data:
        data_copy_str += (l+' ')
    return data_copy_str
