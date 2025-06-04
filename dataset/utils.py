import torch
from torch.utils.data import DataLoader, random_split

from .process import get_image_transformation_from_cfg


__DEFAULT_TRANSFORMATION_CFG = {
    # 'resize': None,
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
            'var': [0.01, 0.02]
        }
    },
    'crop': {
        'img_size': 224,
        'type': 'random'    # ['center', 'random'], according to 'train', 'val'or 'test' mode
    },
    'flip': True,    # set false when testing
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


def get_default_transformation_cfg():
    return __DEFAULT_TRANSFORMATION_CFG


def get_default_transformation(mode='train'):
    cfg = get_default_transformation_cfg()
    if mode == 'train':
        cfg['crop']['type'] = 'random'
    elif mode == 'test' or mode == 'val':
        cfg['crop']['type'] = 'center'
        cfg['flip'] = False
    return get_image_transformation_from_cfg(cfg)


def get_dataloader(dataset, mode='train', bs=32, workers=4, **kwargs):
    params = {'batch_size': bs,
            'shuffle': (mode=='train'),
            'num_workers': workers,
            'drop_last' : (mode=='train')
    }
    for k, v in kwargs.items():
        params[k] = v
    return DataLoader(dataset, **params)


def random_split_dataset(dataset, split_ratio, seed=42):
    total_ratio = sum(split_ratio)
    split_ratio = [x / total_ratio for x in split_ratio]
    dataset_len = len(dataset)
    dataset_split_len = [int(x * dataset_len) for x in split_ratio]
    dataset_split_len[-1] = int(dataset_len - sum(dataset_split_len) + dataset_split_len[-1])
    return random_split(dataset=dataset, lengths=dataset_split_len, generator=torch.Generator().manual_seed(seed))