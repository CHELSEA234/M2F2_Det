import csv
import os
import random
import cv2
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import h5py
from torch.utils.data import Dataset

from .process import get_image_transformation_from_cfg, get_video_transformation_from_cfg
from .utils import get_default_transformation_cfg

class FolderDataset(Dataset):
    def __init__(self, data_root, supported_exts, transform_cfg, selected_cls_labels=None):
        super(FolderDataset, self).__init__()
        self.data_root = data_root
        self.supported_exts = supported_exts
        self.transform_cfg = transform_cfg
        self.selected_cls_labels = selected_cls_labels
        
    def initialize_data_info(self):
        self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
        self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        
    def __find_classes(self, dir, selected_cls_labels):
        if selected_cls_labels is not None:    # use the provided class and label pairs.
            classes = [d[0] for d in selected_cls_labels]
            class_to_idx = {d[0]: d[1] for d in selected_cls_labels}
        else:    # use all classes
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def __make_dataset(self, dir, classes, class_to_idx):
        data_info = []
        for cls in classes:
            for root, _, names in os.walk(os.path.join(dir,cls)):
                for name in sorted(names):
                    if os.path.splitext(name)[-1].lower() in self.supported_exts:
                        data_info.append((os.path.join(root, name), class_to_idx[cls]))
        return data_info
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        raise NotImplementedError
        

class ImageFolderDataset(FolderDataset):
    def __init__(self, 
                 data_root, 
                 transform_cfg=get_default_transformation_cfg(), 
                 selected_cls_labels=None,    # only folders in the dict are loaded.
                 supported_exts=['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp']):
        super(ImageFolderDataset, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.initialize_data_info()
    
    def __getitem__(self, idx):
        """
           image: [C, H, W]
           label: [1]
        """
        transform = get_image_transformation_from_cfg(self.transform_cfg)
        img_pil = Image.open(self.data_info[idx][0]).convert('RGB')
        img_cls = torch.tensor(self.data_info[idx][1], dtype=torch.int64)
        img_t = transform(img_pil)
        
        return img_t, img_cls
    

class ImageFolderDatasetSplitFn(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 selected_cls_labels=None, 
                 supported_exts=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'],
                 split_file=None,
                 return_fn=True):
        super(ImageFolderDatasetSplitFn, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        self.initialize_data_info()
        
    def __getitem__(self, idx):
        """
        image: [C, H, W]
        label: [1]
        """
        transform = get_image_transformation_from_cfg(self.transform_cfg)
        img_pil = Image.open(self.data_info[idx][0]).convert('RGB')
        img_cls = torch.tensor(self.data_info[idx][1], dtype=torch.int64)
        img_t = transform(img_pil)
        
        return self.data_info[idx][0], img_t, img_cls
    
class ImageFolderH5Dataset():
    def __init__(self, 
                 data_root, 
                 transform_cfg=get_default_transformation_cfg(),
                 split_fn=None,
                 co='c40'
                 ):
        super(ImageFolderH5Dataset, self).__init__()
        self.data_root = data_root
        self.transform_cfg = transform_cfg
        self.split_fn = split_fn
        self.initialize_data_info(data_root, split_fn)
    
    def initialize_data_info(self, data_root, split_fn, co='c40'):
        splits = json.load(open(split_fn, 'r')) if split_fn is not None else None
        self.class_to_idx = {}
        self.h5_datasets = []
        self.h5_datasets_ele_info = []
        self.h5_datasets_len = []
        datasets = ['original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face']
        for idx, dname in enumerate(datasets):
            self.class_to_idx[dname] = idx
            dataset = h5py.File(os.path.join(data_root, f'FF++_{dname}_{co}.h5'), 'r')
            self.h5_datasets.append(dataset)
            if splits is None:
                self.h5_datasets_ele_info.append(d.keys())
            else:
                if idx == 0:    # original
                    ele_keys = [x for sublist in splits for x in sublist]
                else:
                    tmp_ele_keys = list(map(lambda x:["_".join([x[0],x[1]]),"_".join([x[1],x[0]])], splits))
                    ele_keys = [x for sublist in tmp_ele_keys for x in sublist]
                self.h5_datasets_ele_info.append(ele_keys)
            self.h5_datasets_len.append(len(ele_keys))
    
    def __len__(self):
        return sum(self.h5_datasets_len)
        
    def __getitem__(self, idx):
        """
           image: [C, H, W]
           label: [1]
        """
        transform = get_image_transformation_from_cfg(self.transform_cfg)
        h5_id = 0
        for h5_dataset_len in self.h5_datasets_len:
            if idx >= h5_dataset_len:
                h5_id += 1
                idx -= h5_dataset_len
        h5_handler = self.h5_datasets[h5_id]
        clip = h5_handler[self.h5_datasets_ele_info[h5_id][idx]][:]
        frame_count = clip.shape[0]
        frame = clip[np.random.randint(0, frame_count - 0.5), :, :, :]
        img_pil = Image.fromarray(frame)
        img_cls = torch.tensor(0, dtype=torch.int64) if h5_id == 0 else torch.tensor(1, dtype=torch.int64)
        img_t = transform(img_pil)
        
        return img_t, img_cls

class ImageFolderH5Dataset_inference():
    def __init__(self, 
                 data_root, 
                 transform_cfg=get_default_transformation_cfg(),
                 split_fn=None,
                 co='c40'
                 ):
        super(ImageFolderH5Dataset_inference, self).__init__()
        self.data_root = data_root
        self.transform_cfg = transform_cfg
        self.split_fn = split_fn
        self.transform = get_image_transformation_from_cfg(self.transform_cfg)

        splits = json.load(open(split_fn, 'r')) if split_fn is not None else None
        self.image_lst = []
        self.h5_datasets = []
        self.datasets = ['original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face']
        for idx, dname in enumerate(self.datasets):
            dataset = h5py.File(os.path.join(data_root, f'FF++_{dname}_{co}_test_only.h5'), 'r')
            self.h5_datasets.append(dataset)
            if idx == 0:    # original
                ele_keys = [x for sublist in splits for x in sublist]
            else:
                tmp_ele_keys = list(map(lambda x:["_".join([x[0],x[1]]),"_".join([x[1],x[0]])], splits))
                ele_keys = [x for sublist in tmp_ele_keys for x in sublist]
            for key in ele_keys:
                for frame_idx in range(100):
                # for frame_idx in np.linspace(0, 99, 10):
                    value = str(idx) + '_' + key + '_' + str(int(frame_idx))
                    self.image_lst.append(value)

    def __len__(self):
        return len(self.image_lst)
        
    def __getitem__(self, idx):
        """
           image: [C, H, W]
           label: [1]
        """
        img_id = self.image_lst[idx]
        underscore_count = img_id.count('_')
        if underscore_count == 2:
            idx, key, frame_idx = img_id.split('_')
            img_cls = torch.tensor(0, dtype=torch.int64)
        else:
            idx, key_1, key_2, frame_idx = img_id.split('_')
            key = "_".join([key_1, key_2])
            img_cls = torch.tensor(1, dtype=torch.int64)
        frame = self.h5_datasets[int(idx)][key][int(frame_idx)]
        img_pil = Image.fromarray(frame)
        img_t = self.transform(img_pil)
        return img_t, img_cls, img_id