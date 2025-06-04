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
    
    
class VideoFolderDataset(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv']):
        super(VideoFolderDataset, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.initialize_data_info()
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, N - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
            
        return torch.stack(frames, dim=0), label


class VideoFolderDatasetCachedForRecons(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.jpg']):
        super(VideoFolderDatasetCachedForRecons, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
        self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
    
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
            ddir = os.path.join(dir, cls)
            fds = os.listdir(ddir)
            assert('original' in fds and 'recons' in fds)
            data_elems = dict()
            for name in sorted(os.listdir(os.path.join(ddir, 'original'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    prefix = fn.rsplit('_', maxsplit=1)[0]
                    if prefix not in data_elems:
                        data_elems[prefix] = 1
                    else:
                        data_elems[prefix] += 1
            for prefix, count in data_elems.items():
                data_info.append(((ddir, prefix, count), class_to_idx[cls]))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_ddir, video_prefix, video_length = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        if self.sample_size > video_length:
            raise ValueError(f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {os.path.join(video_ddir, video_prefix)}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(1, video_length + 1), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(1, video_length - self.sample_size + 1)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(1, video_length + 1))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        original_frames, recons_frames = [], []
        for frame_index in sample_index:
            original_frame = Image.open(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_frame = transform(original_frame)
            original_frames.append(transformed_frame)
            recons_frame = Image.open(os.path.join(video_ddir, 'recons', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_recons_frame = transform(recons_frame)
            recons_frames.append(transformed_recons_frame)
       
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_index = random.choice(range(len(original_frames)))
                repeated_original_frames = []
                for i in range(len(original_frames)):
                    repeated_original_frames.append(original_frames[repeated_index].clone())
                original_frames = repeated_original_frames
                repeated_recons_frames = []
                for i in range(len(recons_frames)):
                    repeated_recons_frames.append(recons_frames[repeated_index].clone())
                recons_frames = repeated_recons_frames
                
        return (torch.stack(original_frames, dim=0), torch.stack(recons_frames, dim=0)), label
    
    
class VideoFolderDatasetRestricted(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv'],
                 restricted_ref=None):
        super(VideoFolderDatasetRestricted, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.restricted_ref = restricted_ref
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_restricted_data_info()

    
    def initialize_restricted_data_info(self):
        self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
        if self.restricted_ref is None:
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            with open(self.restricted_ref) as ref:
                restricted_items = json.load(ref)
                self.data_info = self.__make_restricted_dataset(self.data_root, self.classes, self.class_to_idx, restricted_items)
                
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
    
    def __make_restricted_dataset(self, dir, classes, class_to_idx, restricted_items):
        data_info = []
        for cls in classes:
            for root, _, names in os.walk(os.path.join(dir,cls)):
                for name in sorted(names):
                    if os.path.splitext(name)[-1].lower() in self.supported_exts and name in restricted_items:
                        data_info.append((os.path.join(root, name), class_to_idx[cls]))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, N - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
            
        return torch.stack(frames, dim=0), label
    
class VideoFolderDatasetSplit(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv'],
                 split_file=None):
        super(VideoFolderDatasetSplit, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()

    
    def initialize_split_data_info(self):
        if self.split_file is None:
            self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
                
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
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_info.append((os.path.join(data_root, row['file_name']), int(row['label'])))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, N - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
            
        return torch.stack(frames, dim=0), label

    

class VideoFolderDatasetSplitFn(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv'],
                 split_file=None,
                 return_fn=True):
        super(VideoFolderDatasetSplitFn, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        self.return_fn = return_fn
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()

    
    def initialize_split_data_info(self):
        if self.split_file is None:
            self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
                
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
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_info.append((os.path.join(data_root, row['file_name']), int(row['label'])))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, N - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
            
        return video_path, torch.stack(frames, dim=0), label
    
    
class VideoFolderDatasetSplitFixedSample(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv'],
                 split_file=None):
        super(VideoFolderDatasetSplitFixedSample, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()

    
    def initialize_split_data_info(self):
        if self.split_file is None:
            self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
                
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
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_info.append((os.path.join(data_root, row['file_name']), int(row['label'])))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, 4)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
            
        return torch.stack(frames, dim=0), label
    
    
class VideoFolderDatasetCachedForReconsSplit(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.jpg'],
                 split_file=None):
        super(VideoFolderDatasetCachedForReconsSplit, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()
    
    def initialize_split_data_info(self):
        if self.split_file is None:
            self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
         
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
            ddir = os.path.join(dir, cls)
            fds = os.listdir(ddir)
            assert('original' in fds and 'recons' in fds)
            data_elems = dict()
            for name in sorted(os.listdir(os.path.join(ddir, 'original'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    prefix = fn.rsplit('_', maxsplit=1)[0]
                    if prefix not in data_elems:
                        data_elems[prefix] = 1
                    else:
                        data_elems[prefix] += 1
            for prefix, count in data_elems.items():
                data_info.append(((ddir, prefix, count), class_to_idx[cls]))
        return data_info
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            all_dir_prefixs = dict()
            for row in csv_reader:
                file_name = row['file_name']
                label = int(row['label'])
                base_path = os.path.dirname(file_name)
                fn = os.path.basename(file_name)
                fn_prefix = os.path.splitext(fn)[0]
                if base_path not in all_dir_prefixs:
                    all_dir_prefixs[base_path] = list()
                all_dir_prefixs[base_path].append((fn_prefix, label))
            for base_dir, prefix_info in all_dir_prefixs.items():
                data_elems = dict()
                prefixs = list()
                for prefix_pair in prefix_info:
                    prefixs.append(prefix_pair[0])
                for name in sorted(os.listdir(os.path.join(data_root, base_dir, 'recons'))):
                    fn, ext = os.path.splitext(name)
                    if ext in self.supported_exts:
                        fn_prefix = fn.rsplit('_', maxsplit=1)[0]
                        if fn_prefix in prefixs:
                            if fn_prefix not in data_elems:
                                data_elems[fn_prefix] = 1
                            else:
                                data_elems[fn_prefix] += 1
                for prefix_pair in prefix_info:
                    fn_prefix, label = prefix_pair
                    data_info.append(((os.path.join(data_root, base_dir), fn_prefix, data_elems[fn_prefix]), label))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_ddir, video_prefix, video_length = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        if self.sample_size > video_length:
            raise ValueError(f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {os.path.join(video_ddir, video_prefix)}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(1, video_length + 1), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(1, video_length - self.sample_size + 1)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(1, video_length + 1))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        original_frames, recons_frames = [], []
        for frame_index in sample_index:
            original_frame = Image.open(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_frame = transform(original_frame)
            original_frames.append(transformed_frame)
            recons_frame = Image.open(os.path.join(video_ddir, 'recons', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_recons_frame = transform(recons_frame)
            recons_frames.append(transformed_recons_frame)
       
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_index = random.choice(range(len(original_frames)))
                repeated_original_frames = []
                for i in range(len(original_frames)):
                    repeated_original_frames.append(original_frames[repeated_index].clone())
                original_frames = repeated_original_frames
                repeated_recons_frames = []
                for i in range(len(recons_frames)):
                    repeated_recons_frames.append(recons_frames[repeated_index].clone())
                recons_frames = repeated_recons_frames
                
        return (torch.stack(original_frames, dim=0), torch.stack(recons_frames, dim=0)), label
    
    
class VideoFolderDatasetCachedForReconsSplitFn(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.jpg'],
                 split_file=None):
        super(VideoFolderDatasetCachedForReconsSplitFn, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        if sample_method not in ['random', 'continuous', 'entire']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()
    
    def initialize_split_data_info(self):
        if self.split_file is None:
            self.classes, self.class_to_idx = self.__find_classes(self.data_root, self.selected_cls_labels)
            self.data_info = self.__make_dataset(self.data_root, self.classes, self.class_to_idx)    # [(path, label), ...]
        else:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
         
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
            ddir = os.path.join(dir, cls)
            fds = os.listdir(ddir)
            assert('original' in fds and 'recons' in fds)
            data_elems = dict()
            for name in sorted(os.listdir(os.path.join(ddir, 'original'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    prefix = fn.rsplit('_', maxsplit=1)[0]
                    if prefix not in data_elems:
                        data_elems[prefix] = 1
                    else:
                        data_elems[prefix] += 1
            for prefix, count in data_elems.items():
                data_info.append(((ddir, prefix, count), class_to_idx[cls]))
        return data_info
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            all_dir_prefixs = dict()
            for row in csv_reader:
                file_name = row['file_name']
                label = int(row['label'])
                base_path = os.path.dirname(file_name)
                fn = os.path.basename(file_name)
                fn_prefix = os.path.splitext(fn)[0]
                if base_path not in all_dir_prefixs:
                    all_dir_prefixs[base_path] = list()
                all_dir_prefixs[base_path].append((fn_prefix, label))
            for base_dir, prefix_info in all_dir_prefixs.items():
                data_elems = dict()
                prefixs = list()
                for prefix_pair in prefix_info:
                    prefixs.append(prefix_pair[0])
                for name in sorted(os.listdir(os.path.join(data_root, base_dir, 'recons'))):
                    fn, ext = os.path.splitext(name)
                    if ext in self.supported_exts:
                        fn_prefix = fn.rsplit('_', maxsplit=1)[0]
                        if fn_prefix in prefixs:
                            if fn_prefix not in data_elems:
                                data_elems[fn_prefix] = 1
                            else:
                                data_elems[fn_prefix] += 1
                for prefix_pair in prefix_info:
                    fn_prefix, label = prefix_pair
                    data_info.append(((os.path.join(data_root, base_dir), fn_prefix, data_elems[fn_prefix]), label))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_ddir, video_prefix, video_length = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        if self.sample_size > video_length:
            raise ValueError(f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {os.path.join(video_ddir, video_prefix)}")
        
        sample_index = []
        if self.sample_method == "random":
            sample_index = random.sample(range(1, video_length + 1), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(1, video_length - self.sample_size + 1)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(1, video_length + 1))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        original_frames, recons_frames = [], []
        for frame_index in sample_index:
            original_frame = Image.open(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_frame = transform(original_frame)
            original_frames.append(transformed_frame)
            recons_frame = Image.open(os.path.join(video_ddir, 'recons', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_recons_frame = transform(recons_frame)
            recons_frames.append(transformed_recons_frame)
       
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_index = random.choice(range(len(original_frames)))
                repeated_original_frames = []
                for i in range(len(original_frames)):
                    repeated_original_frames.append(original_frames[repeated_index].clone())
                original_frames = repeated_original_frames
                repeated_recons_frames = []
                for i in range(len(recons_frames)):
                    repeated_recons_frames.append(recons_frames[repeated_index].clone())
                recons_frames = repeated_recons_frames
                
        return os.path.join(video_ddir, f'{video_prefix}__{sample_index[0]}'), (torch.stack(original_frames, dim=0), torch.stack(recons_frames, dim=0)), label
    
    
class VideoFolderDatasetSplitFixedFrame(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='fixed',    # ['random', 'continuous', 'entire', 'fixed]
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv'],
                 split_file=None,
                 sample_file=None
                 ):
        super(VideoFolderDatasetSplitFixedFrame, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        self.sample_file = sample_file
        if sample_method not in ['random', 'continuous', 'entire', 'fixed']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()

    
    def initialize_split_data_info(self):
        if self.sample_file is not None:
            self.data_info = self.__make_fixed_sample_dataset(self.data_root, self.sample_file)
        elif self.split_file is not None:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
        else:
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
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_info.append((os.path.join(data_root, row['file_name']), int(row['label'])))
        return data_info
    
    def __make_fixed_sample_dataset(self, data_root, sample_file):
        data_info = []
        with open(sample_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_info.append((os.path.join(data_root, row['file_name']), int(row['label']), int(row['index'])))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_method != "fixed" and self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == 'fixed':
            assert(self.sample_file is not None)
            sample_start = video[2]
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, N - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
            
        return torch.stack(frames, dim=0), label
    

class VideoFolderDatasetSplitFnFixedFrame(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire', 'fixed']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.mp4', '.avi', '.wmv', '.mkv', '.flv'],
                 split_file=None,
                 sample_file=None,
                 return_fn=True):
        super(VideoFolderDatasetSplitFnFixedFrame, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        self.sample_file = sample_file
        self.return_fn = return_fn
        if sample_method not in ['random', 'continuous', 'entire', 'fixed']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()

    
    def initialize_split_data_info(self):
        if self.sample_file is not None:
            self.data_info = self.__make_fixed_sample_dataset(self.data_root, self.sample_file)
        elif self.split_file is not None:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
        else:
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
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_info.append((os.path.join(data_root, row['file_name']), int(row['label'])))
        return data_info
    
    def __make_fixed_sample_dataset(self, data_root, sample_file):
        data_info = []
        with open(sample_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data_info.append((os.path.join(data_root, row['file_name']), int(row['label']), int(row['index'])))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_path = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)
        cap = cv2.VideoCapture(video_path)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_method != "fixed" and self.sample_size > N:
            print(f"The sample size {self.sample_size} is longer than the total frame number {N} of video {video_path}")
        
        sample_index = []
        if self.sample_method == 'fixed':
            assert(self.sample_file is not None)
            sample_start = video[2]
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "random":
            sample_index = random.sample(range(N), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(0, N - self.sample_size)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(0, N))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        count = 0
        frames = []
        for frame_index in range(N):
            _, frameread = cap.read()
            if frame_index in sample_index:
                frameread = cv2.cvtColor(frameread, cv2.COLOR_BGR2RGB)
                frameread = transform(Image.fromarray(np.uint8(frameread)))
                frames.append(frameread)
                count += 1
            if count >= len(sample_index):
                break
        cap.release()
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_frame = random.choice(frames)
                repeated_frames = []
                for i in range(len(frames)):
                    repeated_frames.append(repeated_frame.clone())
                frames = repeated_frames
        return f'{video_path}__{sample_index[0]}', torch.stack(frames, dim=0), label
    

class VideoFolderDatasetCachedForReconsSplitFixedFrame(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='continuous',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.jpg'],
                 split_file=None,
                 sample_file=None):
        super(VideoFolderDatasetCachedForReconsSplitFixedFrame, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        self.sample_file = sample_file
        if sample_method not in ['random', 'continuous', 'entire', 'fixed']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()
    
    def initialize_split_data_info(self):
        if self.sample_file is not None:
            self.data_info = self.__make_fixed_sample_dataset(self.data_root, self.sample_file)
        elif self.split_file is not None:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
        else:
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
            ddir = os.path.join(dir, cls)
            fds = os.listdir(ddir)
            assert('original' in fds and 'recons' in fds)
            data_elems = dict()
            for name in sorted(os.listdir(os.path.join(ddir, 'original'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    prefix = fn.rsplit('_', maxsplit=1)[0]
                    if prefix not in data_elems:
                        data_elems[prefix] = 1
                    else:
                        data_elems[prefix] += 1
            for prefix, count in data_elems.items():
                data_info.append(((ddir, prefix, count), class_to_idx[cls]))
        return data_info
    
    def __make_fixed_sample_dataset(self, data_root, sample_file):
        data_info = []
        with open(sample_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                file_name = row['file_name']
                label = int(row['label'])
                index = int(row['index'])
                base_path = os.path.dirname(file_name)
                fn = os.path.basename(file_name)
                fn_prefix = os.path.splitext(fn)[0]
                data_info.append(((os.path.join(data_root, base_path), fn_prefix, index), label))
        return data_info
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            all_dir_prefixs = dict()
            for row in csv_reader:
                file_name = row['file_name']
                label = int(row['label'])
                base_path = os.path.dirname(file_name)
                fn = os.path.basename(file_name)
                fn_prefix = os.path.splitext(fn)[0]
                if base_path not in all_dir_prefixs:
                    all_dir_prefixs[base_path] = list()
                all_dir_prefixs[base_path].append((fn_prefix, label))
            for base_dir, prefix_info in all_dir_prefixs.items():
                data_elems = dict()
                prefixs = list()
                for prefix_pair in prefix_info:
                    prefixs.append(prefix_pair[0])
                for name in sorted(os.listdir(os.path.join(data_root, base_dir, 'recons'))):
                    fn, ext = os.path.splitext(name)
                    if ext in self.supported_exts:
                        fn_prefix = fn.rsplit('_', maxsplit=1)[0]
                        if fn_prefix in prefixs:
                            if fn_prefix not in data_elems:
                                data_elems[fn_prefix] = 1
                            else:
                                data_elems[fn_prefix] += 1
                for prefix_pair in prefix_info:
                    fn_prefix, label = prefix_pair
                    data_info.append(((os.path.join(data_root, base_dir), fn_prefix, data_elems[fn_prefix]), label))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_ddir, video_prefix, video_length = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        if self.sample_method != "fixed" and self.sample_size > video_length:
            raise ValueError(f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {os.path.join(video_ddir, video_prefix)}")
        
        sample_index = []
        if self.sample_method == "fixed":
            sample_start = video_length + 1
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "random":
            sample_index = random.sample(range(1, video_length + 1), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(1, video_length - self.sample_size + 1)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(1, video_length + 1))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        original_frames, recons_frames = [], []
        for frame_index in sample_index:
            original_frame = Image.open(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_frame = transform(original_frame)
            original_frames.append(transformed_frame)
            recons_frame = Image.open(os.path.join(video_ddir, 'recons', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_recons_frame = transform(recons_frame)
            recons_frames.append(transformed_recons_frame)
       
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_index = random.choice(range(len(original_frames)))
                repeated_original_frames = []
                for i in range(len(original_frames)):
                    repeated_original_frames.append(original_frames[repeated_index].clone())
                original_frames = repeated_original_frames
                repeated_recons_frames = []
                for i in range(len(recons_frames)):
                    repeated_recons_frames.append(recons_frames[repeated_index].clone())
                recons_frames = repeated_recons_frames
                
        return (torch.stack(original_frames, dim=0), torch.stack(recons_frames, dim=0)), label
    
    
class VideoFolderDatasetCachedForReconsSplitFnFixedSample(FolderDataset):
    def __init__(self, 
                 data_root, 
                 sample_size=10, 
                 sample_method='fixed',    # ['random', 'continuous', 'entire']
                 transform_cfg=get_default_transformation_cfg(),
                 repeat_sample_prob = 0.,
                 selected_cls_labels=None, 
                 supported_exts=['.jpg'],
                 split_file=None,
                 sample_file=None):
        super(VideoFolderDatasetCachedForReconsSplitFnFixedSample, self).__init__(
            data_root=data_root, 
            supported_exts=supported_exts, 
            transform_cfg=transform_cfg,
            selected_cls_labels=selected_cls_labels)
        
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.repeat_sample_prob = repeat_sample_prob
        self.split_file = split_file
        self.sample_file = sample_file
        if sample_method not in ['random', 'continuous', 'entire', 'fixed']:
            raise ValueError(f"Sample method should be either \"random\" or \"continuous\", but not {self.sample_method}")
        self.initialize_split_data_info()
    
    def initialize_split_data_info(self):
        if self.sample_file is not None:
            self.data_info = self.__make_fixed_sample_dataset(self.data_root, self.sample_file)
        elif self.split_file is not None:
            self.data_info = self.__make_split_dataset(self.data_root, self.split_file)
        else:
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
            ddir = os.path.join(dir, cls)
            fds = os.listdir(ddir)
            assert('original' in fds and 'recons' in fds)
            data_elems = dict()
            for name in sorted(os.listdir(os.path.join(ddir, 'original'))):
                fn, ext = os.path.splitext(name)
                if ext in self.supported_exts:
                    prefix = fn.rsplit('_', maxsplit=1)[0]
                    if prefix not in data_elems:
                        data_elems[prefix] = 1
                    else:
                        data_elems[prefix] += 1
            for prefix, count in data_elems.items():
                data_info.append(((ddir, prefix, count), class_to_idx[cls]))
        return data_info
    
    def __make_split_dataset(self, data_root, split_file):
        data_info = []
        with open(split_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            all_dir_prefixs = dict()
            for row in csv_reader:
                file_name = row['file_name']
                label = int(row['label'])
                base_path = os.path.dirname(file_name)
                fn = os.path.basename(file_name)
                fn_prefix = os.path.splitext(fn)[0]
                if base_path not in all_dir_prefixs:
                    all_dir_prefixs[base_path] = list()
                all_dir_prefixs[base_path].append((fn_prefix, label))
            for base_dir, prefix_info in all_dir_prefixs.items():
                data_elems = dict()
                prefixs = list()
                for prefix_pair in prefix_info:
                    prefixs.append(prefix_pair[0])
                for name in sorted(os.listdir(os.path.join(data_root, base_dir, 'recons'))):
                    fn, ext = os.path.splitext(name)
                    if ext in self.supported_exts:
                        fn_prefix = fn.rsplit('_', maxsplit=1)[0]
                        if fn_prefix in prefixs:
                            if fn_prefix not in data_elems:
                                data_elems[fn_prefix] = 1
                            else:
                                data_elems[fn_prefix] += 1
                for prefix_pair in prefix_info:
                    fn_prefix, label = prefix_pair
                    data_info.append(((os.path.join(data_root, base_dir), fn_prefix, data_elems[fn_prefix]), label))
        return data_info
    
    def __make_fixed_sample_dataset(self, data_root, sample_file):
        data_info = []
        with open(sample_file) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                file_name = row['file_name']
                label = int(row['label'])
                index = int(row['index'])
                base_path = os.path.dirname(file_name)
                fn = os.path.basename(file_name)
                fn_prefix = os.path.splitext(fn)[0]
                data_info.append(((os.path.join(data_root, base_path), fn_prefix, index), label))
        return data_info
    
    def __getitem__(self, idx):
        """
           frames: [L, C, H, W]
           label: [1]
        """
        video = self.data_info[idx]
        video_ddir, video_prefix, video_length = video[0]
        label = torch.tensor(video[1], dtype=torch.int64)
        transform = get_video_transformation_from_cfg(self.transform_cfg)

        # if self.sample_method != "fixed" and self.sample_size > video_length:
        #     raise ValueError(f"The sample size {self.sample_size} is longer than the total frame number {video_length} of video {os.path.join(video_ddir, video_prefix)}")
        
        sample_index = []
        if self.sample_method == "fixed":
            sample_start = video_length + 1
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "random":
            sample_index = random.sample(range(1, video_length + 1), self.sample_size)
        elif self.sample_method == "continuous":
            sample_start = random.randint(1, video_length - self.sample_size + 1)
            sample_index = list(range(sample_start, sample_start + self.sample_size))
        elif self.sample_method == "entire":
            sample_index = list(range(1, 10000000))
            # sample_index = list(range(1, video_length + 1))
        else:
            raise ValueError("Sample method should be one of \"random\" , \"continuous\" or \"entire\". Get {}".format(self.sample_method))
            
        original_frames, recons_frames = [], []
        for frame_index in sample_index:
            if not os.path.exists(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')):
                break
            original_frame = Image.open(os.path.join(video_ddir, 'original', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_frame = transform(original_frame)
            original_frames.append(transformed_frame)
            recons_frame = Image.open(os.path.join(video_ddir, 'recons', video_prefix + f'_{frame_index}.jpg')).convert('RGB')
            transformed_recons_frame = transform(recons_frame)
            recons_frames.append(transformed_recons_frame)
        
        if self.repeat_sample_prob > 0.:
            if random.random() > self.repeat_sample_prob:
                repeated_index = random.choice(range(len(original_frames)))
                repeated_original_frames = []
                for i in range(len(original_frames)):
                    repeated_original_frames.append(original_frames[repeated_index].clone())
                original_frames = repeated_original_frames
                repeated_recons_frames = []
                for i in range(len(recons_frames)):
                    repeated_recons_frames.append(recons_frames[repeated_index].clone())
                recons_frames = repeated_recons_frames
                
        return os.path.join(video_ddir, f'{video_prefix}__{sample_index[0]}'), (torch.stack(original_frames, dim=0), torch.stack(recons_frames, dim=0)), label
    
    
class ImageFolderH5Dataset():
    def __init__(self, 
                 data_root, 
                 transform_cfg=get_default_transformation_cfg(),
                 split_fn = None 
                 ):
        super(ImageFolderH5Dataset, self).__init__()
        self.data_root = data_root
        self.transform_cfg = transform_cfg
        self.split_fn = split_fn
        self.initialize_data_info(data_root, split_fn)
    
    def initialize_data_info(self, data_root, split_fn):
        splits = json.load(open(split_fn, 'r')) if split_fn is not None else None
        self.class_to_idx = {}
        self.h5_datasets = []
        self.h5_datasets_ele_info = []
        self.h5_datasets_len = []
        datasets = ['original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face']
        for idx, dname in enumerate(datasets):
            self.class_to_idx[dname] = idx
            dataset = h5py.File(os.path.join(data_root, f'FF++_{dname}_c40.h5'), 'r')
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