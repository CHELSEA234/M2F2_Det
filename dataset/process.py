import random
from functools import partial
from io import BytesIO
import cv2
import numpy as np
import torchvision
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.util import random_noise


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    blurred_img = np.zeros_like(img)
    gaussian_filter(img[:,:,0], output=blurred_img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=blurred_img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=blurred_img[:,:,2], sigma=sigma)
    return blurred_img


def gaussian_noise(img, var=0.01):
    noisy_img = random_noise(np.uint8(img), mode='gaussian', var=var, clip=True)
    return np.uint8(noisy_img * 255)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

###########################################
jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def data_augment(img, post_cfg):
    if isinstance(img, Image.Image):
        img = np.array(img)
    blur_cfg = post_cfg.get('blur', None)
    if blur_cfg is not None:
        blur_prob = blur_cfg.get('prob', 1)
        if random.random() < blur_prob:
            blur_sigs = blur_cfg.get('sig', [0.0, 3.0])
            sig = sample_continuous(blur_sigs)
            img = gaussian_blur(img, sig)
    
    jpeg_cfg = post_cfg.get('jpeg', None)
    if jpeg_cfg is not None:
        jpeg_prob = jpeg_cfg.get('prob', 1)
        if random.random() < jpeg_prob:
            jpeg_methods = jpeg_cfg.get('method', ['cv2', 'pil'])
            jpeg_quals = jpeg_cfg.get('qual', [30, 100])
            method = sample_discrete(jpeg_methods)
            qual = sample_discrete(jpeg_quals)
            img = jpeg_from_key(img, qual, method)
    
    noise_cfg = post_cfg.get('noise', None)
    if noise_cfg is not None:
        noise_prob = noise_cfg.get('prob', 1)
        if random.random() < noise_prob:
            noise_vars = noise_cfg.get('var', [0.01, 0.02])
            var = sample_continuous(noise_vars)
            img = gaussian_noise(img, var)
    return Image.fromarray(np.uint8(img))
 
 
def sample_augment_pipeline(post_cfg):
    pipeline = list()
    blur_cfg = post_cfg.get('blur', None)
    if blur_cfg is not None:
        blur_prob = blur_cfg.get('prob', 1)
        if random.random() < blur_prob:
            blur_sigs = blur_cfg.get('sig', [0.0, 3.0])
            sig = sample_continuous(blur_sigs)
            pipeline.append(partial(gaussian_blur, sigma=sig))
    jpeg_cfg = post_cfg.get('jpeg', None)
    if jpeg_cfg is not None:
        jpeg_prob = jpeg_cfg.get('prob', 1)
        if random.random() < jpeg_prob:
            jpeg_quals = jpeg_cfg.get('qual', [30, 100])
            qual = sample_discrete(jpeg_quals)
            jpeg_methods = jpeg_cfg.get('method', ['cv2', 'pil'])
            method = sample_discrete(jpeg_methods)
            pipeline.append(partial(jpeg_from_key, compress_val=qual, key=method))
    noise_cfg = post_cfg.get('noise', None)
    if noise_cfg is not None:
        noise_prob = noise_cfg.get('prob', 1)
        if random.random() < noise_prob:
            noise_vars = noise_cfg.get('var', [0.01, 0.02])
            var = sample_continuous(noise_vars)
            pipeline.append(partial(gaussian_noise, var=var))
    return pipeline


def forward_post_pipeline(img, pipeline):
    if isinstance(img, Image.Image):
        img = np.array(img)
    for post in pipeline:
        img = post(img)
    return Image.fromarray(np.uint8(img))


def random_crop_pipeline(crop_size):
    cropped_area = [-1, -1]    # top, left
    def fixed_random_crop(img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if cropped_area == [-1, -1]:
            H, W, _ = img.shape
            assert(H >= crop_size and W >= crop_size)
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            cropped_area[0] = top
            cropped_area[1] = left
        top, left = cropped_area
        cropped_img = img[top: top + crop_size, left: left + crop_size, :]
        return Image.fromarray(np.uint8(cropped_img))
    return fixed_random_crop 
        
     
def get_image_transformation_from_cfg(pipeline_cfg):
    # resize
    resize_cfg = pipeline_cfg.get('resize', None)
    if resize_cfg is None:
        rz_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        rz_img_size = pipeline_cfg['resize']['img_size']
        if isinstance(rz_img_size, list) or isinstance(rz_img_size, tuple):
            assert(len(rz_img_size) == 2)
            rz_img_h, rz_img_w = rz_img_size
        else:
            rz_img_h, rz_img_w = rz_img_size, rz_img_size
        rz_func = torchvision.transforms.Resize([rz_img_w, rz_img_h])    # [W, H] for PIL Image resize
    
    # post process
    post_cfg = pipeline_cfg.get('post', None)
    if post_cfg is None:
        post_func = torchvision.transforms.Lambda(lambda img: img)
    else:        
        post_func = torchvision.transforms.Lambda(lambda img: data_augment(img, post_cfg))
    
    # crop
    crop_cfg = pipeline_cfg.get('crop', None)
    if crop_cfg is None:
        crop_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        crop_img_size = pipeline_cfg['crop']['img_size']
        crop_type = pipeline_cfg['crop']['type']
        if crop_type == 'center':
            crop_func = torchvision.transforms.CenterCrop(crop_img_size)
        elif crop_type == 'random':
            crop_func = torchvision.transforms.RandomCrop(crop_img_size)
        else:
            raise ValueError(f'Unexpected Crop Type: {crop_type}')
    
    # flip
    flip_enabled = pipeline_cfg.get('flip', False)
    if flip_enabled:
        flip_func = torchvision.transforms.RandomHorizontalFlip()
    else:
        flip_func = torchvision.transforms.Lambda(lambda img: img)
    
    # normalize
    normalize_cfg = pipeline_cfg.get('normalize', None)
    if normalize_cfg is None:
        normalization_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        mean = normalize_cfg.get('mean')
        std = normalize_cfg.get('std')
        normalization_func = torchvision.transforms.Normalize(mean=mean, std=std)
        
    return torchvision.transforms.Compose([
        rz_func,
        post_func,
        crop_func,
        flip_func,
        torchvision.transforms.ToTensor(),
        normalization_func
    ])


def get_video_transformation_from_cfg(pipeline_cfg):
    # resize
    resize_cfg = pipeline_cfg.get('resize', None)
    if resize_cfg is None:
        rz_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        rz_img_size = pipeline_cfg['resize']['img_size']
        if isinstance(rz_img_size, list) or isinstance(rz_img_size, tuple):
            assert(len(rz_img_size) == 2)
            rz_img_h, rz_img_w = rz_img_size
        else:
            rz_img_h, rz_img_w = rz_img_size, rz_img_size
        rz_func = torchvision.transforms.Resize([rz_img_w, rz_img_h])    # [W, H] for PIL Image resize
    
    # post process
    post_cfg = pipeline_cfg.get('post', None)
    if post_cfg is None:
        post_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        post_pipeline = sample_augment_pipeline(post_cfg)
        post_func = torchvision.transforms.Lambda(lambda img: forward_post_pipeline(img, post_pipeline))
    
    # crop
    crop_cfg = pipeline_cfg.get('crop', None)
    if crop_cfg is None:
        crop_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        crop_img_size = pipeline_cfg['crop']['img_size']
        crop_type = pipeline_cfg['crop']['type']
        if crop_type == 'center':
            crop_func = torchvision.transforms.CenterCrop(crop_img_size)
        elif crop_type == 'random':
            crop_pipeline = random_crop_pipeline(crop_img_size)
            crop_func = torchvision.transforms.Lambda(lambda img: crop_pipeline(img))
        else:
            raise ValueError(f'Unexpected Crop Type: {crop_type}')
    
    # flip. fixed prob = 0.5
    flip_enabled = pipeline_cfg.get('flip', False)
    if flip_enabled:
        if random.random() < 0.5:
            flip_func = torchvision.transforms.RandomHorizontalFlip(p=1)
        else:
            flip_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        flip_func = torchvision.transforms.Lambda(lambda img: img)
    
    # normalize
    normalize_cfg = pipeline_cfg.get('normalize', None)
    if normalize_cfg is None:
        normalization_func = torchvision.transforms.Lambda(lambda img: img)
    else:
        mean = normalize_cfg.get('mean')
        std = normalize_cfg.get('std')
        normalization_func = torchvision.transforms.Normalize(mean=mean, std=std)
    
    return torchvision.transforms.Compose([
        rz_func,
        post_func,
        crop_func,
        flip_func,
        torchvision.transforms.ToTensor(),
        normalization_func
    ])