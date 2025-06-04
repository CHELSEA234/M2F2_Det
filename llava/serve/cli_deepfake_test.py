import argparse
import torch
import os
import h5py
import json
import random
import numpy as np


from torch import nn
from torchvision.transforms import transforms
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEEPFAKE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_deepfake_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, tokenizer_hybrid_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    # disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_deepfake_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    model.eval()
    model = model.to('cuda')
    # model.load_deepfake_encoder('/home/aya/workspace/workshop/replicate/Rec_Video_Det/expts_val/0001_first_try/model_ckpt.pth', verbose=True)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    
    h5_dataset_root = "/user/guoxia11/cvlshare/cvl-guoxia11/FaceForensics_HiFiNet"
    h5_datasets = {}
    datasets = ['original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face']
    for dname in datasets:
        h5_datasets[dname] = h5py.File(os.path.join(h5_dataset_root, f'FF++_{dname}_c40.h5'), 'r')
    with open('/research/cvl-guoxia11/deepfake_AIGC/FaceForensics/dataset/splits/train.json', 'r') as f_json:
        img_folders = json.load(f_json)
    
    inp = 'next'
    while True:
        if inp == 'next':
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            dname = random.choice(datasets)
            handler = h5_datasets[dname]
            label = 0 if dname == 'original' else 1
            if label == 0:
                img_keys = [x for sublist in img_folders for x in sublist]
            else:
                tmp_img_folders = list(map(lambda x:["_".join([x[0],x[1]]),"_".join([x[1],x[0]])], img_folders))
                img_keys = [x for sublist in tmp_img_folders for x in sublist]
            selected_img_key = random.choice(img_keys)
            data = handler[selected_img_key]
            data = torch.tensor(data[:])    # [L, 224, 224, 3]
            frame_count = random.choice(list(range(data.size(0))))
            frame = data[frame_count]
            image = Image.fromarray(np.uint8(frame))
            image_record = image
            image_size = image.size
            print(f'Val {dname}: {selected_img_key}_{frame_count}')
            inp = ''
        else:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("Input \"next\" to test a new image.")
                continue
            print(f"{roles[1]}: ", end="")

            if image_record is not None:
                # first message
                # if model.config.mm_use_im_start_end:
                #     inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                # else:
                #     # inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                #     inp = inp
                conv.append_message(conv.roles[0], inp)
                image_record = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_hybrid_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    images=[image],
                    image_sizes=[image_size],
                    deepfake_inputs=[image],
                    do_sample=False,
                    streamer=streamer,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True)
            output_ids = output['sequences']

            outputs = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = outputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
