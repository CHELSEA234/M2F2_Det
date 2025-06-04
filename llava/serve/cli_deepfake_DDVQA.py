import argparse
import torch
import os
import h5py
import json
import random
import numpy as np

from tqdm import tqdm
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
    model.load_deepfake_encoder(model.config.deepfake_model_path, verbose=True)    

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
    
    # eccv_dataset_root = '/home1/songxiufeng/data/ff++_eccv/images'
    eccv_dataset_root = '/user/guoxia11/cvlshare/cvl-guoxia11/M2F2_Det/image_text_pair/M2F2_json/DDVQA/images'
    images = sorted(os.listdir(eccv_dataset_root))
    out_dir = 'outputs/deepfake'
    os.makedirs(out_dir, exist_ok=True)
    
    for image_idx, image_fn in tqdm(enumerate(images)):
        image = load_image(os.path.join(eccv_dataset_root, image_fn))
        image_size = image.size
 
        prompt = "Assistant: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n <deepfake>\n Is the image real or fake? ###Assistant:"
        # prompt = "Assistant: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n <deepfake>\n Determine the authenticity of this image ###Assistant:"
        input_ids = tokenizer_hybrid_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=[image],
                image_sizes=[image_size],
                deepfake_inputs=[image],
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True)
        output_ids = output['sequences']

        outputs = tokenizer.decode(output_ids[0]).strip()
        # with open(os.path.join(out_dir, 'llava_val.txt'), 'a') as f:
        #     # f.write(f"{{\"image\": \"{image_fn}\", \"response\": \"{outputs[:128]}\"}}\n")
        #     f.write(f"{{\"image\": \"{image_fn}\", \"response\": \"{outputs}\"}}\n")

        answer = {
            "image": image_fn,
            # "question_id": question_id,
            "prompt": prompt,
            "text": outputs,
            # "model_id": model_id,
            "metadata": {}
        }
        with open(os.path.join(out_dir, 'demo.jsonl'), 'a') as f:
            f.write(json.dumps(answer) + '\n')

        if image_idx > 10:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/user/guoxia11/cvlshare/cvl-guoxia11/M2F2_Det/llava-1.5-7b-densenet121-deepfake")
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
