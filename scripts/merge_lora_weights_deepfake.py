import argparse
from llava.model.builder import load_deepfake_model
from llava.mm_utils import get_model_name_from_path
from peft import PeftModel
from torch import nn
import torch

def init_deepfake_branch(model):
    components = [model.deepfake_projector]
    for component in components:
        for m in component.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param)
    
        
def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_deepfake_model(args.model_path, args.model_base, model_name, device_map='cpu')
    model.load_deepfake_encoder(model.config.deepfake_model_path, verbose=True)
    # init_deepfake_branch(model)
    model.save_pretrained(args.save_model_path, safe_serialization=True)
    tokenizer.save_pretrained(args.save_model_path)
    return model

def load_model(model_path):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_deepfake_model(model_path, None, model_name, device_map='cpu')
    # model.load_deepfake_encoder('/home/songxiufeng/weights/dense_deepfake/model_ckpt.pth', verbose=True)
    # init_deepfake_branch(model)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    model = merge_lora(args)
