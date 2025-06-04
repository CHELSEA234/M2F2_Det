import torch

from transformers import CLIPImageProcessor, CLIPVisionModel
from torch import nn


class CLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/clip-vit-large-patch14-336",
        dtype: torch.dtype = torch.float32,
    ):  
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
        self.processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.select_layer = -2    
        self.hidden_size = self.model.config.hidden_size
        for p in self.model.parameters():
            p.requires_grad = False
        print('Freezing CLIP vision encoder.')
        self.dtype = dtype
        self.model.to(dtype)
        
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ):
        inputs_embeds = inputs_embeds.to(self.dtype)        
        outputs = self.model(inputs_embeds, output_hidden_states=True)
        # output = outputs.hidden_states[self.select_layer]   
        return outputs.hidden_states[6][:,:-1,:], outputs.hidden_states[10][:,:-1,:], \
                outputs.hidden_states[14][:,:-1,:], outputs.hidden_states[self.select_layer]