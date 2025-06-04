import numpy as np
import torch

from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel
from torch import nn
from torch.nn import functional as F
from typing import Optional

from .text_encoder import CLIPTextEncoder
from .vision_encoder import CLIPVisionEncoder


def build_deepfake_backbone(
    model_name: str, 
    feature_dim: Optional[int] = None,
    hidden_size: int = 1024,
    ):
    if 'densenet' in model_name:
        from torchvision.models import densenet
        model_init = {
            "densenet121": densenet.densenet121,
            "densenet161": densenet.densenet161,
            "densenet169": densenet.densenet169,
            "densenet201": densenet.densenet201,
        }
        weights = {
            "densenet121": "DenseNet121_Weights.DEFAULT",
            "densenet161": "DenseNet161_Weights.DEFAULT",
            "densenet169": "DenseNet169_Weights.DEFAULT",
            "densenet201": "DenseNet201_Weights.DEFAULT",
        }
        model = model_init[model_name](weights=weights[model_name]).features
    elif 'efficientnet' in model_name:
        from torchvision.models import efficientnet
        model_init = {
            "efficientnet_b0": efficientnet.efficientnet_b0,
            "efficientnet_b1": efficientnet.efficientnet_b1,
            "efficientnet_b2": efficientnet.efficientnet_b2,
            "efficientnet_b3": efficientnet.efficientnet_b3,
            "efficientnet_b4": efficientnet.efficientnet_b4,
            "efficientnet_b5": efficientnet.efficientnet_b5,
            "efficientnet_b6": efficientnet.efficientnet_b6,
            "efficientnet_b7": efficientnet.efficientnet_b7,
        }
        weights = {
            "efficientnet_b0": "EfficientNet_B0_Weights.DEFAULT",
            "efficientnet_b1": "EfficientNet_B1_Weights.DEFAULT",
            "efficientnet_b2": "EfficientNet_B2_Weights.DEFAULT",
            "efficientnet_b3": "EfficientNet_B3_Weights.DEFAULT",
            "efficientnet_b4": "EfficientNet_B4_Weights.DEFAULT",
            "efficientnet_b5": "EfficientNet_B5_Weights.DEFAULT",
            "efficientnet_b6": "EfficientNet_B6_Weights.DEFAULT",
            "efficientnet_b7": "EfficientNet_B7_Weights.DEFAULT",
        }
        model = model_init[model_name](weights=weights[model_name]).features
    else:
        raise ValueError(f'Unsupported deepfake encoder: {model_name}')
    
    if feature_dim is None:
        model.eval()
        input_t = torch.zeros((1, 3, 224, 224))
        o = model(input_t)
        feature_dim = o.shape[1]
    proj = nn.Sequential(
        nn.Linear(feature_dim, hidden_size),
        nn.LayerNorm(hidden_size)
    )
    return model, proj
    

def get_feature_dim(model_name):
    feature_dims = {
        "densenet121": 1024, "efficientnet_b4": 1792
    }
    if model_name in feature_dims:
        return feature_dims[model_name]
    return None
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x 
    
class M2F2Det(nn.Module):
    def __init__(
        self,
        clip_text_encoder_name: str = "openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
        deepfake_encoder_name: str = 'densenet121',
        hidden_size: int = 1024,
        vision_dtype: torch.dtype = torch.float32,
        text_dtype: torch.dtype = torch.float32,
        deepfake_dtype: torch.dtype = torch.float32,
        load_vision_encoder: bool = True
    ):
        super(M2F2Det, self).__init__()
        self.clip_text_encoder = CLIPTextEncoder(clip_text_encoder_name, dtype=text_dtype)
        if load_vision_encoder:
            self.clip_vision_encoder = CLIPVisionEncoder(clip_vision_encoder_name, dtype=vision_dtype)
        else:
            self.clip_vision_encoder = None
        self.hidden_size = hidden_size
        self.vision_dtype = vision_dtype
        self.text_dtype = text_dtype
        deepfake_feature_dim = get_feature_dim(deepfake_encoder_name)
        self.deepfake_encoder, self.deepfake_proj = build_deepfake_backbone(    
            model_name=deepfake_encoder_name,
            feature_dim=deepfake_feature_dim,
            hidden_size=hidden_size
        )

        # Prepare storage for the outputs
        self.block_outputs = {}
        # Define hook functions to store the outputs from specific blocks
        def get_hook(name):
            def hook(model, input, output):
                self.block_outputs[name] = output
            return hook
        # Register hooks to the 2nd, 3rd, and final blocks
        self.deepfake_encoder[4].register_forward_hook(get_hook("b_1"))
        self.deepfake_encoder[5].register_forward_hook(get_hook("b_2"))
        self.deepfake_encoder[6].register_forward_hook(get_hook("b_3"))
        # self.deepfake_encoder[8].register_forward_hook(get_hook("final_block"))

        self.transformer_blocks = [TransformerEncoderBlock(1024, 8, 2048).to(self.clip_vision_encoder.model.device) for _ in range(3)]
        self.linear_1 = nn.Linear(112, 1024)
        self.linear_2 = nn.Linear(160, 1024)
        self.linear_3 = nn.Linear(272, 1024)
        self.linear_lst = [self.linear_1, self.linear_2, self.linear_3]
        self.link_adapter_proj = nn.Linear(1024, 1)


        self.deepfake_dtype = deepfake_dtype
        self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip_text_encoder.model.config.hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        if self.clip_vision_encoder is not None:
            self.vision_proj = nn.Sequential(
                nn.Linear(self.clip_vision_encoder.model.config.hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
        else:
            # default openai/clip-vit-large-patch14-336
            self.vision_proj = nn.Sequential(
                nn.Linear(1024, hidden_size),
                nn.LayerNorm(hidden_size)
            )
        self.clip_vision_alpha = nn.Parameter(torch.tensor(0.5))
        self.clip_text_alpha = nn.Parameter(torch.tensor(4.0))
        self.output = nn.Linear(2 * hidden_size + 697, 2)
        # self.output = nn.Linear(2 * hidden_size, 2)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_vision_encoder_name)
        
        self.deepfake_encoder.to(deepfake_dtype)
        self.deepfake_proj.to(deepfake_dtype)
        self.text_proj.to(text_dtype)
        self.vision_proj.to(vision_dtype)
        self.clip_vision_alpha.to(vision_dtype)
        self.clip_text_alpha.to(text_dtype)
        self.output.to(deepfake_dtype)
        new_components = [self.text_proj, self.vision_proj, self.output]
        for new_component in new_components:
            for m in new_component.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0)
        
        self.cached_clip_text_features = None

    def forward(
        self,
        images: torch.tensor,
        clip_vision_features: Optional[torch.tensor] = None,
        use_cached_clip_text_features: bool = False,
        return_dict: bool = False,
    ):
        new_embeds = []
        for i in images:   
            i = Image.fromarray(np.uint8(i.cpu().permute(1, 2, 0) * 255.0)) 
            image = self.image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0]
            new_embeds.append(image)
        new_embeds = torch.stack(new_embeds) 
        B, C, H, W = new_embeds.shape
        if clip_vision_features is None:    
            clip_0, clip_1, clip_2, clip_vision_features = self.clip_vision_encoder(new_embeds.to(self.clip_vision_encoder.model.device))   
        if use_cached_clip_text_features:
            if self.cached_clip_text_features is None:
                self.cached_clip_text_features = clip_text_features = self.clip_text_encoder()
            else:
                clip_text_features = self.cached_clip_text_features
        else:        
            clip_text_features = self.clip_text_encoder()   
        clip_vision_features = clip_vision_features.to(self.vision_proj[0].weight.dtype)
        clip_vision_features = self.vision_proj(clip_vision_features)   
        clip_text_features = self.text_proj(clip_text_features)
        
        clip_vision_cls, clip_vision_patches = clip_vision_features[:, 0, :], clip_vision_features[:, 1:, :]
        deepfake_features = self.deepfake_encoder(new_embeds.to(self.deepfake_dtype).to(next(self.deepfake_encoder.parameters()).device))  
        
        deepfake_feat_0 = self.block_outputs["b_1"]
        deepfake_feat_1 = self.block_outputs["b_2"]
        deepfake_feat_2 = self.block_outputs["b_3"]

        # Process each pair of tensors from A and B
        outputs = []
        a = [deepfake_feat_0, deepfake_feat_1, deepfake_feat_2]
        b = [clip_0, clip_1, clip_2]
        
        for i in range(3):
            a[i] = a[i].to(self.clip_vision_encoder.model.device)
            b[i] = b[i].to(next(self.deepfake_encoder.parameters()).device)
            spatial_dim = a[i].shape[-2] * a[i].shape[-1]
            A_flattened = a[i].view(-1, a[i].shape[1], spatial_dim).permute(0, 2, 1)  # Shape: [B, spatial_dim, channels]
            
            A_transformed = self.linear_lst[i](A_flattened)  # Shape: [B, spatial_dim, 1024]

            # Concatenate A_transformed with B[i] along the sequence dimension
            combined_tensor = torch.cat((A_transformed, b[i]), dim=1)  # Shape: [B, combined_seq_len, 1024]

            # Prepare for transformer input (seq_len, batch_size, embed_dim)
            combined_tensor = combined_tensor.permute(1, 0, 2)
            
            # Pass the combined tensor through the corresponding Transformer encoder block
            self.transformer_blocks[i] = self.transformer_blocks[i].to(a[i].device)
            output = self.transformer_blocks[i](combined_tensor)
            outputs.append(output)

        # for i, output in enumerate(outputs):
            # print(f"Output shape of block {i+1}:", output.shape)  # Expected: [combined_seq_len, B, 1024]

        clip_adapt_embed = self.link_adapter_proj(outputs[-1].view(-1,1024))
        clip_adapt_embed = clip_adapt_embed.view(B,-1)
        clip_adapt_embed = self.clip_text_alpha * clip_adapt_embed     
        # print("clip_adapt_embed is: ", clip_adapt_embed.size())

        deepfake_features = self.avgpool2d(deepfake_features).view(B, -1)    # torch.Size([1, 1792, 11, 11]) 
        deepfake_features = self.deepfake_proj(deepfake_features)       ## torch.Size([1, 1792]) to [B, 1024]
        clip_vision_cls = self.clip_vision_alpha * clip_vision_cls      
        
        clip_vision_cls.to(self.deepfake_dtype)    # [B, 1024]
        # clip_scores.to(self.deepfake_dtype)    # [B, 576]
        
        features = torch.cat([clip_vision_cls, clip_adapt_embed, deepfake_features], dim=-1)    
        output = self.output(features)
        
        if return_dict:
            output_dict = dict()
            output_dict['pred'] = output
            output_dict['v_a'] = self.clip_vision_alpha
            output_dict['t_a'] = self.clip_text_alpha
            return output_dict
        else:
            return output, sim_scores
    
    def assign_lr(self, module, lr, params_dict_list):
        params_dict_list.append({'params': module.parameters(), 'lr': lr})

    def assign_lr_dict_list(self, lr=1e-4):
        params_dict_list = []

        # backbone
        params_dict_list.append({'params': self.clip_text_encoder.prompt_tokens, 'lr': lr})
        params_dict_list.append({'params': self.clip_vision_alpha, 'lr': 1e-3})
        params_dict_list.append({'params': self.clip_text_alpha, 'lr': 3e-3})

        self.assign_lr(self.deepfake_encoder, lr, params_dict_list)
        self.assign_lr(self.deepfake_proj, lr, params_dict_list)
        self.assign_lr(self.text_proj, lr, params_dict_list)
        self.assign_lr(self.vision_proj, lr, params_dict_list)
        self.assign_lr(self.output, lr, params_dict_list)

        for _ in self.transformer_blocks:
            self.assign_lr(_, lr, params_dict_list)
        for _ in self.linear_lst:
            self.assign_lr(_, lr, params_dict_list)
        self.assign_lr(self.link_adapter_proj, lr, params_dict_list)

        return params_dict_list