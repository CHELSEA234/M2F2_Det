import numpy as np
import torch

from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel
from torch import nn
from torch.nn import functional as F
from typing import Optional
from flash_attn.modules.mha import MHA

try:    ## ugly code for debugging and running~
    from .text_encoder import CLIPTextEncoder
    from .vision_encoder import CLIPVisionEncoder
except:
    from text_encoder import CLIPTextEncoder
    from vision_encoder import CLIPVisionEncoder

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

# Custom View module
class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

# Custom Permute module
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, causal=False):
        super().__init__()
        self.attn = MHA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal
        ).to(torch.bfloat16)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        orig_dtype = x.dtype
        attn_out = self.attn(x.to(torch.bfloat16)).to(orig_dtype)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

# BridgeAdapter module
class BridgeAdapter_Proj(nn.Module):
    def __init__(self, hidden_size, bridge_adapter_dim):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.bridge_adapter_proj = nn.Sequential(
            View(-1, self.hidden_size),
            nn.Linear(self.hidden_size, 1),
            nn.LayerNorm(1)
        )

        self.bridge_adapter_reduction = nn.Sequential(
            nn.Linear(2731, bridge_adapter_dim),
            nn.LayerNorm(bridge_adapter_dim)
        )

    def forward(self, x, batch_size):
        x = self.bridge_adapter_proj(x)
        x = x.view(batch_size, -1)
        x = self.bridge_adapter_reduction(x)
        return x

class M2F2Det(nn.Module):
    def __init__(
        self,
        clip_text_encoder_name: str = "openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
        deepfake_encoder_name: str = 'densenet121',
        hidden_size: int = 1024,
        clip_hidden_size: int = 1024,
        vision_dtype: torch.dtype = torch.float32,
        text_dtype: torch.dtype = torch.float32,
        deepfake_dtype: torch.dtype = torch.float32,
        load_vision_encoder: bool = True,
        bridge_adapter_dim: int = 128
    ):
        super(M2F2Det, self).__init__()
        self.clip_text_encoder = CLIPTextEncoder(clip_text_encoder_name, dtype=text_dtype)
        if load_vision_encoder:
            self.clip_vision_encoder = CLIPVisionEncoder(clip_vision_encoder_name, dtype=vision_dtype)
        else:
            self.clip_vision_encoder = None
        
        self.hidden_size = hidden_size
        self.clip_hidden_size = clip_hidden_size
        self.bridge_adapter_dim = bridge_adapter_dim
        self.vision_dtype = vision_dtype
        self.text_dtype = text_dtype
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_vision_encoder_name)
        self.deepfake_encoder, self.deepfake_proj = build_deepfake_backbone(    
            model_name=deepfake_encoder_name,
            feature_dim=get_feature_dim(deepfake_encoder_name),
            hidden_size=hidden_size
        )

        # Prepare storage for the outputs
        self.block_outputs = {}
        def get_hook(name):
            def hook(model, input, output):
                self.block_outputs[name] = output
            return hook
        self.deepfake_encoder[4].register_forward_hook(get_hook("b_1"))
        self.deepfake_encoder[5].register_forward_hook(get_hook("b_2"))
        self.deepfake_encoder[6].register_forward_hook(get_hook("b_3"))
        self.deepfake_dtype = deepfake_dtype

        self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip_text_encoder.model.config.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        if self.clip_vision_encoder is not None:
            self.vision_proj = nn.Sequential(
                nn.Linear(self.clip_vision_encoder.model.config.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
        else:
            # default openai/clip-vit-large-patch14-336
            self.vision_proj = nn.Sequential(
                nn.Linear(self.clip_hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
        self.clip_vision_alpha = nn.Parameter(torch.tensor(0.5))
        self.clip_text_alpha = nn.Parameter(torch.tensor(4.0))

        self.bridge_adapter = [TransformerEncoderBlock(self.clip_hidden_size//16, 4, 32) for _ in range(3)]
        self.linear_1 = nn.Linear(112, self.clip_hidden_size//16)
        self.linear_2 = nn.Linear(160, self.clip_hidden_size//16)
        self.linear_3 = nn.Linear(272, self.clip_hidden_size//16)
        self.clip_reduction = nn.Linear(self.clip_hidden_size, self.clip_hidden_size//16)
        self.linear_lst = [self.linear_1, self.linear_2, self.linear_3]
        self.bridge_adapter_proj = BridgeAdapter_Proj(self.clip_hidden_size//16, self.bridge_adapter_dim)
        self.output = nn.Linear(2 * self.hidden_size + bridge_adapter_dim, 2)
        
        self.deepfake_encoder.to(deepfake_dtype)
        self.deepfake_proj.to(deepfake_dtype)
        self.text_proj.to(text_dtype)
        self.vision_proj.to(vision_dtype)
        self.clip_vision_alpha.to(vision_dtype)
        self.clip_text_alpha.to(text_dtype)
        self.output.to(deepfake_dtype)
        new_components = [self.text_proj, self.vision_proj, self.output, self.bridge_adapter_proj]
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
        clip_vision_cls = self.clip_vision_alpha * clip_vision_cls.to(self.deepfake_dtype)      

        device = getattr(self.deepfake_encoder, "device", None)
        if device is None:
            try:
                device = next(p.device for p in self.deepfake_encoder.parameters() if p is not None)
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # fallback
        deepfake_features = self.deepfake_encoder(new_embeds.to(device=device, dtype=self.deepfake_dtype))
        # deepfake_features = self.deepfake_encoder(new_embeds.to(self.deepfake_dtype).to(next(self.deepfake_encoder.parameters()).device))  
        # deepfake_features = self.deepfake_encoder(new_embeds.to(self.deepfake_dtype))
        deepfake_feat_0, deepfake_feat_1, deepfake_feat_2 = self.block_outputs["b_1"], self.block_outputs["b_2"], self.block_outputs["b_3"]
        
        deepfake_feat_lst = [deepfake_feat_0, deepfake_feat_1, deepfake_feat_2]
        clip_feat_lst = [clip_0, clip_1, clip_2]
        bridge_adapter_output = None
        for i, (deepfake_feat, clip_feat) in enumerate(zip(deepfake_feat_lst, clip_feat_lst)):
            deepfake_feat = deepfake_feat.to(self.clip_vision_encoder.model.device)
            # clip_feat = clip_feat.to(next(self.deepfake_encoder.parameters()).device)
            clip_feat = clip_feat.to(device)
            clip_feat = self.clip_reduction(clip_feat)
            spatial_dim = deepfake_feat.shape[-2] * deepfake_feat.shape[-1]
            self.linear_lst[i] = self.linear_lst[i].to(device)
            deepfake_feat = self.linear_lst[i](deepfake_feat.view(-1, deepfake_feat.shape[1], spatial_dim).permute(0, 2, 1).to(device))  # Shape: [B, spatial_dim, 1024]

            if bridge_adapter_output is None:
                combined_tensor = torch.cat((deepfake_feat, clip_feat), dim=1) 
            else:
                bridge_adapter_output = bridge_adapter_output.permute(1, 0, 2)
                combined_tensor = torch.cat((bridge_adapter_output, deepfake_feat, clip_feat), dim=1)

            combined_tensor = combined_tensor.permute(1, 0, 2)  # Prepare for transformer input (seq_len, batch_size, embed_dim)
            
            self.bridge_adapter[i] = self.bridge_adapter[i].to(deepfake_feat.device)
            bridge_adapter_output = self.bridge_adapter[i](combined_tensor)

        clip_adapt_embed = self.bridge_adapter_proj(bridge_adapter_output, B)
        clip_adapt_embed = self.clip_text_alpha * clip_adapt_embed.view(B,-1)

        deepfake_features = self.avgpool2d(deepfake_features).view(B, -1)    # torch.Size([1, 1792, 11, 11]) 
        deepfake_features = self.deepfake_proj(deepfake_features)       ## torch.Size([1, 1792]) to [B, 1024]
           
        
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
        # params_dict_list.append({'params': self.clip_text_encoder.prompt_tokens, 'lr': lr})
        params_dict_list.append({'params': self.clip_vision_alpha, 'lr': 1e-3})
        params_dict_list.append({'params': self.clip_text_alpha, 'lr': 3e-3})

        self.assign_lr(self.deepfake_encoder, lr, params_dict_list)
        self.assign_lr(self.deepfake_proj, lr, params_dict_list)
        self.assign_lr(self.text_proj, lr, params_dict_list)
        self.assign_lr(self.vision_proj, lr, params_dict_list)
        self.assign_lr(self.output, lr, params_dict_list)
        self.assign_lr(self.bridge_adapter_proj, lr, params_dict_list)
        self.assign_lr(self.clip_reduction, lr, params_dict_list)

        for _ in self.bridge_adapter:
            self.assign_lr(_, lr, params_dict_list)
        for _ in self.linear_lst:
            self.assign_lr(_, lr, params_dict_list)

        return params_dict_list

if __name__ == "__main__":    ## Model definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M2F2Det(
        clip_text_encoder_name="openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
        deepfake_encoder_name='efficientnet_b4',
        hidden_size=1792,
    )
    # Move the model to device
    model = model.to(device)
    model = torch.nn.DataParallel(model).to(device)     

    input_randn = torch.randn(72, 3, 224, 224).to(device) 
    output_dict = model(input_randn, return_dict=True)
    print(output_dict['pred'].shape)