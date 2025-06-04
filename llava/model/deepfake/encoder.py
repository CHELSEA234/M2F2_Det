import torch
from torchvision.models.densenet import densenet121
from torchvision.models import efficientnet_b0
import torch.nn as nn
import torch.nn.functional as F


class DenseNet_Deepfake(nn.Module):
    def __init__(
        self,
        drop_rate: float = 0.0,
    ):
        super(DenseNet_Deepfake, self).__init__()
        
        self.backbone = densenet121(weights='DenseNet121_Weights.DEFAULT',drop_rate=drop_rate).features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.output = nn.Sequential(nn.ReLU(inplace=True),
                                    nn.Linear(1024, 2))
        
        new_components = [self.output]
        # Official init from torch repo.
        for new_component in new_components:
            for m in new_component.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, **kwargs):
        batch_size, _, H, W = x.size()
        z = self.backbone(x)
        z = self.avgpool(z).view(batch_size, -1)
        out = self.output(z)
        return out
    
    
class EfficientNet_Deepfake(nn.Module):
    def __init__(
        self,
    ):
        super(EfficientNet_Deepfake, self).__init__()
        
        self.backbone = efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT').features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.output = nn.Sequential(nn.ReLU(inplace=True),
                                    nn.Linear(1280, 2))
        
        new_components = [self.output]
        # Official init from torch repo.
        for new_component in new_components:
            for m in new_component.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, **kwargs):
        batch_size, _, H, W = x.size()
        z = self.backbone(x)
        z = self.avgpool(z).view(batch_size, -1)
        out = self.output(z)
        return out
    
    
class CLIP_DenseNet_Deepfake(nn.Module):
    def __init__(
        self,
        load_vision_tower = False,
        vision_tower_path = None
    ):
        super(CLIP_DenseNet_Deepfake, self).__init__()
        self.select_layer = -2
        if load_vision_tower:
            self.vision_tower_path = vision_tower_path
            self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_path, device_map='cuda')
        self.deepfake_encoder = densenet121().features
        self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.output = nn.Linear(2048, 2)
        self.clip_feature_type = 'cls'        
        # new_components = [self.output]
        # # Official init from torch repo.
        # for new_component in new_components:
        #     for m in new_component.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.Linear):
        #             nn.init.constant_(m.weight, 0)
        #             nn.init.constant_(m.bias, 0)

    def forward(self, deepfake_inputs):
        x = deepfake_inputs["image"]
        batch_size, _, H, W = x.size()
        x = x.to(self.output.weight.dtype)
        y = self.deepfake_encoder(x)
        print(f'Deepfake out: {torch.mean(y)}')
        y = self.avgpool2d(y).view(batch_size, -1)
        if self.clip_feature_type == 'cls':
            if 'image_cls' in deepfake_inputs:
                z = deepfake_inputs['image_cls']
            else:
                z = self.vision_tower(x, output_hidden_states=True)
                z = z.hidden_states[self.select_layer]
                z = z[:, 0, :]
        elif self.clip_feature_type == 'patch':
            if "image_features" in deepfake_inputs:
                z = deepfake_inputs['image_features']
            else:
                z = self.vision_tower(x, output_hidden_states=True)
                z = z.hidden_states[self.select_layer]
                z = z[:, 1:, :].permute(0, 2, 1)
                z = self.avgpool1d(z).view(batch_size, -1)
        else:
            raise ValueError(f'Unsupported clip feature type for deepfake encoder: {self.clip_feature_type}')
        print(f'CLIP dense: {torch.mean(y)}')
        print(f'CLIP clip: {torch.mean(z)}')
        out = torch.cat([y, z], dim=1)
        out = out.to(self.output.weight.dtype)
        out = self.output(out)
        print(f'CLIP out: {out}')
        return out