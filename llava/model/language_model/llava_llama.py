#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import transforms

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, AutoProcessor

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..multimodal_projector.builder import build_multimodal_projector
from ..deepfake.encoder import DenseNet_Deepfake, EfficientNet_Deepfake, CLIP_DenseNet_Deepfake
from ..deepfake.M2F2Det.model import M2F2Det

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEEPFAKE_TOKEN_INDEX
from llava.mm_utils import get_anyres_image_grid_shape, process_images

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


def build_deepfake_encoder(
    deepfake_encoder_name: str = 'densenet121',
    vision_model_name: str = 'openai/clip-vit-large-patch14-336',
    text_model_name: str = 'openai/clip-vit-large-patch14-336',
    vision_dtype: torch.dtype = torch.float16,
    text_dtype: torch.dtype = torch.float16,
    deepfake_dtype: torch.dtype = torch.float32,
    load_vision_encoder: bool = False,
    pretrained: bool = False
):
    model = M2F2Det(
        clip_text_encoder_name=text_model_name,
        clip_vision_encoder_name=vision_model_name,
        deepfake_encoder_name=deepfake_encoder_name,
        vision_dtype=vision_dtype,
        text_dtype=text_dtype,
        deepfake_dtype=deepfake_dtype,
        load_vision_encoder=load_vision_encoder,
        pretrained=pretrained
    )
    return model
    
    
class LlavaLlamaForCausalLMDummy(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlavaLlamaForCausalLMDummy, self).__init__(config)
        self.config = config
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.deepfake_token = nn.Parameter(torch.ones((1, 2)), requires_grad=True)    # (have or not have)
        self.deepfake_replicate = False
        for p in self.model.vision_tower.parameters():
            p.requires_grad = False
        self.deepfake_projector = build_multimodal_projector(config, projector_type='mlp2x_gelu', multimodal_hidden_size=2)
        # Initialize weights and apply final processing
        self.processors = {
            "clip_processor": self.get_vision_tower().image_processor,
            "deepfake_processor": transforms.Compose([
                # transforms.ToPILImage(), # Next line takes PIL images as input (ToPILImage() preserves the values in the input array or tensor)
                transforms.ToTensor(), # To bring the pixel values in the range [0,1]
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])
        }
        self.init_deepfake_branch()
        
    
    def init_deepfake_branch(self):
        pass
            
    def get_model(self):
        return self.model
    
    # encode dummy token
    # use deepfake_token as the input
    def encode_deepfake_tokens(self, image_tensor):
        B = image_tensor.shape[0]
        out = self.deepfake_token.to(self.deepfake_projector[0].weight.dtype).unsqueeze(0).repeat(B, 1, 1)

        # out = self.deepfake_encoder(image_tensor, image_tensor).unsqueeze(1)
        if self.deepfake_replicate:
            out = torch.repeat_interleave(out, 200, dim=1)
        out = self.deepfake_projector(out)
        return out
    
    
    def encode_conditional_deepfake_tokens(self, image_tensor, conditional_labels=[]):
        B = image_tensor.shape[0]
        assert(len(conditional_labels) == B)
        conditional_probs = []
        for label in conditional_labels:
            pos_prob = 0.5 * torch.rand((1, 1)).item() + 0.5 * label
            conditional_probs.append(torch.tensor([1 - pos_prob, pos_prob]))
        conditional_probs = torch.stack(conditional_probs, dim=0)
        
        print(f'Prob: {conditional_probs}')
        out = conditional_probs.to(self.deepfake_projector[0].weight.dtype).unsqueeze(1).to(self.deepfake_projector[0].weight.device)    # [B, 1, 2]

        # out = self.deepfake_encoder(image_tensor, image_tensor).unsqueeze(1)
        if self.deepfake_replicate:
            out = torch.repeat_interleave(out, 200, dim=1)
        out = self.deepfake_projector(out)
        return out
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[list] = None,
        image_sizes: Optional[List[List[int]]] = None,
        deepfake_inputs: Optional[list] = None,
        conditional_labels: Optional[list] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            image_tensor = None
            deepfake_processed_inputs = None
        if images is not None or deepfake_inputs is not None:
            image_tensor = None
            deepfake_processed_inputs = None
            if images is not None:
                image_tensor = process_images(images, self.processors['clip_processor'], self.config)
                if isinstance(image_tensor, list):
                    image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            if deepfake_inputs is not None:
                if isinstance(deepfake_inputs, list):
                    deepfake_processed_inputs = []
                    for deepfake_input_video in deepfake_inputs:
                        deepfake_processed_frames = []
                        for deepfake_input_frame in deepfake_input_video:
                            deepfake_processed_frames.append(self.processors['deepfake_processor'](deepfake_input_frame))
                        deepfake_processed_inputs.append(torch.stack(deepfake_processed_frames, dim=0))
                    deepfake_processed_inputs = torch.stack(deepfake_processed_inputs, dim=0).to(self.device)
                else:
                    raise notImplementedError
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_deepfake_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                image_tensor,
                image_sizes,
                deepfake_processed_inputs,
                conditional_labels
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[list] = None,
        image_sizes: Optional[torch.Tensor] = None,
        deepfake_inputs: Optional[list] = None,
        conditional_labels: Optional[list] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None or deepfake_inputs is not None:
            image_tensor = None
            if images is not None:
                image_tensor = process_images(images, self.processors['clip_processor'], self.config)
                if isinstance(image_tensor, list):
                    image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            if deepfake_inputs is not None:
                if isinstance(deepfake_inputs, list):
                    deepfake_processed_inputs = []
                    for deepfake_input_video in deepfake_inputs:
                        deepfake_processed_frames = []
                        for deepfake_input_frame in deepfake_input_video:
                            deepfake_processed_frames.append(self.processors['deepfake_processor'](deepfake_input_frame))
                        deepfake_processed_inputs.append(torch.stack(deepfake_processed_frames, dim=0))
                    deepfake_processed_inputs = torch.stack(deepfake_processed_inputs, dim=0).to(self.device)
                else:
                    raise notImplementedError
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_deepfake_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images=image_tensor,
                image_sizes=image_sizes,
                deepfake_inputs=deepfake_processed_inputs,
                conditional_labels=conditional_labels
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def prepare_inputs_labels_for_deepfake_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images=None, image_sizes=None, deepfake_inputs=None, conditional_labels=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or input_ids.shape[1] == 1 or (images is None and deepfake_inputs is None):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if images is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                if mm_patch_merge_type == 'flat':
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                vision_input = images.to(self.device, dtype=self.get_model().mm_projector[0].weight.dtype)
                image_features = self.encode_images(vision_input)
        
        if deepfake_inputs is not None:
            deepfake_outs = self.encode_conditional_deepfake_tokens(deepfake_inputs, conditional_labels)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_deepfake_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_deepfake_embeds = (cur_input_ids == DEEPFAKE_TOKEN_INDEX).sum()
            if num_images == 0 and num_deepfake_embeds == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            image_token_indice = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            deepfake_token_indice = torch.where(cur_input_ids == DEEPFAKE_TOKEN_INDEX)[0].tolist()
            special_token_elems = [(pos, index) for indice, token_index in zip([image_token_indice, deepfake_token_indice], [IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX]) \
                for pos, index in zip(indice, [token_index] * len(indice))]
            special_token_elems = sorted(special_token_elems, key=lambda x: x[0])
            special_token_pos_indices, special_token_index = zip(*special_token_elems)
            
            special_token_pos_indices = [-1] + list(special_token_pos_indices) + [cur_input_ids.shape[0]]
            cur_input_ids_no_sp = []
            cur_labels = labels[batch_idx]
            cur_labels_no_sp = []
            for i in range(len(special_token_pos_indices) - 1):
                cur_input_ids_no_sp.append(cur_input_ids[special_token_pos_indices[i]+1:special_token_pos_indices[i+1]])
                cur_labels_no_sp.append(cur_labels[special_token_pos_indices[i]+1:special_token_pos_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_no_sp]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_no_sp))
            cur_input_embeds_no_sp = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(len(special_token_index) + 1):    # insert all special tokens
                cur_new_input_embeds.append(cur_input_embeds_no_sp[i])
                cur_new_labels.append(cur_labels_no_sp[i])
                if i < len(special_token_index):
                    cur_token_index = special_token_index[i]
                    if cur_token_index == IMAGE_TOKEN_INDEX:
                        cur_token_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                    elif cur_token_index == DEEPFAKE_TOKEN_INDEX:
                        cur_token_features = deepfake_outs[cur_deepfake_idx]
                        cur_deepfake_idx += 1
                    else:
                        raise NotImplementedError('Unsupported Token')
                    cur_new_input_embeds.append(cur_token_features)
                    # print(f'Multimodal embed: {cur_new_input_embeds[-1].shape}')

                    cur_new_labels.append(torch.full((cur_token_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    
class LlavaLlamaForCausalLMDeepfake(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlavaLlamaForCausalLMDeepfake, self).__init__(config)
        self.config = config
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.deepfake_replicate = False
        self.deepfake_encoder = build_deepfake_encoder(
            # deepfake_encoder_name=config.deepfake_model_name,
            vision_model_name=config.mm_vision_tower,
            text_model_name=config.mm_vision_tower,
            vision_dtype=torch.bfloat16,
            text_dtype=torch.bfloat16,
            deepfake_dtype=torch.bfloat16,
            load_vision_encoder=False,
            pretrained=False
        )
        for p in self.deepfake_encoder.parameters():
            p.requires_grad = False
        for p in self.model.vision_tower.parameters():
            p.requires_grad = False
        self.deepfake_projector = build_multimodal_projector(config, projector_type='mlp2x_gelu', multimodal_hidden_size=2)
        # Initialize weights and apply final processing
        self.processors = {
            "clip_processor": self.get_vision_tower().image_processor,
            "deepfake_processor": self.get_vision_tower().image_processor,
        }
        if self.config.mm_vision_select_feature != "cls_patch":
            raise ValueError("Only cls_patch is supported for mm_vision_select_feature.")
        
    def load_deepfake_encoder(self, model_path, verbose=True):
        ckpt = torch.load(model_path)
        state_dict = dict()
        for k, v in self.deepfake_encoder.state_dict().items():
            if k in ckpt:
                state_dict[k] = ckpt[k].to(v.dtype)
        missing_keys, unexpected_keys = self.deepfake_encoder.load_state_dict(state_dict, strict=False)
        if verbose:
            print('Load deepfake encoder')
            print(f'Missing keys: {missing_keys}')
            print(f'Unexpected keys: {unexpected_keys}')
            
        
        
    def get_model(self):
        return self.model

    def encode_deepfake_tokens(self, deepfake_inputs):
        out = self.deepfake_encoder(**deepfake_inputs).unsqueeze(1)    # [B, 1, 2]
        out = F.softmax(out, dim=-1)
        
        out = out.to(self.deepfake_projector[0].weight.dtype)
        if self.deepfake_replicate:
            out = torch.repeat_interleave(out, 200, dim=1)
        # print(f'Prob: {out}')
        out = self.deepfake_projector(out)
        return out
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[list] = None,
        image_sizes: Optional[List[List[int]]] = None,
        deepfake_inputs: Optional[list] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            image_tensor = None
            deepfake_processed_inputs = None
        if images is not None or deepfake_inputs is not None:
            image_tensor = None
            deepfake_processed_inputs = None
            if images is not None:
                image_tensor = process_images(images, self.processors['clip_processor'], self.config)
                if isinstance(image_tensor, list):
                    image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            if deepfake_inputs is not None:
                if isinstance(deepfake_inputs, list):
                    deepfake_processed_inputs = []
                    for deepfake_input in deepfake_inputs:
                        deepfake_processed_inputs.append(self.processors['deepfake_processor'].preprocess(deepfake_input, return_tensors='pt')['pixel_values'][0])
                    deepfake_processed_inputs = torch.stack(deepfake_processed_inputs, dim=0).to(self.device)
                else:
                    raise notImplementedError
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_deepfake_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                image_tensor,
                image_sizes,
                deepfake_processed_inputs
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[list] = None,
        image_sizes: Optional[torch.Tensor] = None,
        deepfake_inputs: Optional[list] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None or deepfake_inputs is not None:
            image_tensor = None
            if images is not None:
                image_tensor = process_images(images, self.processors['clip_processor'], self.config)
                if isinstance(image_tensor, list):
                    image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            if deepfake_inputs is not None:
                if isinstance(deepfake_inputs, list):
                    deepfake_processed_inputs = []
                    for deepfake_input in deepfake_inputs:
                        deepfake_processed_inputs.append(self.processors['deepfake_processor'].preprocess(deepfake_input, return_tensors='pt')['pixel_values'][0])
                    deepfake_processed_inputs = torch.stack(deepfake_processed_inputs, dim=0).to(self.device)
                else:
                    raise notImplementedError
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_deepfake_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images=image_tensor,
                image_sizes=image_sizes,
                deepfake_inputs=deepfake_processed_inputs
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def prepare_inputs_labels_for_deepfake_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images=None, image_sizes=None, deepfake_inputs=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or input_ids.shape[1] == 1 or (images is None and deepfake_inputs is None):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if images is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                v_image_features, image_features = self.encode_images_multimodal_features(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                if mm_patch_merge_type == 'flat':
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                vision_input = images.to(self.device, dtype=self.get_model().mm_projector[0].weight.dtype)
                v_image_features, image_features = self.encode_images_multimodal_features(vision_input)
        image_cls = None
        if self.config.mm_vision_select_feature == 'cls_patch':
            v_image_cls = v_image_features[:, 0, :]
            v_image_features = v_image_features[:, 1:, :]
            image_features = image_features[:, 1:, :]
        else:
            raise NotImplementedError('Unsupported mm_vision_select_feature:', self.config.mm_vision_select_feature)
        if deepfake_inputs is not None:
            deepfake_outs = self.encode_deepfake_tokens(
                {
                    "images": deepfake_inputs,
                    "clip_vision_features": torch.cat([v_image_cls.unsqueeze(1), v_image_features], dim=1),
                    "use_cached_clip_text_features": True
                }
            )
            
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_deepfake_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_deepfake_embeds = (cur_input_ids == DEEPFAKE_TOKEN_INDEX).sum()
            if num_images == 0 and num_deepfake_embeds == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            image_token_indice = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            deepfake_token_indice = torch.where(cur_input_ids == DEEPFAKE_TOKEN_INDEX)[0].tolist()
            special_token_elems = [(pos, index) for indice, token_index in zip([image_token_indice, deepfake_token_indice], [IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX]) \
                for pos, index in zip(indice, [token_index] * len(indice))]
            special_token_elems = sorted(special_token_elems, key=lambda x: x[0])
            special_token_pos_indices, special_token_index = zip(*special_token_elems)
            
            special_token_pos_indices = [-1] + list(special_token_pos_indices) + [cur_input_ids.shape[0]]
            cur_input_ids_no_sp = []
            cur_labels = labels[batch_idx]
            cur_labels_no_sp = []
            for i in range(len(special_token_pos_indices) - 1):
                cur_input_ids_no_sp.append(cur_input_ids[special_token_pos_indices[i]+1:special_token_pos_indices[i+1]])
                cur_labels_no_sp.append(cur_labels[special_token_pos_indices[i]+1:special_token_pos_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_no_sp]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_no_sp))
            cur_input_embeds_no_sp = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(len(special_token_index) + 1):    # insert all special tokens
                cur_new_input_embeds.append(cur_input_embeds_no_sp[i])
                cur_new_labels.append(cur_labels_no_sp[i])
                if i < len(special_token_index):
                    cur_token_index = special_token_index[i]
                    if cur_token_index == IMAGE_TOKEN_INDEX:
                        cur_token_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                    elif cur_token_index == DEEPFAKE_TOKEN_INDEX:
                        cur_token_features = deepfake_outs[cur_deepfake_idx]
                        cur_deepfake_idx += 1
                    else:
                        raise NotImplementedError('Unsupported Token')
                    cur_new_input_embeds.append(cur_token_features)
                    # print(f'Multimodal embed: {cur_new_input_embeds[-1].shape}')

                    cur_new_labels.append(torch.full((cur_token_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
        
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLMDeepfake)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLMDummy)
