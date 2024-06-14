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
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor, CLIPModel
from  transformers import CLIPTextModel, AutoTokenizer

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class sbaConfig(LlamaConfig):
    model_type = "sba"


class sbaLlamaModel(LlamaModel):
    config_class = sbaConfig

    def __init__(self, config: LlamaConfig, mm_text_tower=None, mm_hidden_size=None):
        super(sbaLlamaModel, self).__init__(config)

        if hasattr(config, "mm_text_tower"):
            # HACK: for FSDP
            #self.text_tower = nn.ModuleList([CLIPTextModel.from_pretrained(config.mm_text_tower)])
            clip_model = CLIPModel.from_pretrained(config.mm_text_tower)
            text_tower = CLIPTextModel.from_pretrained(config.mm_text_tower)
            self.text_projection = clip_model.text_projection
            self.text_tower = [text_tower]
        if hasattr(config, "use_mm_proj"):
            self.mm_proj = nn.Linear(config.mm_hidden_size, config.hidden_size)
            #self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_text_modules(self, text_tower, mm_text_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_text_tower = text_tower

        if not hasattr(self, 'text_tower'):
            clip_model = CLIPModel.from_pretrained(text_tower)
            text_tower = CLIPTextModel.from_pretrained(text_tower)
            text_projection = clip_model.text_projection
            text_tower = text_tower
        else:
            text_tower = self.text_tower[0]
            text_projection = self.text_projection
        text_tower.requires_grad_(False)
        text_projection.requires_grad_(False)
        text_tower = text_tower.to(torch.float16)
        self.text_projection = text_projection.to(torch.float16)
        self.text_tower = [text_tower]

        text_config = text_tower.config
        #num_patches = (text_config.image_size // text_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = text_config.hidden_size
        self.config.mm_text_select_layer = mm_text_select_layer

        if not hasattr(self, 'mm_proj'):
            self.mm_proj = nn.Linear(text_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_proj.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            text_config=text_config
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        encoder_text_ids: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for sba pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        text_tower = getattr(self, 'text_tower', None)
        if text_tower is not None and (input_ids.shape[1] != 1 or self.training) and encoder_text_ids is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            text_tower = text_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                text_outputs = text_tower(**encoder_text_ids)
                pooled_output = text_outputs[1]
                text_features = self.text_projection(pooled_output)
            if type(encoder_text_ids) is list:
                text_features = [self.mm_proj(text_feature)[0] for text_feature in text_features]
            else:
                text_features = self.mm_proj(text_features)
            #dummy_text_features = torch.zeros(1, 768, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_text_features = torch.zeros(text_features.shape[1], 768, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_text_features = self.mm_proj(dummy_text_features)
            text_features = text_features.unsqueeze(1)
            new_input_embeds = []
            cur_text_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == text_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_text_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_text_idx += 1
                    continue
                if text_tower.config.use_im_start_end:
                    cur_text_features = text_features[cur_text_idx]
                    #num_patches = 1
                    num_patches = cur_text_features.shape[0]
                    if (cur_input_ids == text_tower.config.im_start_token).sum() != (cur_input_ids == text_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    text_start_tokens = torch.where(cur_input_ids == text_tower.config.im_start_token)[0]
                    for text_start_token_pos in text_start_tokens:
                        cur_text_features = text_features[cur_text_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_text_features.shape[0]
                        if cur_input_ids[text_start_token_pos + num_patches + 1] != text_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:text_start_token_pos].detach(), cur_input_embeds[text_start_token_pos:text_start_token_pos+1], cur_text_features, cur_input_embeds[text_start_token_pos + num_patches + 1:text_start_token_pos + num_patches + 2], cur_input_embeds[text_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:text_start_token_pos+1], cur_text_features, cur_input_embeds[text_start_token_pos + num_patches + 1:]), dim=0)
                        cur_text_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_text_features = text_features[cur_text_idx]
                    num_patches = cur_text_features.shape[0]
                    if (cur_input_ids == text_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == text_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_text_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_text_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_text_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(sbaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class sbaLlamaForCausalLM(LlamaForCausalLM):
    config_class = sbaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = sbaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        encoder_text_ids: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_text_ids=encoder_text_ids
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "encoder_text_ids": kwargs.get("encoder_text_ids", None),
            }
        )
        return model_inputs

    def initialize_text_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        text_config = self.get_model().text_tower[0].config
        text_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            text_config.im_start_token, text_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        text_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

AutoConfig.register("sba", sbaConfig)
AutoModelForCausalLM.register(sbaConfig, sbaLlamaForCausalLM)
