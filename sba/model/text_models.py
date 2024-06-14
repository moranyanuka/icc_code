import torch
import torch.nn as nn
from transformers import CLIPModel, BertModel
from transformers import AutoModel


class CustomTextClip(nn.Module):
    def __init__(self, mm_text_tower, torch_dtype=None, low_cpu_mem_usage=False):
        super(CustomTextClip, self).__init__()
        self.model = CLIPModel.from_pretrained(mm_text_tower, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        self.config = self.model.text_model.config
        self.config._name_or_path = self.model.config._name_or_path
        self.device = self.model.device
        self.config.hidden_size = self.model.text_model.config.projection_dim

    def forward(self, encoder_text_ids):
        return self.model.get_text_features(**encoder_text_ids)


class CustomMpnetBaseV2(nn.Module):
    def __init__(self, mm_text_tower, torch_dtype=None, low_cpu_mem_usage=False):
        super(CustomMpnetBaseV2, self).__init__()
        self.model = AutoModel.from_pretrained(mm_text_tower, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        self.config = self.model.config
        self.device = self.model.device
    
    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(torch.float16)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, encoder_text_ids):
        model_outputs = self.model(**encoder_text_ids)
        text_features = self._mean_pooling(model_outputs, encoder_text_ids['attention_mask'])
        return text_features


class CustomTextBlip(nn.Module):
    def __init__(self, mm_text_tower, torch_dtype=None, low_cpu_mem_usage=False):
        super(CustomTextBlip, self).__init__()
        self.model = AutoModel.from_pretrained(mm_text_tower, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
        self.config =  self.model.text_model.config
        self.config.hidden_size = self.model.config.projection_dim
        self.config._name_or_path = self.model.config._name_or_path
        self.device = self.model.device

    def forward(self, encoder_text_ids):
        return self.model.get_text_features(**encoder_text_ids)


class CustomFlava(nn.Module):
    def __init__(self, mm_text_tower, torch_dtype=None, low_cpu_mem_usage=False):
        super(CustomFlava, self).__init__()
        self.model = AutoModel.from_pretrained(mm_text_tower, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        self.config =  self.model.text_model.config
        self.config._name_or_path = self.model.config._name_or_path
        self.device = self.model.device

    def forward(self, encoder_text_ids):
        text_outputs = self.model.text_model(**encoder_text_ids)
        pooled_output = text_outputs[1]
        text_features = self.model.text_projection(pooled_output)
        return text_features


class CustomBert(nn.Module):
    def __init__(self, mm_text_tower, torch_dtype=None, low_cpu_mem_usage=False):
        super(CustomBert, self).__init__()
        self.model = BertModel.from_pretrained(mm_text_tower, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        self.config =  self.model.config
        self.device = self.model.device

    def forward(self, encoder_text_ids):
        text_outputs = self.model(**encoder_text_ids)
        text_features = text_outputs.pooler_output
        return text_features