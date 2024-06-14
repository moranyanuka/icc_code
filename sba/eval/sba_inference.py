import argparse
import json
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sba.utils import disable_torch_init
from sba.model import *
from sba.model.utils import KeywordsStoppingCriteria
from sba.model.utils import build_custom_model
from nltk.metrics import edit_distance
from tqdm import tqdm
import math

import os
import pandas as pd

""" Given a pretrained llava model, and a captions dataset, create 
    an output caption and edit distance score.
"""

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class generation_dataset(Dataset):
    """Dataset for batched generation."""
    def __init__(self, df: pd.DataFrame):
        super(generation_dataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = df.copy()
        list_data_dict["output_caption"] = 0
        list_data_dict["gt_edit_distance"] = 0
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        text = self.list_data_dict.caption[i]
        return i, text


def generate_dataset(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.cache_dir)
    text_encoder_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, cache_dir=args.cache_dir)
    model = sbaLlamaForCausalLM.from_pretrained(model_path, 
                                                low_cpu_mem_usage=True, 
                                                torch_dtype=torch.float16, 
                                                cache_dir=args.cache_dir).cuda()

    mm_use_im_start_end = False

    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    text_tower = model.get_model().text_tower[0]
    if text_tower.device.type == 'meta':
        text_tower = build_custom_model(text_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().text_tower[0] = text_tower
    else:
        text_tower.to(device='cuda', dtype=torch.float16)
    text_config = text_tower.config
    text_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    text_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        text_config.im_start_token, text_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    text_token_len = 1
    
    df = pd.read_csv(args.input_path)
    input_dataset = generation_dataset(df)
    input_dataloader = DataLoader(input_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    for batch_idx, text in tqdm(input_dataloader):
        encoder_text_ids = text_encoder_tokenizer(text, padding=True, truncation=True, return_tensors="pt").to('cuda')

        if mm_use_im_start_end:
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * text_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = DEFAULT_IMAGE_PATCH_TOKEN * text_token_len

        prompt = qs 
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        input_ids = input_ids.repeat(encoder_text_ids.data['input_ids'].shape[0], 1)

        stop_str = '\n'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                encoder_text_ids=encoder_text_ids,
                no_repeat_ngram_size=2,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])


        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids[0] != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        try:
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        except:
            continue
        min_dist = math.inf
        sample_idx = 0
        for i, output in enumerate(outputs):
            output = output.split(stop_str)[0].strip()
            dist = edit_distance(text[sample_idx], output) / max(len(text[sample_idx]), len(output))
            if dist < min_dist:
                output_sentence = output
                min_dist = dist 
            if (i + 1) % (args.num_beams) == 0:
                df.loc[int(batch_idx[sample_idx]), 'output_caption'] = output_sentence
                df.loc[int(batch_idx[sample_idx]), 'edit_distance'] = min_dist
                sample_idx += 1
                min_dist = math.inf
    df.to_csv(args.output_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--text-encoder", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_path", type=str, default="data/cc3m_concept_balanced_test.csv")
    parser.add_argument("--output_path", type=str, default="data/cc3m_concept_balanced_test_with_sba_predictions.csv")
    args = parser.parse_args()

    generate_dataset(args)

