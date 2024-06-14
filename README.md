# ICC: Quantifying Image Caption Concreteness  <br />  for Multimodal Dataset Curation

**ACL 2024 (Findings)** 

<!-- [Project Page](https://moranyanuka.github.io/icc/) &nbsp; &nbsp; [Paper](https://arxiv.org/abs/2403.01306) -->

<a href="https://moranyanuka.github.io/icc/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>

<a href="https://arxiv.org/abs/2403.01306"><img src="https://img.shields.io/badge/arXiv-2311.13608-b31b1b.svg"></a>

# Release
We will soon release the code needed to reproduce our paper. 

we release the *ICC* model on huggingFace [here](https://huggingface.co/moranyanuka/icc).


# Running the ICC model with HuggingFace 🤗
*ICC* model can be run with a few lines of code using HuggingFace:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("moranyanuka/icc")
model = AutoModelForSequenceClassification.from_pretrained("moranyanuka/icc").to("cuda")

captions = ["a great method of quantifying concreteness", "a man with a white shirt"]
text_ids = tokenizer(captions, padding=True, return_tensors="pt", truncation=True).to("cuda")
with torch.inference_mode():
  icc_scores = model(**text_ids)["logits"]

# tensor([[0.0339], [1.0068]])
```

If you just want to filter your multimodal dataset, you can stop here. Otherwise, follow the below instructions.

# Setup
## Clone Project
```
git clone https://github.com/moranyanuka/icc_code.git
cd icc_code
```

## Create the Environment
To set up our environment, please run:
```
conda env create -f environment.yml
conda activate icc
```
<br>

# SBA

The SBA code is highly based on [LLaVA codebase](https://github.com/haotian-liu/LLaVA), we thank the authors for their great repo!
 
## SBA Training

Run the following command:

```Shell
torchrun --nnodes 1 --nproc_per_node 1 --master_port 25000 sba/train/train_mem.py 
         --model_name_or_path meta-llama/Llama-2-7b-hf 
         --cache_dir <path-to-hf-cache_dir (defaults to None)>
         --train_data_path data/cc3m_concept_balanced_train.csv
         --text_tower openai/clip-vit-large-patch14 
         --evaluation_strategy no 
         --tune_mm_mlp_adapter True 
         --mm_text_select_layer -2 
         --bf16 True 
         --output_dir <path-to-model-output-dir>
         --num_train_epochs 2
         --per_device_train_batch_size 32 
         --per_device_eval_batch_size 64 
         --gradient_accumulation_steps 4 
         --evaluation_strategy no 
         --save_strategy steps 
         --save_steps 500 
         --save_total_limit 1 
         --learning_rate 2e-3 
         --weight_decay 0. 
         --warmup_ratio 0.03 
         --lr_scheduler_type cosine 
         --logging_steps 1 
         --tf32 True 
         --model_max_length 2048 
         --gradient_checkpointing True 
         --lazy_preprocess True 
         --report_to wandb
```

In case you don't have sufficient GPU memory, try decreasing the batch size, and increasing the gradient_accumulation_steps.

## SBA inference

To generate the reconstructed caption through the SBA, run:
```Shell
python sba/eval/sba_inference.py
       --model_path <path-to-trained-sba-model>
       --cache_dir <path-to-hf-cache_dir (default is None)>
       --num_beams <num-beams>
       --batch_size <batch-size>
```

This will generate a new file, with the reconstructed SBA captions, and the corresponding edit-distance between the reconstructions and original captions.

# VBA

We provide an example script for generating the reconstructed captions throught the VBA with BLIP2 captioner and stable diffusion 2 as the text-to-image model.

Simply run:

```bash
python vba/vba_inference.py
       --batch_size <batch-size>
       --cache_dir <path-to-hf-cache_dir (default is None)>
       --num_beams <num-beams>
       --output-path data/cc3m_concept_balanced_test_with_vba_predictions.csv
```

This will generate a file in output_path with the VBA reconstructions.