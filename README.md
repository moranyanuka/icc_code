# ICC: Quantifying Image Caption Concreteness  <br />  for Multimodal Dataset Curation

**ACL 2024 (Findings)** 

[Project Page](https://moranyanuka.github.io/icc/) &nbsp; &nbsp; [Paper](https://arxiv.org/abs/2403.01306)
# Release
We will soon release the code needed to reproduce our paper. 

For now, we release the *ICC* model on huggingFace [here](https://huggingface.co/moranyanuka/icc).

# Running the ICC model with Huggingface 🤗
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
