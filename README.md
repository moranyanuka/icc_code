# icc_code
Official Code for the paper ICC: Quantifying Image Caption Concreteness for Multimodal Dataset Curation

# Release
We will soon release the code needed to replicate our [paper](https://arxiv.org/abs/2403.01306). 

For now, we release the ICC model.

# Running the ICC model with Huggingface 🤗

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("moranyanuka/icc")
model = AutoModelForSequenceClassification.from_pretrained("moranyanuka/icc).to("cuda")

captions = ["a great method of quantifying concreteness", "a man with a white shirt"]
text_ids = tokenizer(captions, padding=True, return_tensors="pt", truncation=True).to('cuda')
with torch.inference_mode():
  icc_scores = model(**text_ids)['logits']

# tensor([[0.0339], [1.0068]])
```
