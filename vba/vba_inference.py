import pandas as pd
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse
import os
from tqdm.auto import tqdm
from glob import glob

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration

torch.manual_seed(0)

"""generate stable diffusion images for CC captions"""
SD_KWARGS = {
    'guidance_scale': 9,
    'num_inference_steps': 20
    }


def generate_dataset(args):
    device = args.device

    # load diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    # load captioning model
    processor = Blip2Processor.from_pretrained(args.captioning_model_id, cache_dir=args.cache_dir)
    captioning_model = Blip2ForConditionalGeneration.from_pretrained(
        args.captioning_model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir, device_map='auto'
    )

    df = pd.read_csv(args.data_dir)
    df = df.set_index(df.index // args.batch_size).copy()
    out_captions = []
    prompts = ['a photo of ' for _ in range(args.batch_size)]
    for i in tqdm(range(int(df.index.max()+1))):
        captions = df.loc[i].caption.tolist()
        #captions = [str(caption) for caption in captions]
        with torch.no_grad():
            out = pipe(
                captions,
                num_images_per_prompt=1,
                **SD_KWARGS
            )
        inputs = processor(text=prompts, images=out.images, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            generated_ids = captioning_model.generate(**inputs, num_beams=args.num_beams)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        out_captions += generated_texts
    df['output_caption'] = out_captions
    df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captioning-model-id", type=str, default="Salesforce/blip2-opt-2.7b-coco")
    parser.add_argument("--sd-model-id", type=str, default="stabilityai/stable-diffusion-2")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/cc3m_concept_balanced_test.csv")
    parser.add_argument("--output_path", type=str, default="data/cc3m_concept_balanced_test_with_vba_predictions.csv")
    args = parser.parse_args()

    generate_dataset(args)