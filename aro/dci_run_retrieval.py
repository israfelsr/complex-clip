#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from dataset_zoo import COCO_Retrieval, Flickr30k_Retrieval, Urban1k_Retrieval
from clip_aro_wrap import AROtoHFCLIPWrap
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import pandas as pd

import argparse

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ARO_DIR = "/fsx/homes/Yova.Kementchedjhieva@mbzuai.ac.ae/.cache/prerelease_bow"
COCO_DIR = "/fsx/homes/Yova.Kementchedjhieva@mbzuai.ac.ae/.cache/coco/2014"
FLICKR_DIR = "/fsx/homes/Yova.Kementchedjhieva@mbzuai.ac.ae/.cache/flickr30k/images"


def run_aro_evals(args, model: CLIPModel, processor: CLIPProcessor):
    model = AROtoHFCLIPWrap(model, processor)
  
    if args.coco:
      root_dir = COCO_DIR
      coco_dataset = COCO_Retrieval(
        image_preprocess=processor.image_processor,
        download=True,
        root_dir=root_dir,
        split="test",
      )
      collate_fn = _default_collate if processor.image_processor is None else None
      coco_loader = DataLoader(coco_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
      coco_scores = model.get_retrieval_scores_dataset(coco_loader)
      coco_records = coco_dataset.evaluate_scores(coco_scores)

    if args.flickr:
      root_dir = FLICKR_DIR
      flickr_dataset = Flickr30k_Retrieval(
        image_preprocess=processor.image_processor,
        download=True,
        root_dir=root_dir,
        split="test",
      )
      collate_fn = _default_collate if processor.image_processor is None else None
      flickr_loader = DataLoader(flickr_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
      flickr_scores = model.get_retrieval_scores_dataset(flickr_loader)
      flickr_records = flickr_dataset.evaluate_scores(flickr_scores)

    if args.urban:
      urban_dataset = Urban1k_Retrieval(
         image_preprocess=processor.image_processor,
      )
      collate_fn = _default_collate if processor.image_processor is None else None
      urban_loader = DataLoader(urban_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
      urban_scores = model.get_retrieval_scores_dataset(urban_loader)
      urban_records = urban_dataset.evaluate_scores(urban_scores)

def run_aro_on_lora(args, processor, base_clip_model, lora_weight_path):
    from peft import PeftModel

    loaded = PeftModel.from_pretrained(base_clip_model, lora_weight_path)
    loaded = loaded.merge_and_unload()
    run_aro_evals(loaded, processor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument('--adapter', type=str)
    parser.add_argument('--coco', action='store_true')
    parser.add_argument('--flickr', action='store_true')
    parser.add_argument('--urban', action='store_true')

    args = parser.parse_args()

    clip_model = CLIPModel.from_pretrained(args.base_model)
    clip_processor = CLIPProcessor.from_pretrained(args.base_model)

    if args.adapter:
      run_aro_on_lora(args, clip_processor, clip_model, args.adapter)
    else:
      run_aro_evals(args, clip_model, clip_processor)
