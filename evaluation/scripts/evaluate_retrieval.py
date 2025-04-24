# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from aro.dataset_zoo import (
    COCO_Retrieval,
    Flickr30k_Retrieval,
    Urban1k_Retrieval,
    sDCI_Retrieval,
    DOCCI_Retrieval,
    IIW_Retrieval,
)
from aro.clip_aro_wrap import AROtoHFCLIPWrap
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from torch.utils.data import DataLoader

import argparse

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

ARO_DIR = "/leonardo_work/EUHPC_D12_071/ARO"
COCO_DIR = "/leonardo_work/EUHPC_D12_071/coco/2014"
FLICKR_DIR = "/leonardo_work/EUHPC_D12_071/data/flickr30k"
URBAN_ROOT = "/leonardo_work/EUHPC_D12_071/Urban1k"
SDCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/sdci_retrieval.hf"
DOCCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/docci_retrieval.hf"
IIW_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/iiw_retrieval.hf"


def run_aro_evals(
    args,
    model: CLIPModel,
    image_processor: CLIPImageProcessor,
    tokenizer: CLIPTokenizer,
    device,
):
    model = AROtoHFCLIPWrap(model, tokenizer, device)

    if args.coco:
        root_dir = COCO_DIR
        coco_dataset = COCO_Retrieval(
            image_preprocess=image_processor,
            download=False,
            root_dir=root_dir,
            split="test",
        )
        collate_fn = _default_collate if image_processor is None else None
        coco_loader = DataLoader(
            coco_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
        )
        coco_scores = model.get_retrieval_scores_dataset(coco_loader)
        coco_records = coco_dataset.evaluate_scores(coco_scores)

    if args.flickr:
        root_dir = FLICKR_DIR
        flickr_dataset = Flickr30k_Retrieval(
            image_preprocess=image_processor,
            download=True,
            root_dir=root_dir,
            split="test",
        )
        collate_fn = _default_collate if image_processor is None else None
        flickr_loader = DataLoader(
            flickr_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
        )
        flickr_scores = model.get_retrieval_scores_dataset(flickr_loader)
        flickr_records = flickr_dataset.evaluate_scores(flickr_scores)

    if args.urban:
        root_dir = URBAN_ROOT
        urban_dataset = Urban1k_Retrieval(
            image_preprocess=image_processor,
            root_dir=root_dir,
        )
        collate_fn = _default_collate if image_processor is None else None
        urban_loader = DataLoader(
            urban_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
        )
        urban_scores = model.get_retrieval_scores_dataset(urban_loader)
        urban_records = urban_dataset.evaluate_scores(urban_scores)

    if args.sdci:
        root_dir = SDCI_ROOT
        sdci_dataset = sDCI_Retrieval(
            image_preprocess=image_processor,
            root_dir=root_dir,
        )
        collate_fn = _default_collate if image_processor is None else None
        sdci_loader = DataLoader(
            sdci_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
        )
        sdci_scores = model.get_retrieval_scores_dataset(sdci_loader)
        sdci_records = sdci_dataset.evaluate_scores(sdci_scores)

    if args.docci:
        root_dir = DOCCI_ROOT
        docci_dataset = DOCCI_Retrieval(
            image_preprocess=image_processor,
            root_dir=root_dir,
        )
        collate_fn = _default_collate if image_processor is None else None
        docci_loader = DataLoader(
            docci_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
        )
        docci_scores = model.get_retrieval_scores_dataset(docci_loader)
        docci_records = docci_dataset.evaluate_scores(docci_scores)

    if args.iiw:
        root_dir = IIW_ROOT
        iiw_dataset = IIW_Retrieval(
            image_preprocess=image_processor,
            root_dir=root_dir,
        )
        collate_fn = _default_collate if image_processor is None else None
        iiw_loader = DataLoader(
            iiw_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
        )
        iiw_scores = model.get_retrieval_scores_dataset(iiw_loader)
        iiw_records = iiw_dataset.evaluate_scores(iiw_scores)

    print("----Results----")
    if args.coco:
        print(f"COCO Dataset")
        print(
            f"ImagePrec@1={coco_records[0]['ImagePrec@1']} - TextPrec@1={coco_records[0]['TextPrec@1']}"
        )
    if args.flickr:
        print(f"flickr Dataset")
        print(
            f"ImagePrec@1={flickr_records[0]['ImagePrec@1']} - TextPrec@1={flickr_records[0]['TextPrec@1']}"
        )
    if args.urban:
        print(f"urban Dataset")
        print(
            f"ImagePrec@1={urban_records[0]['ImagePrec@1']} - TextPrec@1={urban_records[0]['TextPrec@1']}"
        )
    if args.sdci:
        print(f"sdci Dataset")
        print(
            f"ImagePrec@1={sdci_records[0]['ImagePrec@1']} - TextPrec@1={sdci_records[0]['TextPrec@1']}"
        )
    if args.docci:
        print(f"docci Dataset")
        print(
            f"ImagePrec@1={docci_records[0]['ImagePrec@1']} - TextPrec@1={docci_records[0]['TextPrec@1']}"
        )
    if args.iiw:
        print(f"iiw Dataset")
        print(
            f"ImagePrec@1={iiw_records[0]['ImagePrec@1']} - TextPrec@1={iiw_records[0]['TextPrec@1']}"
        )


def run_aro_on_lora(
    args, image_processor, tokenizer, base_clip_model, lora_weight_path
):
    from peft import PeftModel

    loaded = PeftModel.from_pretrained(base_clip_model, lora_weight_path)
    loaded = loaded.merge_and_unload()
    loaded.to("cuda")
    run_aro_evals(args, loaded, image_processor, tokenizer, "cuda")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ARO classification")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="evalute retrieval in coco dataset",
    )
    parser.add_argument(
        "--flickr",
        action="store_true",
        help="evalute retrieval in flickr dataset",
    )
    parser.add_argument(
        "--urban",
        action="store_true",
        help="evalute retrieval in urban1k dataset",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Load lora model using with Peft",
    )
    parser.add_argument(
        "--sdci",
        action="store_true",
        help="evalute retrieval in sdci dataset",
    )
    parser.add_argument(
        "--docci",
        action="store_true",
        help="evalute retrieval in docci dataset",
    )
    parser.add_argument(
        "--iiw",
        action="store_true",
        help="evalute retrieval in docci dataset",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    image_processor = CLIPImageProcessor.from_pretrained(
        "/leonardo_work/EUHPC_D12_071/data/HF/clip_processor.hf", local_files_only=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "/leonardo_work/EUHPC_D12_071/data/HF/clip_tokenizer.hf", local_files_only=True
    )

    if args.lora:
        print(f"Using LoRA model from {args.model_path}")
        clip_model = CLIPModel.from_pretrained(
            "/leonardo_work/EUHPC_D12_071/data/HF/clip-base", local_files_only=True
        )
        run_aro_on_lora(
            args,
            image_processor,
            tokenizer,
            clip_model,
            args.model_path,
        )
    else:
        print(f"Using finetuned model from {args.model_path}")
        loaded = CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(
            "cuda"
        )
        run_aro_evals(args, loaded, image_processor, tokenizer, "cuda")
