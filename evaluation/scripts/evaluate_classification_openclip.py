import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import torch
from tqdm import tqdm
from pathlib import Path
import os
from accelerate import Accelerator
from accelerate.utils import gather_object

from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor, HfArgumentParser
from datasets import load_from_disk
from peft import PeftModel
import open_clip

from evaluation.templates.templates import DATASET


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="openai/clip-vit-large-patch14", metadata={"help": "CLIP Model"}
    )
    tokenizer_name: Optional[str] = field(
        default="", metadata={"help": "Tokenizer used to compute similarity"}
    )
    image_processor_name: Optional[str] = field(
        default="", metadata={"help": "Image Processor used to compute similarity"}
    )
    local_files_only: Optional[bool] = field(
        default=False, metadata={"help": "If using only local files"}
    )
    lora: Optional[bool] = field(
        default=None, metadata={"help": "Load lora model using with Peft."}
    )


def evaluate_dataset(
    clip,
    tokenizer,
    processor,
    dataset,
    classes,
    templates,
    image_column,
    label_column,
    device,
    accelerator,
):
    """Evaluate the model on a specific dataset."""

    def zeroshot_classifier(classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [
                    template.format(classname) for template in templates
                ]  # format with class
                texts = tokenizer(texts).to(device)
                class_embeddings = clip.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    # Prepare zero-shot classifier
    print("Processing labels")
    zeroshot_weights = zeroshot_classifier(classes, templates)

    def accuracy(output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [
            float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            for k in topk
        ]

    # Evaluate
    with torch.no_grad():
        with accelerator.split_between_processes(
            list(range(len(dataset)))
        ) as dataset_indices:
            ds = dataset.select(dataset_indices)
            top1, top5, n = [], [], 0.0
            for sample in tqdm(ds):
                images = processor(sample[image_column]).unsqueeze(0).to(device)
                target = torch.tensor(sample[label_column]).to(device)

                image_features = clip.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ zeroshot_weights

                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1.append(acc1)
                top5.append(acc5)

    # Summarize results
    top1 = np.sum(gather_object([top1]))
    top5 = np.sum(gather_object([top5]))

    return (top1 / len(dataset)) * 100, (top5 / len(dataset)) * 100


def main():
    parser = HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]
    accelerator = Accelerator()
    device = accelerator.device
    print(model_args)
    clip, _, processor = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
        lora=model_args.lora,
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    checkpoint = torch.load(model_args.model_name_or_path, map_location="cpu")
    sd = checkpoint["state_dict"]

    if next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    clip.load_state_dict(sd)
    print(f"Model loaded from: {model_args.model_name_or_path}")
    clip.to(accelerator.process_index)

    datasets_to_evaluate = [
        {
            "name": "cifar10",
            "path": "/leonardo_work/EUHPC_D12_071/data/eval/cifar10.hf",
            "image_column": "img",
            "label_column": "label",
            "split": "test",
        },
        {
            "name": "cifar100",
            "path": "/leonardo_work/EUHPC_D12_071/data/eval/cifar100.hf",
            "image_column": "img",
            "label_column": "fine_label",
            "split": "test",
        },
        {
            "name": "imagenet",
            "path": "/leonardo_work/EUHPC_D12_071/data/eval/imagenet.hf",
            "image_column": "image",
            "label_column": "label",
            "split": None,
        },
    ]
    results = []
    for dataset_info in datasets_to_evaluate:
        dataset = load_from_disk(dataset_info["path"])
        if dataset_info["split"] is not None:
            dataset = dataset[dataset_info["split"]]

        classes, templates = DATASET[dataset_info["name"]]

        print("Dataset loaded")
        print(dataset)

        top1, top5 = evaluate_dataset(
            clip,
            tokenizer,
            processor,
            dataset,
            classes,
            templates,
            dataset_info["image_column"],
            dataset_info["label_column"],
            device,
            accelerator,
        )
        # Store the results in the list
        results.append({"Dataset": dataset_info["name"], "Top-1": top1, "Top-5": top5})

    if accelerator.is_main_process:
        # Summary of all results at the end
        print("\nSummary of results:")
        print(f"{'Dataset':<20} {'Top-1 Accuracy (%)':<20} {'Top-5 Accuracy (%)':<20}")

        # Print each dataset's result in a clean, tabular format
        for result in results:
            print(
                f"{result['Dataset']:<20} {result['Top-1']:<20.2f} {result['Top-5']:<20.2f}"
            )


if __name__ == "__main__":
    main()
