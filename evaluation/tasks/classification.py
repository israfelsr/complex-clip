from datasets import load_from_disk
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from evaluation.templates.templates import DATASET


def evaluate_dataset(
    model,
    dataset,
    classes,
    templates,
    image_column,
    label_column,
    device,
    accelerator,
    batch_size=32,
):
    """Evaluate the model on a specific dataset."""

    def zeroshot_classifier(classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]
                class_embeddings = model.encode_text(texts, device)
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

    # Create a DataLoader for batched inference
    def collate_fn(batch):
        images = [sample[image_column] for sample in batch]
        labels = [sample[label_column] for sample in batch]
        return images, torch.tensor(labels, device=device)

    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,  # Faster data transfer to GPU
    )

    # Prepare the DataLoader for distributed inference
    dataloader = accelerator.prepare(dataloader)

    # Evaluate
    top1, top5, n = [], [], 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [model.processor(images)]
            import code

            code.interact(local=locals())
            target = torch.tensor(sample[label_column]).to(device)
            image_embedding = model.encode_image(sample[image_column], device)
            logits = 100.0 * image_embedding @ zeroshot_weights

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1.append(acc1)
            top5.append(acc5)

    # Summarize results
    top1 = np.sum(gather_object([top1]))
    top5 = np.sum(gather_object([top5]))

    return (top1 / len(dataset)) * 100, (top5 / len(dataset)) * 100


def evaluate_classification(model, device, accelerator):
    print("Starting Classification")
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

        print(f"Dataset {dataset} loaded")
        top1, top5 = evaluate_dataset(
            model,
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
    return results

    if accelerator.is_main_process:
        # Summary of all results at the end
        print("\nSummary of results:")
        print(f"{'Dataset':<20} {'Top-1 Accuracy (%)':<20} {'Top-5 Accuracy (%)':<20}")

        # Print each dataset's result in a clean, tabular format
        for result in results:
            print(
                f"{result['Dataset']:<20} {result['Top-1']:<20.2f} {result['Top-5']:<20.2f}"
            )
