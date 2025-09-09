from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch.nn as nn
import torch

DATASET_PATH = "/leonardo_work/EUHPC_D12_071/projects/complex-clip/dataset/winoground"


def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]


def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]


def group_correct(result):
    return image_correct(result) and text_correct(result)


def evaluate_winoground(model, device):
    scores = []
    # dataset = load_dataset("facebook/winoground", split="test")
    dataset = load_from_disk(DATASET_PATH)
    with torch.no_grad():
        for item in tqdm(dataset):
            image0 = item["image_0"]
            image1 = item["image_1"]
            caption0 = item["caption_0"]
            caption1 = item["caption_1"]

            image0_feats = model.encode_image(image0, device)
            image1_feats = model.encode_image(image1, device)
            caption0_feats = model.encode_text(caption0, device)
            caption1_feats = model.encode_text(caption1, device)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_c0_i0 = cos(caption0_feats, image0_feats)
            cos_c0_i1 = cos(caption0_feats, image1_feats)
            cos_c1_i0 = cos(caption1_feats, image0_feats)
            cos_c1_i1 = cos(caption1_feats, image1_feats)

            scores.append(
                {
                    "id": item["id"],
                    "c0_i0": cos_c0_i0,
                    "c0_i1": cos_c0_i1,
                    "c1_i0": cos_c1_i0,
                    "c1_i1": cos_c1_i1,
                }
            )

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(scores)
    return {
        "text": text_correct_count / denominator,
        "image": image_correct_count / denominator,
        "group": group_correct_count / denominator,
    }
