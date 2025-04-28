from PIL import Image
import torch, os, json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

img_path = "/leonardo_work/EUHPC_D12_071/coco/2014/val2014/COCO_val2014_"  #'path to the images folder'
data_path = "../clip-fine-cap/scpp/data/"  #'path to folder with caption files'
fnames = os.listdir(data_path)
image_size = 224


def evaluate_scpp(model, device):
    results = {}
    scpp = []
    for fname in fnames:
        print(
            "=======================================================================",
            flush=True,
        )
        print(
            "=======================================",
            fname,
            "=====================",
            flush=True,
        )
        json_path = os.path.join(data_path, fname)
        total = 0
        correct_img_p1 = 0
        correct_img_p2 = 0

        correct_full = 0  ###  the main task: P1 and P2 closer to Image than Negative
        correct_text = 0

        f = open(json_path)
        data = json.load(f)

        for line in data:
            p1 = line["caption"]
            ref = line["negative_caption"]
            p2 = line["caption2"]  # discard = fp[6]
            img_fname = line["filename"]
            ipath = img_path + img_fname
            image = Image.open(ipath).convert("RGB")
            model.model.eval()

            img_feats = model.encode_image(image, device)

            p1_feats = model.encode_text(p1, device)

            p2_feats = model.encode_text(p2, device)

            neg_feats = model.encode_text(ref, device)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_p1 = cos(
                img_feats, p1_feats
            )  ###  cosine similarities between image and P1 (positive caption 1)
            cos_p2 = cos(
                img_feats, p2_feats
            )  ###  cosine similarities between image and P2 (positive caption 2)
            cos_neg = cos(
                img_feats, neg_feats
            )  ###  cosine similarities between image and Negative (negative caption)
            cos_p1p2 = cos(
                p1_feats, p2_feats
            )  ###  cosine similarities between P1 and P2 for text-only task
            cos_p1_neg = cos(
                p1_feats, neg_feats
            )  ###  cosine similarities between P1 and Negative for text-only task
            cos_p2_neg = cos(
                p2_feats, neg_feats
            )  ###  cosine similarities between P2 and Negative for text-only task

            total += 1

            if cos_p1 > cos_neg and cos_p2 > cos_neg:
                correct_full += 1
            if cos_p1 > cos_neg:
                correct_img_p1 += 1
            if cos_p2 > cos_neg:
                correct_img_p2 += 1
            if cos_p1p2 > cos_p1_neg and cos_p1p2 > cos_p2_neg:
                correct_text += 1

        print(f"====== evaluation results ======", flush=True)
        ave_score = float(correct_full) / float(total)
        print(f"Accuracy image-to-text task: {ave_score * 100}", flush=True)
        ave_score_orig_p1 = float(correct_img_p1) / float(total)
        print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1 * 100}", flush=True)
        ave_score_orig_p2 = float(correct_img_p2) / float(total)
        print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2 * 100}", flush=True)
        ave_score_txt = float(correct_text) / float(total)
        print(f"Accuracy text-only task: {ave_score_txt * 100}", flush=True)

        results[fname] = {
            "I2T": ave_score * 100,
            "P1": ave_score_orig_p1 * 100,
            "P2": ave_score_orig_p2 * 100,
            "TOT": ave_score_txt * 100,
        }
        scpp.append(ave_score * 100)
        scpp.append(ave_score_txt * 100)

    results["scpp"] = np.mean(scpp)
    return results
