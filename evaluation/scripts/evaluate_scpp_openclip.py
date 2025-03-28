###########  Code to test Hugging face multi-modal models  ###
import argparse
import open_clip
from PIL import Image
import torch, os, json
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer

img_path = "/leonardo_work/EUHPC_D12_071/coco/2014/val2014/COCO_val2014_"  #'path to the images folder'
data_path = "../clip-fine-cap/scpp/data/"  #'path to folder with caption files'
fnames = os.listdir(data_path)
image_size = 224
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ARO classification")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--lora",
        type=int,
        default=None,
        help="Load lora model using with Peft",
    )

    args = parser.parse_args()
    return args


def run_scpp_on_lora(image_processor, tokenizer, base_clip_model, lora_weight_path):
    from peft import PeftModel

    loaded = PeftModel.from_pretrained(base_clip_model, lora_weight_path)
    loaded = loaded.merge_and_unload()
    loaded.to("cuda")
    run_scpp_evals(loaded, image_processor, tokenizer, device)


def run_scpp_evals(model, image_processor, tokenizer, device):

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
            model.eval()

            inputs = image_processor(image).unsqueeze(0).to(device)
            img_feats = model.encode_image(inputs)
            img_feats = F.normalize(img_feats, dim=-1)

            inputs = tokenizer(p1).to(device)
            p1_feats = model.encode_text(inputs)
            p1_feats = F.normalize(p1_feats, dim=-1)

            inputs = tokenizer(p2).to(device)
            p2_feats = model.encode_text(inputs)
            p2_feats = F.normalize(p2_feats, dim=-1)

            inputs = tokenizer(ref).to(device)
            neg_feats = model.encode_text(inputs)
            neg_feats = F.normalize(neg_feats, dim=-1)

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


def main():
    args = parse_args()
    clip, _, processor = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
        lora=args.lora,
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    clip.load_state_dict(sd)
    print(f"Model loaded from: {args.model_path}")
    clip.to(device)

    run_scpp_evals(clip, processor, tokenizer, device)


if __name__ == "__main__":
    main()
