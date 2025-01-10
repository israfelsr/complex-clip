import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from dac.src.training.params import parse_args
from dac.src.open_clip import create_model_and_transforms
from dataset_zoo import (
    COCO_Retrieval,
    Flickr30k_Retrieval,
    Urban1k_Retrieval,
    DOCCI_Retrieval,
    sDCI_Retrieval,
)
from model_zoo.clip_models import CLIPWrapper
import csv
from dac.src.open_clip import tokenize

ARO_ROOT = "/fsx/homes/Yova.Kementchedjhieva@mbzuai.ac.ae/.cache/prerelease_bow"
COCO_ROOT = "/fsx/homes/Yova.Kementchedjhieva@mbzuai.ac.ae/.cache/coco/2014"
FLICKR_ROOT = "/fsx/homes/Yova.Kementchedjhieva@mbzuai.ac.ae/.cache/flickr30k/images"
SDCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/sdci_retrieval.hf"
DOCCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/docci_retrieval.hf"


def interpolate_models(theta_0, theta_1, alpha):
    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()
    }
    return theta


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.debug:
        if args.debug_ip is None:
            import pydevd_pycharm

            pydevd_pycharm.settrace(
                os.environ["SSH_CONNECTION"].split()[0],
                port=args.debug_port,
                stdoutToServer=True,
                stderrToServer=True,
                suspend=False,
            )

    # model, image_preprocess = clip.load("ViT-B/32", jit=False, lora=lora)
    model, _, image_preprocess = create_model_and_transforms(
        "/leonardo_work/EUHPC_D12_071/ViT-B-32.pt",
        "openai",
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        lora=args.lora,
        freeze_img=args.freeze_img,
        kqv_lora=args.kqv_lora,
    )
    if args.resume != "None":
        checkpoint = torch.load(args.resume, map_location="cpu")
        print("Load from path:", args.resume)
        sd = checkpoint["state_dict"]

        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        model.load_state_dict(sd)
    else:
        print("no checkpoint")
        args.resume = "Output_no_checkpoint/"
    model.eval()
    model = CLIPWrapper(model, args.device)

    # root_dir = COCO_ROOT
    # coco_dataset = COCO_Retrieval(
    #    image_preprocess=image_preprocess,
    #    download=True,
    #    root_dir=root_dir,
    #    split="test",
    # )
    # collate_fn = _default_collate if image_preprocess is None else None
    # coco_loader = DataLoader(coco_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    # coco_scores = model.get_retrieval_scores_dataset(coco_loader)
    # coco_records = coco_dataset.evaluate_scores(coco_scores)

    # root_dir = FLICKR_ROOT
    # flickr_dataset = Flickr30k_Retrieval(
    #    image_preprocess=image_preprocess,
    #    download=True,
    #    root_dir=root_dir,
    #    split="test",
    # )
    # collate_fn = _default_collate if image_preprocess is None else None
    # flickr_loader = DataLoader(flickr_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    # flickr_scores = model.get_retrieval_scores_dataset(flickr_loader)
    # flickr_records = flickr_dataset.evaluate_scores(flickr_scores)

    # urban_dataset = Urban1k_Retrieval(
    #     image_preprocess=image_preprocess,
    # )
    # collate_fn = _default_collate if image_preprocess is None else None
    # urban_loader = DataLoader(
    #     urban_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    # )
    # urban_scores = model.get_retrieval_scores_dataset(urban_loader)
    # urban_records = urban_dataset.evaluate_scores(urban_scores)

    root_dir = SDCI_ROOT
    sdci_dataset = sDCI_Retrieval(
        image_preprocess=image_preprocess,
        root_dir=root_dir,
    )
    collate_fn = _default_collate if image_preprocess is None else None
    sdci_loader = DataLoader(
        sdci_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    sdci_scores = model.get_retrieval_scores_dataset(sdci_loader)
    sdci_records = sdci_dataset.evaluate_scores(sdci_scores)

    root_dir = DOCCI_ROOT
    docci_dataset = DOCCI_Retrieval(
        image_preprocess=image_preprocess,
        root_dir=root_dir,
    )
    collate_fn = _default_collate if image_preprocess is None else None
    docci_loader = DataLoader(
        docci_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    docci_scores = model.get_retrieval_scores_dataset(docci_loader)
    docci_records = docci_dataset.evaluate_scores(docci_scores)

    print(f"sdci Dataset")
    print(
        f"ImagePrec@1={sdci_records[0]['ImagePrec@1']} - TextPrec@1={sdci_records[0]['TextPrec@1']}"
    )
    print(f"docci Dataset")
    print(
        f"ImagePrec@1={docci_records[0]['ImagePrec@1']} - TextPrec@1={docci_records[0]['TextPrec@1']}"
    )
