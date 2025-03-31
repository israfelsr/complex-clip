from torch.utils.data import DataLoader
import torch

from aro.clip_aro_wrap import AroWrap
from aro.dataset_zoo import (
    COCO_Retrieval,
    Flickr30k_Retrieval,
    Urban1k_Retrieval,
    sDCI_Retrieval,
    DOCCI_Retrieval,
    IIW_Retrieval,
)

ARO_DIR = "/leonardo_work/EUHPC_D12_071/ARO"
COCO_DIR = "/leonardo_work/EUHPC_D12_071/coco/2014"
FLICKR_DIR = "/leonardo_work/EUHPC_D12_071/data/flickr30k"
URBAN_ROOT = "/leonardo_work/EUHPC_D12_071/Urban1k"
SDCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/sdci_retrieval.hf"
DOCCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/docci_retrieval.hf"
IIW_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/iiw_retrieval.hf"


def evaluate_retrieval(retrieval, model, device):
    model = AroWrap(model, device)

    def collate_fn(batch):
        import code

        code.interact(local=locals())
        images = [sample[dataset_info["image_column"]] for sample in batch]
        labels = [sample[dataset_info["label_column"]] for sample in batch]
        return images, torch.tensor(labels, device=device)

    if "coco" in retrieval:
        root_dir = COCO_DIR
        coco_dataset = COCO_Retrieval(
            root_dir=root_dir,
            split="test",
        )
        coco_loader = DataLoader(
            coco_dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=collate_fn,
        )
        coco_scores = model.get_retrieval_scores_dataset(coco_loader)
        coco_records = coco_dataset.evaluate_scores(coco_scores)
