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

COCO_DIR = "/leonardo_work/EUHPC_D12_071/coco/2014"
FLICKR_DIR = "/leonardo_work/EUHPC_D12_071/data/flickr30k"
ARO_DIR = "/leonardo_work/EUHPC_D12_071/ARO"
URBAN_ROOT = "/leonardo_work/EUHPC_D12_071/Urban1k"
SDCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/sdci_retrieval.hf"
DOCCI_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/docci_retrieval.hf"
IIW_ROOT = "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/iiw_retrieval.hf"


DATASETS = {
    "coco": {
        "class": COCO_Retrieval,
        "root_dir": "/leonardo_work/EUHPC_D12_071/coco/2014",
        "split": "test",
    },
    "flickr": {  # Example new dataset
        "class": Flickr30k_Retrieval,
        "root_dir": "/leonardo_work/EUHPC_D12_071/data/flickr30k",
        "split": "test",
    },
}


def evaluate_retrieval(retrieval, model, device):
    model = AroWrap(model, device)

    def collate_fn(batch):
        images = [sample["image"] for sample in batch]
        return images

    for dataset_name in retrieval:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = DATASETS[dataset_name]
        dataset = config["class"](
            root_dir=config["root_dir"],
            split=config["test"],
        )
        loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=collate_fn,
        )
        scores = model.get_retrieval_scores_dataset(loader)
        records = dataset.evaluate_scores(scores)

    import code

    code.interact(local=locals())
