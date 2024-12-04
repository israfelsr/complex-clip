from datasets import Dataset, concatenate_datasets, DatasetDict
import json
from pathlib import Path
from tqdm import tqdm

DATASET_BASE = Path("<path-to-dataset>/densely_captioned_images")
BASE_IMAGES_PATH = Path("photos")


def main():
    dataset = {}
    with open(DATASET_BASE / "splits.json") as jsonf:
        split_metadata = json.load(jsonf)

    for split in split_metadata.keys():
        sources = split_metadata[split]

        datasets = []
        for source_path in tqdm(sources, desc=f"Creating {split} dataset"):
            complete_caption_path = DATASET_BASE / "complete" / source_path
            with open(complete_caption_path) as entry_file:
                base_data = json.load(entry_file)
            image_path = DATASET_BASE / BASE_IMAGES_PATH / base_data["image"]
            annotations = base_data["summaries"]["base"]

            ds = Dataset.from_dict(
                {"image": [str(image_path)] * len(annotations), "caption": annotations}
            )
            datasets.append(ds)
        dataset[split] = concatenate_datasets(datasets)

    dci = DatasetDict(dataset)
    hf_path = "<path-to-save>"
    dci.save_to_disk(hf_path)
    print(f"Dataset saved to {hf_path}")


if __name__ == "__main__":
    main()
