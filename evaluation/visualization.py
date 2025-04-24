from datasets import load_from_disk
from transformers import CLIPTokenizer
import numpy as np
import matplotlib.pyplot as plt


def main():
    dataset = load_from_disk(
        "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/sharegpt4v.hf"
    )

    def explode_captions(batch):
        # Repeat image paths for each caption
        images = [
            img_path
            for img_path, captions in zip(batch["image"], batch["caption"])
            for _ in range(len(captions))
        ]
        # Flatten all captions
        captions = [caption for sublist in batch["caption"] for caption in sublist]
        return {"image": images, "caption": captions}

    dataset = dataset.map(
        explode_captions,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=1000,
        desc="Creating 1 caption per sample",
    )

    print(f"Dataset samples {len(dataset)}")
    print(f"Dataset images {len(set(dataset['image']))}")
    print(f"Dataset captions {len(set(dataset['caption']))}")
    model_path = (
        "leonardo_work/EUHPC_D12_071/projects/complex-clip/models/clip-vit-base-patch32"
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
    if isinstance(dataset[0]["caption"], list):

        def map_list_to_str(sample):
            sample["caption"] = sample["caption"][0]
            return sample

        dataset = dataset.map(map_list_to_str)

    def get_batch_token_lengths(batch, tokenizer):
        # Tokenize all captions in batch at once
        encodings = tokenizer(
            batch["caption"],  # Assumes 'caption' is the text field
            truncation=False,  # Get true lengths (or True to match your eval setup)
            return_length=True,  # Returns token counts directly
        )
        batch["token_length"] = encodings["length"]  # List of lengths
        return batch

    # Apply to dataset with batch processing
    dataset = dataset.map(
        lambda x: get_batch_token_lengths(x, tokenizer),
        batched=True,
        batch_size=1000,  # Adjust based on RAM (larger = faster)
        desc="Batched length calculation",
    )

    # Extract all lengths
    lengths = np.array(dataset["token_length"])

    # Basic stats
    print(f"Mean length: {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"Min/Max: {lengths.min()}, {lengths.max()}")
    print(f"Percentiles (25/50/75/90): {np.percentile(lengths, [25, 50, 75, 90])}")

    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=50, edgecolor="black")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title("Caption Length Distribution")
    plt.axvline(
        lengths.mean(), color="r", linestyle="--", label=f"Mean: {lengths.mean():.1f}"
    )
    plt.legend()
    plt.savefig("plot.png")


if __name__ == "__main__":
    main()
