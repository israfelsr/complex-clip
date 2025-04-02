from datasets import load_from_disk
from transformers import CLIPTokenizer
import numpy as np
import matplotlib.pyplot as plt


def main():
    dataset = load_from_disk(
        "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/sharegpt4v.hf"
    )
    model_path = (
        "leonardo_work/EUHPC_D12_071/projects/complex-clip/models/clip-vit-base-patch32"
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)

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
    plt.show()
