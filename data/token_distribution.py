from transformers import CLIPTokenizer
import argparse
import json
import os
from collections import Counter
from tqdm import tqdm
from datasets import load_from_disk

COCO_DIR = "/home/bzq999/data/complexclip/eval/coco/2014/coco_karpathy_test.json"
FLICKR_DIR = "/home/bzq999/data/complexclip/eval/flickr30k/flickr30k_test.json"
URBAN1K_DIR = "/home/bzq999/data/complexclip/eval/Urban1k/caption/"
SDCI_ROOT = "/home/bzq999/data/complexclip/eval/sdci_retrieval.hf"
DOCCI_ROOT = "/home/bzq999/data/complexclip/eval/docci_retrieval.hf"
IIW_ROOT = "/home/bzq999/data/complexclip/eval/iiw_retrieval.hf"
LN_ROOT = "/home/bzq999/data/complexclip/eval/localized_narratives.hf"
SHAREGPT4V_ROOT = "/home/bzq999/data/complexclip/eval/sharegpt4v.hf"
SDCI_TRAIN_ROOT = "/home/bzq999/data/complexclip/eval/sdci_train.hf"
DOCCI_TRAIN_ROOT = "/home/bzq999/data/complexclip/eval/docci.hf"

def args_parser():
    parser = argparse.ArgumentParser(description="Analyze token distribution in a dataset using CLIP tokenizer.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to evaluate: coco, flickr30k, or urban1k."
    )
    return parser.parse_args()

def load_captions(dataset):
    if dataset == "coco":
        annotation = json.load(open(COCO_DIR, "r"))
        captions = []
        for item in annotation:
            captions.extend(item['caption'])
        return captions
    elif dataset == "flickr30k":
        annotation = json.load(open(FLICKR_DIR, "r"))
        captions = []
        for item in annotation:
            captions.extend(item['caption'])
        return captions
    elif dataset == "urban1k":
        captions = []
        for f in os.listdir(URBAN1K_DIR):
            filename = os.path.join(URBAN1K_DIR, f)
            with open(filename, 'r', encoding='utf-8') as file:
                caption = file.read().strip()
                captions.append(caption)
        return captions
    elif dataset == "sdci_retrieval":
        dataset = load_from_disk(SDCI_ROOT)
        captions = [caption for item in dataset for caption in item['caption']]
        return captions
    elif dataset == "docci_retrieval":
        dataset = load_from_disk(DOCCI_ROOT)
        captions = [caption for item in dataset for caption in item['caption']]
        return captions
    elif dataset == "ln":
        dataset = load_from_disk(LN_ROOT)
        return dataset['caption']
    elif dataset == "sharegpt4v":
        dataset = load_from_disk(SHAREGPT4V_ROOT)
        return dataset['caption']
    elif dataset == "sdci":
        dataset = load_from_disk(SDCI_TRAIN_ROOT)
        captions = [cap for sample in dataset for cap in sample['caption']]
        return captions
    elif dataset == "docci":
        dataset = load_from_disk(DOCCI_TRAIN_ROOT)
        return dataset['caption']
    elif dataset == "iiw":
        dataset = load_from_disk(IIW_ROOT)
        captions = [caption for item in dataset for caption in item['caption']]
        return captions
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

def main(args):
    tokenizer = CLIPTokenizer.from_pretrained("/home/bzq999/data/complexclip/models/clip-vit-base-patch32/")
    vocab_size = tokenizer.vocab_size
    vocab_set = set(range(vocab_size))

    captions = load_captions(args.dataset)
    token_counter = Counter()
    all_tokens = set()

    for caption in tqdm(captions, desc="Tokenizing captions"):
        tokens = tokenizer.encode(caption, add_special_tokens=True)
        token_counter.update(tokens)
        all_tokens.update(tokens)

    covered = len(all_tokens)
    proportion_covered = covered / vocab_size

    print(f"Total unique tokens in CLIP vocabulary: {vocab_size}")
    print(f"Unique tokens found in dataset: {covered}")
    print(f"Proportion of vocabulary covered: {proportion_covered:.4f}")
    print("\nToken distribution (token_id: count) for top 50 tokens:")
    for token_id, count in token_counter.most_common(50):
        print(f"{token_id}: {count}")

    # Save the full token distribution to a file for later analysis/plotting
    with open(f"{args.dataset}token_distribution.json", "w") as f:
        # Save as a list of [token_id, count] pairs, sorted by count descending
        json.dump(token_counter.most_common(), f, indent=2)

if __name__ == "__main__":
    args = args_parser()
    main(args)
