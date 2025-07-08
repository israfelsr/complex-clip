import argparse
from datasets import load_from_disk
import stanza
from tqdm import tqdm
stanza.download('en')
import json
import os


COCO_DIR = "/home/bzq999/data/complexclip/eval/coco/2014/coco_karpathy_test.json"
FLICKR_DIR = "/home/bzq999/data/complexclip/eval/flickr30k/flickr30k_test.json"
URBAN1K_DIR = "/home/bzq999/data/complexclip/eval/Urban1k/caption/"
SDCI_ROOT = "/home/bzq999/data/complexclip/eval/sdci_retrieval.hf"
DOCCI_ROOT = "/home/bzq999/data/complexclip/eval/docci_retrieval.hf"
IIW_ROOT = "/home/bzq999/data/complexclip/eval/iiw_retrieval.hf"

def get_complexity_scores(sentence: str, nlp_pipeline):
    """
    Calculates the Yngve and Frazier complexity scores for a single sentence.

    Args:
        sentence: The input text string.
        nlp_pipeline: The initialized stanza pipeline.

    Returns:
        A dictionary containing the Yngve score and Frazier score.
        Returns {'yngve_score': 0, 'frazier_score': 0} if the sentence can't be parsed.
    """
    doc = nlp_pipeline(sentence)
    if not doc.sentences or not doc.sentences[0].constituency:
        return {'yngve_score': 0, 'frazier_score': 0}

    tree = doc.sentences[0].constituency
    words = tree.leaf_labels()
    num_words = len(words)

    if num_words == 0:
        return {'yngve_score': 0, 'frazier_score': 0}

    # --- Yngve Score Calculation ---
    total_yngve = 0
    # Helper function to traverse for Yngve score
    def _calculate_yngve(node, depth):
        nonlocal total_yngve
        if node.is_leaf():
            total_yngve += depth
            return
        # The depth for children is the current depth + number of preceding sisters
        for i, child in enumerate(node.children):
            # The number of preceding sisters is 'i'
            _calculate_yngve(child, depth + i)

    # Start traversal from the root's children
    for i, child in enumerate(tree.children):
        _calculate_yngve(child, i)
    
    avg_yngve = total_yngve / num_words if num_words > 0 else 0

    # --- Frazier Score Calculation ---
    total_frazier = 0
    # Helper function to traverse for Frazier score
    def _calculate_frazier(node):
        nonlocal total_frazier
        if node.is_leaf() or node.is_preterminal():
            return
        
        # Add 1.5 for each non-rightmost constituent attached to any S-node
        # (This is a common interpretation of the Frazier heuristic)
        if node.label.startswith('S'):
            for i, child in enumerate(node.children):
                if i < len(node.children) - 1 and not child.is_leaf(): # If not the last child
                    total_frazier += 1.5 # Or another value based on specific literature

        # Recurse through all children
        for child in node.children:
            _calculate_frazier(child)

    _calculate_frazier(tree)

    return {
        'yngve_score': round(avg_yngve, 4),
        'frazier_score': round(total_frazier, 4)
    }

def args_parser():
    parser = argparse.ArgumentParser(description="Compute complexity scores for a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset to evalute."
    )
    return parser.parse_args()

def main(args):
    nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', use_gpu=True)

    if args.dataset == "coco":
        annotation = json.load(open(COCO_DIR, "r"))
        captions = []
        for item in annotation:
            captions.extend(item['caption'])
    if args.dataset == "flickr30k":
        annotation = json.load(open(FLICKR_DIR, "r"))
        captions = []
        for item in annotation:
            captions.extend(item['caption'])
    if args.dataset == "urban1k":
        captions = []
        for f in os.listdir(URBAN1K_DIR):
            filename = os.path.join(URBAN1K_DIR, f)
            with open(filename, 'r', encoding='utf-8') as file:
                caption = file.read().strip()
                captions.append(caption)
    if args.dataset == "sdci_retrieval":
        dataset = load_from_disk(SDCI_ROOT)
        captions = []
        for _, ann in dataset:
            captions.extend(ann)
    if args.dataset == "docci_retrieval":
        dataset = load_from_disk(DOCCI_ROOT)
        captions = []
        for _, ann in dataset:
            captions.extend(ann)
    if args.dataset == "iiw":
        dataset = load_from_disk(IIW_ROOT)
        captions = []
        for _, ann in dataset:
            captions.extend(ann)

    y_score = []
    f_score = []
    for caption in tqdm(captions):
        scores = get_complexity_scores(caption, nlp)
        y_score.append(scores['yngve_score'])
        f_score.append(scores['frazier_score'])

    avg_yngve = sum(y_score) / len(y_score) if y_score else 0
    avg_frazier = sum(f_score) / len(f_score) if f_score else 0
    print(f"Average Yngve score: {avg_yngve:.4f}")
    print(f"Average Frazier score: {avg_frazier:.4f}")

if __name__ == "__main__":
    args = args_parser()
    main(args)
