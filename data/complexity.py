import argparse
from datasets import load_from_disk
import stanza
from tqdm import tqdm
stanza.download('en')
import json
import os
from nltk.tree import Tree
from stanza.models.common.doc import Document

BATCH_SIZE = 512

COCO_DIR = "/home/bzq999/data/complexclip/eval/coco/2014/coco_karpathy_test.json"
FLICKR_DIR = "/home/bzq999/data/complexclip/eval/flickr30k/flickr30k_test.json"
URBAN1K_DIR = "/home/bzq999/data/complexclip/eval/Urban1k/caption/"
SDCI_ROOT = "/home/bzq999/data/complexclip/eval/sdci_retrieval.hf"
DOCCI_ROOT = "/home/bzq999/data/complexclip/eval/docci_retrieval.hf"
IIW_ROOT = "/home/bzq999/data/complexclip/eval/iiw_retrieval.hf"
LN_ROOT = "/home/bzq999/data/complexclip/eval/localized_narratives.hf"
SHAREGPT4V_ROOT = "/home/bzq999/data/complexclip/eval/sharegpt4v.hf"

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


def calc_yngve(t, par):
    if type(t) == str:
        return par
    else:
        val = 0
        for i, child in enumerate(reversed(t)):
            val += calc_yngve(child, par+i)
        return val

def is_sent(val):
    return len(val) > 0 and val[0] == "S"

def calc_frazier(t, par, par_lab):
    # print t
    # print par
    if type(t) == str:
        # print par-1
        return par-1
    else:
        val = 0
        for i, child in enumerate(t):
            # For all but the leftmost child, zero
            score = 0
            if i == 0:
                my_lab = t.label()
                # If it's a sentence, and not duplicated, add 1.5
                if is_sent(my_lab):
                    score = (0 if is_sent(par_lab) else par+1.5)
                # Otherwise, unless it's a root node, add one
                elif my_lab != "" and my_lab != "ROOT" and my_lab != "TOP":
                    score = par + 1
            val += calc_frazier(child, score, my_lab)
        return val

def get_scores(sentence: str, nlp_pipeline):
    doc = nlp_pipeline(sentence)
    tree = doc.sentences[0].constituency
    t = Tree.fromstring(str(tree))
    yngve = calc_yngve(t, 0)
    frazier = calc_frazier(t, 0, "")
    return {'yngve_score': yngve, 'frazier_score': frazier}

def get_dataset_scores(sentences: list, nlp_pipeline, batch_size:int=32):
    yngve = []
    frazier = []
    
    processed_docs = []
    for i in tqdm(range(0,len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        docs = [Document([], text=sent) for sent in batch]
        processed_docs.extend(nlp_pipeline.bulk_process(docs))
    
    for doc in processed_docs:
        tree = doc.sentences[0].constituency
        t = Tree.fromstring(str(tree))
        yngve.append(calc_yngve(t, 0))
        frazier.append(calc_frazier(t, 0, ""))
    return {'yngve_score': yngve, 'frazier_score': frazier}


def args_parser():
    parser = argparse.ArgumentParser(description="Compute complexity scores for a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset to evalute."
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
    elif dataset == "iiw":
        dataset = load_from_disk(IIW_ROOT)
        captions = [caption for item in dataset for caption in item['caption']]
        return captions
    elif dataset == "ln":
        dataset = load_from_disk(LN_ROOT)
        dataset = dataset.select(range(0,1000))
        return dataset['caption']
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

def main(args):
    nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', use_gpu=True, tokenize_batch_size=512)
    captions = load_captions(args.dataset)

    # y_score = []
    # f_score = []

    # for caption in tqdm(captions):
    #     scores = get_complexity_scores(caption, nlp)
    #     y_score.append(scores['yngve_score'])
    #     f_score.append(scores['frazier_score'])

    scores = get_dataset_scores(captions, nlp, batch_size=BATCH_SIZE)

    # avg_yngve = sum(y_score) / len(y_score) if y_score else 0
    # avg_frazier = sum(f_score) / len(f_score) if f_score else 0

    avg_yngve = sum(scores['yngve_score']) / len(scores['yngve_score']) if scores['yngve_score'] else 0
    avg_frazier = sum(scores['frazier_score']) / len(scores['frazier_score']) if scores['frazier_score'] else 0
    print(f"Average Yngve score: {avg_yngve:.4f}")
    print(f"Average Frazier score: {avg_frazier:.4f}")

if __name__ == "__main__":
    args = args_parser()
    main(args)
