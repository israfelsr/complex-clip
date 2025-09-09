import argparse
from datasets import load_from_disk
import stanza
from tqdm import tqdm
stanza.download('en')
import json
import os
from nltk.tree import Tree
from stanza.models.common.doc import Document
import pickle
import glob

# Download model once if not present
# stanza.download('en')

# --- Your Scoring Functions (Unchanged) ---
# (calc_yngve, is_sent, calc_frazier functions remain the same)
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
    if type(t) == str:
        return par-1
    else:
        val = 0
        for i, child in enumerate(t):
            score = 0
            if i == 0:
                my_lab = t.label()
                if is_sent(my_lab):
                    score = (0 if is_sent(par_lab) else par+1.5)
                elif my_lab != "" and my_lab != "ROOT" and my_lab != "TOP":
                    score = par + 1
            val += calc_frazier(child, score, my_lab)
        return val

# --- Dataset Loading (Unchanged) ---
# (load_captions function and directory constants remain the same)
COCO_DIR = "/home/bzq999/data/complexclip/eval/coco/2014/coco_karpathy_test.json"
FLICKR_DIR = "/home/bzq999/data/complexclip/eval/flickr30k/flickr30k_test.json"
URBAN1K_DIR = "/home/bzq999/data/complexclip/eval/Urban1k/caption/"
SDCI_ROOT = "/home/bzq999/data/complexclip/eval/sdci_retrieval.hf"
DOCCI_ROOT = "/home/bzq999/data/complexclip/eval/docci_retrieval.hf"
IIW_ROOT = "/home/bzq999/data/complexclip/eval/iiw_retrieval.hf"
LN_ROOT = "/home/bzq999/data/complexclip/eval/localized_narratives.hf"
SHAREGPT4V_ROOT = "/home/bzq999/data/complexclip/eval/sharegpt4v.hf"

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
        return dataset['caption']
    elif dataset == "sharegpt4v":
        dataset = load_from_disk(SHAREGPT4V_ROOT)
        return dataset['caption']
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

# --- STAGE 1: PARSE AND SAVE ---
def run_parse_stage(args):
    """Loads captions, processes them in chunks, and saves processed docs."""
    print(f"--- Running Stage 1: Parsing dataset '{args.dataset}' ---")
    nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', use_gpu=True, tokenize_batch_size=512)
    captions = load_captions(args.dataset)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Total captions to process: {len(captions)}")
    print(f"Processing in chunks of {args.chunk_size}. Batches will be saved to '{output_dir}'")
    
    processed_docs = []
    for i in tqdm(range(0, len(captions), args.chunk_size), desc="Processing Chunks"):
        chunk_captions = captions[i:i+args.chunk_size]
        
        # Use nlp(list_of_strings) for efficient batch processing
        docs = [Document([], text=sent) for sent in chunk_captions]
        processed_docs.extend(nlp.bulk_process(docs))
        
        # Save the processed chunk to a file
        chunk_filename = os.path.join(output_dir, f"processed_chunk_{i//args.chunk_size}.pkl")
        with open(chunk_filename, 'wb') as f_out:
            pickle.dump(processed_docs, f_out)
            
    print(f"✅ Parsing complete. Processed chunks saved in '{output_dir}'.")

# --- STAGE 2: LOAD AND SCORE ---
def run_score_stage(args):
    """Loads processed docs from files and calculates scores."""
    print(f"--- Running Stage 2: Scoring files from '{args.output_dir}' ---")
    
    chunk_files = sorted(glob.glob(os.path.join(args.output_dir, "processed_chunk_*.pkl")))
    if not chunk_files:
        raise FileNotFoundError(f"No processed chunk files found in '{args.output_dir}'. Did you run the 'parse' stage?")
        
    print(f"Found {len(chunk_files)} chunk files to score.")
    
    all_yngve = []
    all_frazier = []
    
    for chunk_file in tqdm(chunk_files, desc="Scoring Chunks"):
        with open(chunk_file, 'rb') as f_in:
            processed_docs = pickle.load(f_in)
            
        for doc in processed_docs:
            if not doc.sentences or not doc.sentences[0].constituency:
                continue
            
            tree = doc.sentences[0].constituency
            t = Tree.fromstring(str(tree))
            all_yngve.append(calc_yngve(t, 0))
            all_frazier.append(calc_frazier(t, 0, ""))

    avg_yngve = sum(all_yngve) / len(all_yngve) if all_yngve else 0
    avg_frazier = sum(all_frazier) / len(all_frazier) if all_frazier else 0
    
    print("--- Final Scores ---")
    print(f"Total scored captions: {len(all_yngve)}")
    print(f"✅ Average Yngve score: {avg_yngve:.4f}")
    print(f"✅ Average Frazier score: {avg_frazier:.4f}")

def args_parser():
    parser = argparse.ArgumentParser(description="Compute complexity scores for a dataset.")
    parser.add_argument("--stage", type=str, required=True, choices=['parse', 'score'], help="Stage to run: 'parse' docs or 'score' them.")
    parser.add_argument("--dataset", type=str, help="Dataset to evaluate (required for 'parse' stage).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save/load processed chunks.")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of captions to process before saving a chunk.")
    return parser.parse_args()

def main():
    args = args_parser()
    if args.stage == 'parse':
        if not args.dataset:
            raise ValueError("Argument --dataset is required for the 'parse' stage.")
        run_parse_stage(args)
    elif args.stage == 'score':
        run_score_stage(args)

if __name__ == "__main__":
    main()