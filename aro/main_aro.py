import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    parser.add_argument("--dataset", default="VG_Relation", type=str, choices=["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"])
    parser.add_argument("--seed", default=1, type=int)
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="outputs", type=str)
    return parser.parse_args()

    
def main(args):
    seed_all(args.seed)
    
    model, image_preprocess = get_model(args.model_name, args.device)
    
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    
    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
    collate_fn = _default_collate if image_preprocess is None else None
    
    joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    scores = model.get_retrieval_scores_batched(joint_loader)
    result_records = dataset.evaluate_scores(scores)
    
    for record in result_records:
        record.update({"Model": args.model_name, "Dataset": args.dataset, "Seed": args.seed})
    
    output_file = os.path.join(args.output_dir, f"{args.dataset}.csv")
    df = pd.DataFrame(result_records)

    if args.dataset == "VG_Relation":
      df = df[~df.Relation.isin(symmetric)]
      vgr_metric = df.Accuracy.mean()
      print(f"VG-Relation Macro Accuracy: {vgr_metric}")
    elif args.dataset == "VG_Attribution":
      vga_metric = df.Accuracy.mean()
      print(f"VG-Attribution Macro Accuracy: {vga_metric}")
    elif args.dataset == "COCO_Order":
      coco_metric = df['Precision@1'].mean()
      print(f"COCO Precision@1: {coco_metric}")  
    elif args.dataset == "Flickr30k_Order":
      flickr_metric = df['Precision@1'].mean()
      print(f"Flickr Precision@1: {flickr_metric}")
    
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)

    else:
        df.to_csv(output_file)
        
    if args.save_scores:
        save_scores(scores, args)

    
if __name__ == "__main__":
    args = config()
    main(args)
