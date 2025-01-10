model=longclip #openai-clip:ViT-B/32 # Choose the model you want to test

# Deterministic Experiments
for dataset in Urban1k_Retrieval #COCO_Retrieval Flickr30k_Retrieval
do
    python3 main_retrieval.py --dataset=$dataset --model-name=$model --device=cuda
done
