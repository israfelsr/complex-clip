model=longclip #openai-clip:ViT-B/32 # Choose the model you want to test

for dataset in Flickr30k_Order # VG_Relation VG_Attribution COCO_Order Flickr30k_Order
do
    python3 main_aro.py --dataset=$dataset --model-name=$model --device=cuda
done
