export PYTHONPATH=$(pwd)

python evaluation/evaluate.py \
--model_variant HuggingFace \
--model_path models/clip-vit-base-patch32 \
--retrieval flickr
