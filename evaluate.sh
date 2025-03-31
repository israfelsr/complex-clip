export PYTHONPATH=$(pwd)

python evaluation/evaluate.py \
--model_variant OpenCLIP \
--retrieval urban sdci docci iiw \
--model_path models/negclip
#--model_path models/clip-vit-base-patch32
