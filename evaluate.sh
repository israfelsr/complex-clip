export PYTHONPATH=$(pwd)

python evaluation/evaluate.py \
--model_variant OpenCLIP \
--retrieval sdci docci iiw \
--model_path models/negclip/negclip.pth
#--model_path models/clip-vit-base-patch32
