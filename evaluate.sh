export PYTHONPATH=$(pwd)

python evaluation/evaluate.py \
--model_variant OpenCLIP \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/negclip/negclip.pth \
--classification
