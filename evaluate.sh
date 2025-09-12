#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --job-name=evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/%j.log
export PYTHONPATH=$(pwd)

# Base CLIP
python evaluation/evaluate.py \
--model_variant HuggingFace \
--winoground \
--output_dir results/winoground/base.json \
--processor_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/clip-vit-base-patch32/ \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/clip-vit-base-patch32/
#--retrieval coco flickr urban sdci docci iiw \

# CE-CLIP
python evaluation/evaluate.py \
--model_variant OpenCLIP \
--winoground \
--output_dir results/winoground/CE-CLIP.json \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/Clip_all.pt # OpenCLIP
#--retrieval coco flickr urban sdci docci iiw \

# LongCLIP
python evaluation/evaluate.py \
--model_variant LongCLIP \
--winoground \
--output_dir results/winoground/longclip.json \
--model_path /leonardo_work/EUHPC_D12_071/longclip/checkpoints/longclip-B.pt #LongCLIP

# DCI
python evaluation/evaluate.py \
--model_variant HuggingFace \
--winoground \
--lora \
--output_dir results/winoground/dci.json \
--processor_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/clip-vit-base-patch32/ \
--model_path /leonardo_work/EUHPC_D12_071/dci_pick1/

# DAC
python evaluation/evaluate.py \
--model_variant OpenCLIP \
--winoground \
--lora \
--output_dir results/winoground/dac_llm.json \
--model_path /leonardo_work/EUHPC_D12_071/LLM_cp.pt #OpenClip lora

# NegCLIP
python evaluation/evaluate.py \
--model_variant OpenCLIP \
--winoground \
--output_dir results/winoground/negclip.json \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/negclip/negclip.pth

# LSS_sharegpt
python evaluation/evaluate.py \
--model_variant HuggingFace \
--winoground \
--output_dir results/winoground/lss.json \
--processor_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/clip-vit-base-patch32/ \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/logs/15340816/checkpoint-3000

# DreamLIP
python evaluation/evaluate.py \
--model_variant OpenCLIP \
--winoground \
--output_dir results/winoground/dreamlip.json \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/models/cc30m_dreamlip_vitb16.pt