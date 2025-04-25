#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/%j.log

export PYTHONPATH=$(pwd)

python evaluation/evaluate.py \
--model_variant HuggingFace \
--scpp \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/logs/15196243/checkpoint-500/ \
--processor_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/logs/15165092/
--output_dir ./results/15196243/scpp.json
#--model_path /leonardo_work/EUHPC_D12_071/LLM_cp.pt #OpenClip lora
#--model_path $WORK/dci_pick1/ #HuggingFace lora
#--model_path $WORK/projects/complex-clip/models/negclip/negclip.pth
#--model_path $WORK/projects/complex-clip/models/clip-vit-base-patch32 #HuggingFace
#--model_path /leonardo_work/EUHPC_D12_071/longclip/checkpoints/longclip-B.pt #LongCLIP