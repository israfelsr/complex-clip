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
--retrieval urban sdci docci iiw \
--model_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/logs/9904959-cclip/checkpoint-300 \
--processor_path /leonardo_work/EUHPC_D12_071/projects/complex-clip/logs/9904959-cclip \
--output_dir ./results/9904959/ckpt300.json
#--model_path models/clip-vit-base-patch32
