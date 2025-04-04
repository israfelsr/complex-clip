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
--model_variant OpenCLIP \
--aro \
--model_path $WORK/projects/complex-clip/models/negclip/negclip.pth \
--output_dir ./results/base/clip-vit-base-patch32.json
#--model_path $WORK/projects/complex-clip/models/clip-vit-base-patch32 \

