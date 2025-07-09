#!/bin/bash
#SBATCH --job-name=complexity
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=%j_slurm.out
#SBATCH --nodes=1

conda activate complexclip

# Run your script using poetry
python ./data/complexity.py --dataset ln