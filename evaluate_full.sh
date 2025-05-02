#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --job-name=evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=24:00:00  # Increased time since we're running multiple evaluations
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/%j.log

export PYTHONPATH=$(pwd)

# Define base paths
CHECKPOINTS_DIR="$WORK/projects/complex-clip/logs/15340002"
OUTPUT_BASE_DIR="./results/15340002"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Loop through all checkpoint directories
for checkpoint_dir in "$CHECKPOINTS_DIR"/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        # Extract checkpoint number (e.g., 100 from checkpoint-100)
        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)
        
        echo "Evaluating checkpoint: $checkpoint_num"
        
        python evaluation/evaluate.py \
            --model_variant HuggingFace \
            --classification \
            --retrieval coco flickr urban sdci docci iiw \
            --scpp \
            --aro \
            --model_path "$checkpoint_dir" \
            --processor_path "$WORK/projects/complex-clip/logs/15165092/" \
            --output_dir "$OUTPUT_BASE_DIR/results-$checkpoint_num.json"
    fi
done

echo "All evaluations completed!"