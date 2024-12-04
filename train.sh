#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --mem=40G
#SBATCH --job-name=complex-clip
#SBATCH --time=0-05:00:00
#SBATCH --output=complex-clip_%j.log      # Output and error log file
export CUDA_VISIBLE_DEVICES=0

# Hyperparameters

# Set environment variables
export PYTHONPATH=$(pwd)
wandb online

export OUTPUT_DIR=$(date '+%d-%m-%YT%H-%M-%S')'-detailclip-b32'
mkdir ../logs/$OUTPUT_DIR

python clipdetails/scripts/run_clip.py \
    --lambda_contrast 1.0 \
    --lambda_details 2.0 \
    --lambda_neg 1.0 \
    --epsilon=1e-3 \
    --num_train_epochs=5 \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --eval_accumulation_steps=1 \
    --learning_rate=1e-3 \
    --full_determinism=True \
    --logging_strategy "steps" \
    --logging_steps=0.01 \
    --evaluation_strategy "steps" \
    --eval_steps 0.05 \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --lr_scheduler_type=cosine \
    --ddp_find_unused_parameters=False \
    --weight_decay 0 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --model_name_or_path openai/clip-vit-base-patch32 \
    --output_dir ../logs/$OUTPUT_DIR \
    --dataset_name ../data/docci.hf \
    --do_train \
    --do_eval \
    --remove_unused_columns False \
    --lora True \
    --warmup_steps 50