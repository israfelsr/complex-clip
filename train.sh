#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --job-name=cclip
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm/cclip_%j.log
#export CUDA_VISIBLE_DEVICES=0

# Set environment variables
export PYTHONPATH=$(pwd)
wandb offline

export HF_DATASETS_OFFLINE=1
export PHF_HUB_OFFLINE=1

export OUTPUT_DIR=${SLURM_JOB_ID}
mkdir $WORK/projects/complex-clip/logs/$OUTPUT_DIR

#python scripts/run_clip_offline.py \
torchrun --nproc_per_node=4 scripts/run_clip_offline.py \
    --max_steps=150 \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=256 \
    --eval_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --full_determinism=True \
    --logging_strategy "steps" \
    --logging_steps=1 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 150 \
    --lr_scheduler_type=cosine \
    --ddp_find_unused_parameters=False \
    --weight_decay 0 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --model_name_or_path $WORK/projects/complex-clip/models/clip-vit-base-patch32 \
    --tokenizer_name $WORK/projects/complex-clip/models/clip-vit-base-patch32 \
    --image_processor_name $WORK/projects/complex-clip/models/clip-vit-base-patch32 \
    --output_dir $WORK/projects/complex-clip/logs/$OUTPUT_DIR \
    --dataset_name $FAST/clipfinecap/data/docci.hf \
    --do_train \
    --do_eval \
    --remove_unused_columns False \
    --warmup_steps 10
#--num_train_epochs=10 \