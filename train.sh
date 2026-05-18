#!/bin/bash
#SBATCH --job-name=vlm_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err

mkdir -p logs

export HF_HOME="/data/dl4eo_p2g1/hf_cache"
export TORCH_HOME="/data/dl4eo_p2g1/torch_cache"

uv run python src/main.py \
    --mode train \
    --vision_type RemoteCLIP \
    --llm_model mistral-7b \
    --caption_method random \
    --batch_size 4 \
    --epochs 5 \
    --lr 1e-4