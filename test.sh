#!/bin/bash

#SBATCH --job-name=vlm_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_%j.log
#SBATCH --error=logs/test_%j.err

mkdir -p logs

export HF_HOME="/data/dl4eo_p2g1/hf_cache"
export TORCH_HOME="/data/dl4eo_p2g1/torch_cache"

uv run python src/main.py \
    --mode test \
    --checkpoint  "data/checkpoints/merge_standard_concat_qwen2.5-7B/best_projector.pt" \
    --llm_model "qwen2.5-7B" \
    --vision_type "standard" \
    --fusion_type "concat" \
    --tags_cache "unified_yolo_prompts.json" \
    --marine \
    --guidance_strength 0.7 \
    --prompt "Given these 2 satellite images, describe in one sentence the geographical changes that occurred :"