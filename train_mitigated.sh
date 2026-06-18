#!/bin/bash
#SBATCH --job-name=vlm_mitigation
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err

mkdir -p logs

export HF_HOME="/data/dl4eo_p2g1/hf_cache"
export TORCH_HOME="/data/dl4eo_p2g1/torch_cache"

uv run python src/main.py \
    --mode train \
    --fusion_type concat \
    --vision_type standard \
    --llm_model qwen2.5-7B \
    --caption_method merge \
    --merge_captions_path /data/dl4eo_p2g1/data/merged-SECOND-CC.json \
    --prompt "Given these 2 satellite images, describe in one sentence the geographical changes that occurred :" \
    --tags_cache unified_yolo_prompts.json \
    --marine \
    --guidance_strength 0.7 \
    --batch_size 16 \
    --epochs 5 \
    --lr 1e-4


    