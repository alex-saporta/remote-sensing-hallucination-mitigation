#!/bin/bash
#SBATCH --job-name=ova_tag_extraction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ova_job_%j.log
#SBATCH --error=logs/ova_job_%j.err

mkdir -p logs

uv run python src/utils/extract_ova_tags.py