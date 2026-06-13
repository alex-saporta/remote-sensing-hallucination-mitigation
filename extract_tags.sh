#!/bin/bash
#SBATCH --job-name=dino_tag_extraction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dino_job_%j.log
#SBATCH --error=logs/dino_job_%j.err

mkdir -p logs

# Run the python script natively inside your clean workspace environment
uv run python src/utils/extract_dino_tags.py