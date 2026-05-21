#!/bin/bash
#SBATCH --job-name=merge_captions
#SBATCH --output=logs/merge_captions_%j.out
#SBATCH --error=logs/merge_captions_%j.err
#SBATCH --time=04:00:00           
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=16G

echo "Starting caption merging with Qwen2.5-1.5B-Instruct..."
echo "Date: $(date)"

uv run python src/run_precompute.py

echo "Caption merging finished."
echo "Date: $(date)"