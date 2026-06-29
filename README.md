# Hallucination Mitigation for Change Captioning in Remote Sensing

This project implements a Large Vision-Language Model (LVLM) pipeline for Remote Sensing Change Captioning. It features a custom [**MARINE**](https://github.com/Linxi-ZHAO/MARINE) framework designed to mitigate object hallucinations at inference time.

## Overview
This repository provides a modular pipeline to train, test, and evaluate LVLMs on the [Second-CC Dataset](https://github.com/ChangeCapsInRS/SecondCC). It supports multi-modal fusion, various vision-encoder backbones based on the CLIP ViT Large Patch 14 architecture (Standard ViT-L-14, RemoteCLIP, SkyCLIP, GeoRSCLIP), and includes a suite of standard generative metrics and specialized hallucination benchmarks.

## Project Structure

```text
project-2-team-1/
├── data/               # Dataset files and checkpoints
│   ├── checkpoints/            # Trained model weights
│   ├── tags/                   # Pre-computed tags for MARINE
│   ├── test/                   # Test set images
│   ├── train/                  # Training set images
│   ├── val/                    # Validation set images
│   └── SECOND-CC-AUG.json      # Captions for the dataset
│   └── merged-SECOND-CC.json   # Merged captions for the dataset
├── src/                # Core source code
│   ├── utils/             # Metric computation and helpers
│   ├── data_loader.py     # Dataset loading and processing
│   ├── inference.py       # Inference engine and MARINE logic
│   ├── main.py            # Entry point for train/test modes
│   ├── model.py           # VLM architecture definition
│   └── train.py           # Training logic
├── README.md           # Project documentation
```

## Installation

Ensure you have [uv](https://docs.astral.sh/uv/) installed on your system. Then, simply run the following commands to set up and activate the environment:
```bash
# Clone the repository
git clone https://git.tu-berlin.de/rsim/dl4eo-2026-summer/hallucination-mitigation-for-change-captioning-in-remote-sensing/project-2-team-1
cd project-2-team-1
# Automatically create and activate the virtual environment and install all dependencies
uv sync
source .venv/bin/activate
```

## Data Preparation

Download the [Second-CC dataset](https://github.com/ChangeCapsInRS/SecondCC) and place it in the project root to match the following structure:

```
project-2-team-1/data/
├── checkpoints/
├── tags/
├── test/
├── train/
├── val/
├── SECOND-CC-AUG.json
└── merged-SECOND-CC.json
```

## Usage

To discover all the custom possibilities run :
```
uv run python src/main.py --help
```

### 1. Training

To train the pipeline with custom backbones and parameters :
```
uv run python src/main.py \
    --mode train \
    --fusion_type concat \
    --vision_type  standard \
    --llm_model qwen2.5-7B \
    --caption_method merge \
    --merge_captions_path /data/dl4eo_p2g1/data/merged-SECOND-CC.json \
    --prompt "Given these 2 satellite images, describe in one sentence the geographical changes that occurred :" \
    --batch_size 16 \
    --epochs 5 \
    --lr 1e-4
```

### 2. Testing

```
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
```

## Metrics

The pipeline computes a comprehensive suite of metrics:

* **Caption Quality:** BLEU-4, ROUGE-L, CIDEr, METEOR.
* **Hallucination (Generative):** CHAIR (Caption Hallucination Assessment) and AMBER.
* **Hallucination (Discriminative):** POPE (Polling-based Object Probing Evaluation).


## References

We have made use of the following repositories 
- [Second-CC dataset](https://github.com/ChangeCapsInRS/SecondCC)
- [Image Grounded Guidance](https://github.com/Linxi-ZHAO/MARINE)