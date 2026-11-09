# Hallucination mitigation for change captioning in remote sensing

Course project, *Deep Learning for Earth Observation* (DL4EO), TU Berlin, summer semester 2026, supervised by Marvin Beese.
Authors: Apolline Colnot, Alexandre Saporta.

Vision-language models trained on ground-level images tend to hallucinate when they describe satellite images: their language priors override the visual evidence, and they mention buildings or roads that are not there. This project builds a bi-temporal vision-language model that describes the change between two satellite images, and adapts [MARINE](https://github.com/Linxi-ZHAO/MARINE), a training-free guidance method, to reduce these hallucinations at inference.

## Method

**Model.** A LLaVA-style architecture adapted to two images. Images at t1 and t2 are encoded separately by a frozen CLIP ViT-L/14 encoder, their features are concatenated, and a 2-layer MLP projector (GELU) maps them into the embedding space of a frozen Qwen 2.5-7B. **Only the projector is trained.** Three encoders were compared in the report: OpenCLIP ViT-L-14, RemoteCLIP and SkyCLIP. The code also supports GeoRSCLIP, which was not evaluated.

**Training.** SECOND-CC (6,041 image pairs, five captions each). The five captions of a pair are merged into one reference of at most 15 words with Qwen1.5-Instruct. AdamW, learning rate 1e-4 with linear warm-up and cosine decay, batch size 16, 5 epochs, cross-entropy loss. Split: 8,438 training, 1,190 validation and 1,227 test samples.

**MARINE for two images.**
1. Three open-vocabulary detectors (YOLO-World, Grounding DINO, OVA-Det) run on both images.
2. For each detector, the two tag lists are merged with change verbs: *appeared*, *disappeared*, *remains unchanged*.
3. The three lists are aggregated by majority vote (rule-based) or by Mistral-7B-Instruct (LLM-based).
4. At each decoding step, the logits with and without the tag prompt are mixed with a guidance strength γ = 0.7.

**Metrics.** Caption quality: BLEU-4, ROUGE-L, CIDEr, METEOR. Hallucination: CHAIR, AMBER (generative), POPE (discriminative).

## Results

- Without mitigation, OpenCLIP and SkyCLIP give better captions than RemoteCLIP, which is the most robust to hallucination on POPE and AMBER.
- MARINE, rule-based or LLM-based, improves the hallucination metrics for all three encoders (for example AMBER F1 from 0.05 to 0.26 with SkyCLIP). The best POPE F1 is 0.45, with SkyCLIP and rule-based guidance.
- The cost is fluency: forcing detector tags makes the captions more rigid.
- **Overall the scores stay low.** The mitigation works in principle, but not fully: when the detectors miss complex remote-sensing structures, the model inherits their errors.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/alex-saporta/remote-sensing-hallucination-mitigation
cd remote-sensing-hallucination-mitigation
uv sync
source .venv/bin/activate
```

## Data

Download the [SECOND-CC dataset](https://github.com/ChangeCapsInRS/SecondCC) and arrange it as follows (data, tags and checkpoints are not stored in this repository):

```
data/
├── checkpoints/            # trained projector weights
├── tags/                   # pre-computed detector tags for MARINE
├── train/  val/  test/     # images
├── SECOND-CC-AUG.json      # original captions
└── merged-SECOND-CC.json   # merged captions
```

## Usage

All options: `uv run python src/main.py --help`

**Train** the projector:

```bash
uv run python src/main.py \
    --mode train \
    --fusion_type concat \
    --vision_type standard \
    --llm_model qwen2.5-7B \
    --caption_method merge \
    --merge_captions_path data/merged-SECOND-CC.json \
    --prompt "Given these 2 satellite images, describe in one sentence the geographical changes that occurred :" \
    --batch_size 16 \
    --epochs 5 \
    --lr 1e-4
```

**Test**, with MARINE guidance:

```bash
uv run python src/main.py \
    --mode test \
    --checkpoint "data/checkpoints/merge_standard_concat_qwen2.5-7B/best_projector.pt" \
    --llm_model "qwen2.5-7B" \
    --vision_type "standard" \
    --fusion_type "concat" \
    --tags_cache "unified_yolo_prompts.json" \
    --marine \
    --guidance_strength 0.7 \
    --prompt "Given these 2 satellite images, describe in one sentence the geographical changes that occurred :"
```

`train_mitigated.sh` is a job script for the course cluster; adapt its paths before use.

## Repository layout

| Path | Content |
|---|---|
| `src/main.py` | Entry point for training and testing |
| `src/model.py` | Bi-temporal model: encoders, fusion, projector, frozen LLM |
| `src/train.py` | Training loop |
| `src/inference.py` | Generation and the MARINE guidance loop |
| `src/data_loader.py` | SECOND-CC loading |
| `src/utils/` | Metrics and helpers |

## References

- SECOND-CC dataset: https://github.com/ChangeCapsInRS/SecondCC
- MARINE (image-grounded guidance): https://github.com/Linxi-ZHAO/MARINE
