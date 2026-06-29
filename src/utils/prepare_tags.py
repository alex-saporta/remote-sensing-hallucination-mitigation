import json
import torch
import argparse
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# DYNAMIC PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TAGS_DIR = PROJECT_ROOT / "data" / "tags"
OUTPUT_PATH = PROJECT_ROOT / "data" / "tags"/ "final_tags.json"
DETECTOR_FILES = ["yolo_tags.json", "dino_tags.json", "ova_tags.json"]

def get_consensus_tags(all_detections_for_image):
    """Rule-based: Majority voting (Threshold >= 2 detectors)."""
    a_tags = [tag for det in all_detections_for_image for tag in det.get("A", [])]
    b_tags = [tag for det in all_detections_for_image for tag in det.get("B", [])]
    return ([t for t, c in Counter(a_tags).items() if c >= 2], 
            [t for t, c in Counter(b_tags).items() if c >= 2])

def aggregate_via_llm(all_dets, model, tokenizer, device):
    """LLM-based: Uses the VLM to resolve detector conflicts."""
    prompt = f"Detectors provided these labels: {str(all_dets)}. Reconcile these into a concise summary."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def create_unified_caption(tags_a, tags_b):
    """Converts tags into the requested 'noun verb' format."""
    set_a, set_b = set(tags_a), set(tags_b)
    all_tags = sorted(list(set_a | set_b))
    phrases = []
    for tag in all_tags:
        if tag in set_b and tag not in set_a: phrases.append(f"{tag} appeared")
        elif tag in set_a and tag not in set_b: phrases.append(f"{tag} disappeared")
        else: phrases.append(f"{tag} remained unchanged")
    return ", ".join(phrases) if phrases else "no significant land-cover change detected"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["rule", "llm"], default="rule")
    parser.add_argument("--output_name", default="report_rule")
    args = parser.parse_args()

    # Load data
    data_by_detector = [json.load(open(TAGS_DIR / f)) for f in DETECTOR_FILES if (TAGS_DIR/f).exists()]
    image_keys = set().union(*[d.keys() for d in data_by_detector])

    # Setup LLM if needed
    model, tokenizer, device = None, None, "cpu"
    if args.strategy == "llm":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to(device).eval()

    final_guidance = {}
    for img_name in image_keys:
        all_dets = [d[img_name] for d in data_by_detector if img_name in d]
        
        if args.strategy == "rule":
            tags_a, tags_b = get_consensus_tags(all_dets)
            final_guidance[img_name] = create_unified_caption(tags_a, tags_b)
        else:
            final_guidance[img_name] = aggregate_via_llm(all_dets, model, tokenizer, device)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_guidance, f, indent=4)
    print(f"Success! Guidance ({args.strategy}) saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()