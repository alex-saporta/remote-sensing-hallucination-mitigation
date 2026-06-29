import torch
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm
from model import VLM
from utils.metrics import Metrics

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

def load_inference_model(args, device):
    model = VLM(fusion_type=args.fusion_type, llm_model=args.llm_model, vision_type=args.vision_type)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device).eval()
    return model

def run_model_forward(sample, model, tokenizer, args, device):
    """
    Executes inference for a single sample, including generation, 
    CHAIR/AMBER processing, and POPE discriminative probes.
    """
    # 1. Image preparation
    pix_a = sample['pixel_values_a'].unsqueeze(0).to(device)
    pix_b = sample['pixel_values_b'].unsqueeze(0).to(device)

    # 2. Generate Caption
    if args.marine:
        ids_base = sample['input_ids_base'].unsqueeze(0).to(device)
        ids_guided = sample['input_ids_guided'].unsqueeze(0).to(device)
        with torch.no_grad():
            return model.generate_marine(pix_a, pix_b, ids_base, ids_guided, tokenizer, args.guidance_strength, device=device)
    else:
        with torch.no_grad():
            return model.generate(pix_a, pix_b, tokenizer, prompt=args.prompt, device=device)

def caption_dataset(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    model = load_inference_model(args, device)
    evaluator = Metrics(device)
    
    # Load data (SecondCCDataset handles gt_changes/POPE probes automatically)
    from data_loader import SecondCCDataset
    test_ds = SecondCCDataset(project_root/"data/SECOND-CC-AUG.json", project_root/"data", split="test")
    
    all_preds, all_targets, all_records = [], [], []

    for i in tqdm(range(len(test_ds)), desc="Testing"):
        sample = test_ds[i]
        pred = run_model_forward(sample, model, tokenizer, args, device)
        
        # Metrics Accumulation
        all_preds.append(pred)
        all_targets.append([s['raw'] for s in test_ds.samples[i]['sentences']])
        
        # Generative Hallucination (CHAIR/AMBER)
        gt_changes = [sample['gt_changes']]
        evaluator.compute_chair([pred], gt_changes)
        evaluator.compute_amber([pred], gt_changes)
        
        # Discriminative Hallucination (POPE)
        probes = evaluator.pope.prepare_probes(gt_changes)
        ans = [model.generate(sample['pixel_values_a'].unsqueeze(0).to(device), 
                             sample['pixel_values_b'].unsqueeze(0).to(device), 
                             tokenizer, prompt=p["question"], max_new_tokens=5) for p in probes[0]]
        evaluator.compute_pope([ans], probes)
        
        all_records.append({"sample": test_ds.samples[i]['filename'], "pred": pred, "gt": all_targets[-1][0]})

    # Compute Final Report
    report = {**evaluator.compute_caption_quality(all_preds, all_targets),
              **evaluator.compute_chair_epoch(), **evaluator.compute_amber_epoch(), **evaluator.compute_pope_epoch()}
    
    # Save JSON Report
    output_path = project_root / "data" / args.report_output
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": report, "predictions": all_records}, f, indent=4)