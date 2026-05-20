import json
import torch
from pathlib import Path
from utils.metrics import Metrics

def eval_merge_quality(merge_json_path, original_json_path, output_file="merge_quality_results.txt"):
    """
    Evaluates the quality of merged captions and saves the report to a text file.
    """
    # 1. Load Data
    merge_path = Path(merge_json_path)
    if not merge_path.exists():
        return f"Error: Merged file not found at {merge_json_path}"

    with open(merge_json_path, 'r') as f:
        merged_data = json.load(f)
        
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)

    # 2. Setup Evaluator
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    evaluator = Metrics(device)
    
    all_preds = []
    all_targets = []

    # 3. Match merged captions with references
    for item in original_data.get("images", []):
        fname = item["filename"]
        if fname in merged_data:
            all_preds.append(merged_data[fname])
            references = [s["raw"] for s in item["sentences"]]
            all_targets.append(references)

    if not all_preds:
        return "No matching samples found to evaluate."

    # 4. Compute Metrics
    results = evaluator.compute_caption_quality(all_preds, all_targets)
    
    # 5. Write to file
    report = [
        "Merged caption quality evaluation",
        f"Merged File:    {merge_path.name}",
        f"Samples:        {len(all_preds)}",
        f"BLEU-4:         {results.get('bleu4', 0.0):.4f}",
        f"ROUGE-L:        {results.get('rougeL_fmeasure', 0.0):.4f}",
        f"CIDEr:          {results.get('cider', 0.0):.4f}",
        f"METEOR:         {results.get('meteor', 0.0):.4f}",
        "="*40
    ]
    
    report_content = "\n".join(report)
    
    with open(output_file, "w") as f:
        f.write(report_content)
    
    print(f"Merge quality evaluation saved to: {output_file}")
    return results