import json
import torch
from pathlib import Path
from utils.metrics import Metrics
import argparse

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
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue la qualité des captions mergées.")
    
    parser.add_argument(
        "--merge_file", 
        type=str, 
        required=True, 
        help="Chemin vers le fichier JSON des captions générées/mergées."
    )
    parser.add_argument(
        "--ref_file", 
        type=str, 
        required=True, 
        help="Chemin vers le fichier JSON original contenant les références."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Dossier de destination pour sauvegarder les résultats."
    )

    args = parser.parse_args()

    # 1. S'assurer que le dossier de sortie existe, sinon le créer
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Définir le chemin complet du fichier de résultats
    output_file_path = out_dir / "merge_quality_results.txt"

    # 3. Lancer l'évaluation
    print("Début de l'évaluation des métriques...")
    results = eval_merge_quality(
        merge_json_path=args.merge_file,
        original_json_path=args.ref_file,
        output_file=str(output_file_path)
    )
    
    if isinstance(results, str):
        # Si la fonction retourne un string, c'est qu'elle a rencontré une erreur (comme défini dans ton code)
        print(results)