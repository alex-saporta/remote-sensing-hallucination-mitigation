import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from train import train
from eval_merge_quality import eval_merge_quality

def main():
    parser = argparse.ArgumentParser(description="VLM Remote Sensing Pipeline")
    
    # Global arguments
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "eval_merge"], help="Run mode")
    parser.add_argument("--vision_type", type=str, default="RemoteCLIP", choices=['RemoteCLIP', 'SkyCLIP', 'GeoRSCLIP', 'standard'])
    parser.add_argument("--llm_model", type=str, default="mistral-7b", choices=['qwen2.5-7B', 'vicuna-7b', 'mistral-7b', 'llama2-7b'])
    parser.add_argument("--caption_method", type=str, default="random", choices=['one', 'two', 'three', 'four', 'five', 'random', 'merge'])
    parser.add_argument("--merge_captions_path", type=str, default="data/merged_captions_qwen.json")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Evaluation
    parser.add_argument("--checkpoint", type=str, help="Path to weights for eval mode")
    parser.add_argument("--eval_output", type=str, default="merge_quality_report.txt")

    args = parser.parse_args()

    if args.mode == "train":
        print(f"Starting training M: {args.vision_type} + {args.llm_model}")
        train(args)
    
    elif args.mode == "eval":
        # Placeholder for evaluation logic
        pass

    elif args.mode == "eval_merge":
        project_root = current_dir.parent
        data_json = project_root / "data" / "SECOND-CC-AUG.json"
        merge_json = project_root / args.merge_captions_path
        eval_merge_quality(
            merge_json_path=merge_json,
            original_json_path=data_json,
            output_file=args.eval_output
        )

if __name__ == "__main__":
    main()
