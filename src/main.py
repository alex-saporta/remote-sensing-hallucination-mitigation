import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from train import train
from inference import caption_dataset

def main():
    parser = argparse.ArgumentParser(description="VLM Remote Sensing Pipeline")
    
    # Global arguments
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "captio"], help="Run mode")
    parser.add_argument("--fusion_type", type=str, default="concat", choices =["concat", "diff"], help="Type of late fusion for the visual embeddings.")
    parser.add_argument("--vision_type", type=str, default="RemoteCLIP", choices=['RemoteCLIP', 'SkyCLIP', 'GeoRSCLIP', 'standard'], help="Vision encoder to use.")
    parser.add_argument("--llm_model", type=str, default="mistral-7b", choices=['qwen2.5-7B', 'vicuna-7b', 'mistral-7b', 'llama2-7b'], help="LLM to use.")
    parser.add_argument("--caption_method", type=str, default="merge", choices=['one', 'two', 'three', 'four', 'five', 'random', 'merge', 'all'], help="Caption to use as ground truth.")
    parser.add_argument("--merge_captions_path", type=str, default="data/merged_captions_qwen.json", help="Path to the merged captions file to use if caption_method is merge ")
    parser.add_argument("--prompt", type=str, default="Describe the changes between the first image and the second image.", help="The instruction prompt fed to the LLM.")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Image Grounded Guidance
    parser.add_argument("--marine", action="store_true", help="Activate logit-level contrastive guidance")
    parser.add_argument("--guidance_strength", type=float, default=0.7, help="Gamma logit scaling multiplier factor.")
    parser.add_argument("--tags_cache", type=str, default="unified_yolo_prompts.json", help="Name of the pre-computed tags JSON cache file to load.")
    
    # Test
    parser.add_argument("--checkpoint", type=str, help="Path to weights for eval mode")
    parser.add_argument("--report_output", type=str, default="report", help="Name of the file to write the generated captiond and evaluation results into.")

    args = parser.parse_args()

    if args.mode == "train":
        print(f"Starting training M: {args.vision_type} + {args.llm_model}")
        train(args)
    
    elif args.mode == "test":
        caption_dataset(args)

if __name__ == "__main__":
    main()
