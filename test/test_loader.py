import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

# Setup path to import src modules
current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from data_loader import SecondCCDataset, precompute_merge_captions

def test_dataloader(split="train", caption_method="random"):
    print(f"Testing DataLoader with split='{split}' and caption_method='{caption_method}'...")
    
    # 1. Initialize dataset
    try:
        dataset = SecondCCDataset(
            json_path="data/SECOND-CC-AUG.json", 
            data_root="data",
            llm_model="qwen2.5-3B",
            split=split,
            caption_method=caption_method
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if len(dataset) == 0:
        return

    # 2. Initialize DataLoader
    # Added shuffle=True to get different samples on multiple runs
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Get one batch
    print("\nFetching a batch...")
    try:
        batch = next(iter(loader))
        
        # 4. Verify Outputs
        print("-" * 50)
        print("SUCCESS! Batch successfully loaded.")
        print("-" * 50)
        print(f"Image A shape:      {batch['pixel_values_a'].shape}") # Expected: [2, 3, 224, 224]
        print(f"Image B shape:      {batch['pixel_values_b'].shape}") # Expected: [2, 3, 224, 224]
        print(f"Input IDs shape:    {batch['input_ids'].shape}")      # Expected: [2, max_length]
        print(f"Attention Mask:     {batch['attention_mask'].shape}") # Expected: [2, max_length]
        print(f"Prompt Lengths:     {batch['prompt_len']}")           # Expected: tensor([len, len])
        print(f"\nSample Image ID:    {batch['image_id'][0]}")
        print(f"Sample Caption:     {batch['raw_caption'][0]}")
        print("-" * 50)
        
    except StopIteration:
        print("Error: The dataset is empty!")
    except Exception as e:
        print(f"Error during batch fetching: {e}")

def test_precompute(json_path, output_path, model_id, limit):
    print(f"Running precompute_merge_captions...")
    print(f"  JSON path:   {json_path}")
    print(f"  Output path: {output_path}")
    print(f"  Model:       {model_id}")
    print("-" * 50)

    try:
        precompute_merge_captions(
            json_path=json_path,
            output_path=output_path,
            model_id=model_id,
            limit=limit
        )
        print("-" * 50)
        print(f"SUCCESS! Merge captions saved to: {output_path}")

    except Exception as e:
        print(f"Failed to precompute merge captions: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for the VLM DataLoader.")
    parser.add_argument(
        "--split", 
        type=str, 
        default="train", 
        choices=["train", "val", "test"],
        help="Dataset split to test (train, val, test)"
    )
    # The 'choices' argument strictly restricts the inputs to the allowed methods
    parser.add_argument(
        "--method", 
        type=str, 
        default="random", 
        choices=["one", "two", "three", "four", "five", "random", "merge"],
        help="Caption selection method to test"
    )

    parser.add_argument(
        "--merge_captions_path",
        type=str,
        default="data/merge_captions_test.json",
        help="Path to pre-computed merge captions JSON (default: data/merge_captions_test.json)"
    )

    # Precompute-specific arguments
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Run precompute_merge_captions() instead of testing the dataloader"
    )
    parser.add_argument(
        "--precompute_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID to use for precomputing merge captions (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )

    parser.add_argument(
    "--precompute_limit",
    type=int,
    default=2,
    help="Limit the number of samples to precompute (for testing purposes)"
    )

    args = parser.parse_args()

    if args.precompute:
        test_precompute(
            json_path="data/SECOND-CC-AUG.json",
            output_path=args.merge_captions_path,
            model_id=args.precompute_model,
            limit=args.precompute_limit
        )

    else:
        # Early check before loading the dataset
        if args.method == "merge" and not Path(args.merge_captions_path).exists():
            print(
                f"Error: --method=merge requires a pre-computed captions file.\n"
                f"File not found: '{args.merge_captions_path}'\n\n"
                f"Run precompute first:\n"
                f"  python test_loader.py --precompute --merge_captions_path {args.merge_captions_path}"
            )
            sys.exit(1)

        test_dataloader(
            split=args.split,
            caption_method=args.method,
            merge_captions_path=args.merge_captions_path if args.method == "merge" else None
        )