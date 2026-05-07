import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

# Setup path to import src modules
current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from data_loader import SecondCCDataset

def test_dataloader(split="train", caption_method="random"):
    print(f"Testing DataLoader with split='{split}' and caption_method='{caption_method}'...")
    
    # 1. Initialize dataset
    try:
        dataset = SecondCCDataset(
            json_path="data/SECOND-CC-AUG.json", 
            data_root="data", 
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
    
    args = parser.parse_args()
    
    test_dataloader(split=args.split, caption_method=args.method)