import argparse
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.extractors import ExtractorFactory

VOCABULARY = [
    "buildings", 
    "roads", 
    "water bodies", 
    "bare land", 
    "trees", 
    "forests", 
    "agricultural land"
]

class TagExtractionDataset(Dataset):
    def __init__(self, samples, project_root, processor):
        self.samples = samples
        self.project_root = project_root
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample['filename']
        split = sample.get('split', 'train')
        
        data_root_split = self.project_root / "data" / split
        path_a = data_root_split / "rgb" / "A" / img_name
        path_b = data_root_split / "rgb" / "B" / img_name
        
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        
        # Extract pixel values natively
        inputs_a = self.processor(images=img_a, return_tensors="pt")
        inputs_b = self.processor(images=img_b, return_tensors="pt")

        val_a = inputs_a['pixel_values']
        val_b = inputs_b['pixel_values']

        return {
            "img_name": img_name,
            "pixel_values_a": val_a.squeeze(0),
            "pixel_values_b": val_b.squeeze(0)
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-computation of MARINE tags for each image of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for server GPU parallel execution")
    parser.add_argument("--top_k", type=int, default=2, help="Number of tags to extract")
    parser.add_argument("--output_name", type=str, default="marine_tags.json", help="Saved cache name")
    parser.add_argument("--extractor", type=str, default="SkyCLIP", choices=["SkyCLIP"], help="Object extractor to use on the dataet")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Setup depends on device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device} | Extractor: {args.extractor} | Batch Size: {args.batch_size} | Top K: {args.top_k}")
    
    # 2. Data preparation
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    json_path = project_root / "data" / "SECOND-CC-AUG.json"
    
    with open(json_path, "r") as f:
        full_data = json.load(f)
    all_samples = full_data.get("images", [])

    # Ensure all sample are pairs (i.e. both A and B img are present)
    valid_samples = []
    for sample in all_samples:
        img_name = sample['filename']
        split = sample.get('split', 'train')
        path_a = project_root / "data" / split / "rgb" / "A" / img_name
        path_b = project_root / "data" / split / "rgb" / "B" / img_name
        if path_a.exists() and path_b.exists():
            valid_samples.append(sample)

    print(f"Processing {len(valid_samples)} valid entries via parallel batches...")

    # 3. Initialize the components
    extractor = ExtractorFactory.get_extractor(args.extractor, device)

    dataset = TagExtractionDataset(valid_samples, project_root, extractor.processor)

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,        
        pin_memory=True, 
    )

    tag_cache = {}


    print("Running batch extraction matrix...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Tags"):
            img_names = batch['img_name']
            p_a = batch['pixel_values_a'].to(device)
            p_b = batch['pixel_values_b'].to(device)

            tags_batch_a = extractor.extract_batch_tags(p_a, VOCABULARY, args.top_k)
            tags_batch_b = extractor.extract_batch_tags(p_b, VOCABULARY, args.top_k)

            # Zip results back to map tracking structures
            for idx, name in enumerate(img_names):
                tag_cache[name] = {
                    "A": tags_batch_a[idx],
                    "B": tags_batch_b[idx]
                }

    output_path = project_root / "data" / args.output_name
    with open(output_path, "w") as f:
        json.dump(tag_cache, f, indent=4)

    print(f"\nProcessing finished successfully. Output generated at: {output_path}")

if __name__ == "__main__":
    main()