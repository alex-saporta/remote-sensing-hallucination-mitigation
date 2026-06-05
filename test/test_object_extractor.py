import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

# Resolve paths relative to the test/ folder location
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.extractors import ExtractorFactory
from src.utils.extract_tags import TagExtractionDataset, VOCABULARY

def get_real_dataset_samples(limit=10): 
    """Get images from the Second-CC dataset."""
    json_path = project_root / "data" / "SECOND-CC-AUG.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Master dataset annotation file not found at: {json_path}")
        
    with open(json_path, "r") as f:
        full_data = json.load(f)
    all_samples = full_data.get("images", [])
    
    valid_samples = []
    for sample in all_samples:
        img_name = sample['filename']
        split = sample.get('split', 'train')
        
        path_a = project_root / "data" / split / "rgb" / "A" / img_name
        path_b = project_root / "data" / split / "rgb" / "B" / img_name
        
        if path_a.exists() and path_b.exists():
            valid_samples.append(sample)
            if len(valid_samples) == limit:
                break
                
    if not valid_samples:
        raise FileNotFoundError(
            f"Could not find at least {limit} image pairs inside data/ directories to execute test."
        )
        
    return valid_samples

def run_extractor_test():
    # Setup device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running validation pass on device context: {device}\n" + "="*50)

    try:
        TARGET_SAMPLE_COUNT = 10
        real_samples = get_real_dataset_samples(limit=TARGET_SAMPLE_COUNT)
        print(f"Successfully selected {len(real_samples)} image pairs from the Second-CC dataset.")

        extractor = ExtractorFactory.get_extractor("SkyCLIP", device)
        
        dataset = TagExtractionDataset(real_samples, project_root, extractor.processor)

        loader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False,
        )

        tag_cache = {}
        with torch.no_grad():
            for batch in loader:
                img_names = batch['img_name']
                
                # Directly map vision features up to your active execution environment 
                p_a = batch['pixel_values_a'].to(device)
                p_b = batch['pixel_values_b'].to(device)

                tags_batch_a = extractor.extract_batch_tags(p_a, VOCABULARY, top_k=2)
                tags_batch_b = extractor.extract_batch_tags(p_b, VOCABULARY, top_k=2)

                for idx, name in enumerate(img_names):
                    tag_cache[name] = {
                        "A": tags_batch_a[idx],
                        "B": tags_batch_b[idx]
                    }

        # Show extracted objects
        print("\n" + "="*15 + " OBJECT DETECTED " + "="*15)
        print(json.dumps(tag_cache, indent=4))
        print("="*60)
        
        # Structural assertions
        for sample in real_samples:
            name = sample['filename']
            assert name in tag_cache, f"Failure: Image token target key '{name}' missing from output."
            assert len(tag_cache[name]["A"]) == 2, f"Failure: Insufficient tags generated for index slice A."

        print(f"SUCCESS: Successfully processed and verified all {TARGET_SAMPLE_COUNT} image samples.")

    except Exception as e:
        print(f"\nTEST CANCELLED :\n{e}")
        raise e

if __name__ == "__main__":
    run_extractor_test()