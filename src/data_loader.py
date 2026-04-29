import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer

class SecondCCDataset(Dataset):
    """
    Dataset for Remote Sensing Change Captioning.
    Links two satellite images (Time 1 and Time 2) to a change description.
    """
    def __init__(self, json_path, data_root, split="train", max_length=128):
        """
        Args:
            json_path: Path to the SECOND-CC-AUG.json file.
            data_root: Path to the 'data' directory.
            split: 'train', 'val', or 'test'.
            max_length: Maximum sentence length for the LLM.
        """
        self.data_root = Path(data_root) / split
        self.max_length = max_length
        
        # 1. Initialize tools
        print(f"Loading processors for {split}...")
        self.vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        
        # 2. Parse the JSON structure
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        # The file provided uses the "images" key as the main list
        all_entries = full_data.get("images", [])
        
        # 3. Filter entries belonging to this split
        self.samples = [s for s in all_entries if s.get("split") == split]
        
        if len(self.samples) == 0:
            print(f"Warning: No samples found for split '{split}'. Check your JSON 'split' keys.")
        else:
            print(f"Loaded {len(self.samples)} samples for {split} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # filename from JSON file of dataset
        img_name = sample['filename']
        
        # Construct paths to RGB images (A = Before, B = After)
        path_a = self.data_root / "rgb" / "A" / img_name
        path_b = self.data_root / "rgb" / "B" / img_name
        
        # Load and ensure 3-channel RGB format
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        
        # 4. Prepare Images (Resize to 224x224 & Normalize)
        # We process both as a list to ensure identical treatment
        vis_inputs = self.vision_processor(
            images=[img_a, img_b], 
            return_tensors="pt"
        )
        
        # 5. Prepare Text
        """
        TODO : how to handle the 5 different captions : use all ? use one ? merge the 5 into 1 ?
        for now : only 1st caption is used 
        """
        caption = sample['sentences'][0]['raw']
        
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image_id": img_name,
            "pixel_values_a": vis_inputs['pixel_values'][0], # Shape: [3, 224, 224]
            "pixel_values_b": vis_inputs['pixel_values'][1], # Shape: [3, 224, 224]
            "input_ids": text_inputs['input_ids'].squeeze(),
            "attention_mask": text_inputs['attention_mask'].squeeze(),
            "raw_caption": caption
        }