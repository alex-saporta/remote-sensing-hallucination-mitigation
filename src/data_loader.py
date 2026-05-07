import json
import torch
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer

class SecondCCDataset(Dataset):
    """
    Dataset for Remote Sensing Change Captioning.
    Links two satellite images (Time 1 and Time 2) to a change description.
    """
    def __init__(self, json_path, data_root, split="train", max_length=128, caption_method="random"):
        """
        Args:
            json_path: Path to the SECOND-CC-AUG.json file.
            data_root: Path to the 'data' directory.
            split: 'train', 'val', or 'test'.
            max_length: Maximum sentence length for the LLM.
            caption_method: Method for selecting captions ('random', 'one', 'five').
        """
        self.data_root = Path(data_root) / split
        self.max_length = max_length
        self.caption_method = caption_method
        
        # 1. Initialize tools
        print(f"Loading processors for {split}")
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
            print(f"No samples found for split '{split}'")
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
        
        # Load images
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        
        # 4. Prepare Images (resize to 224x224 & normalize) : we process both as a list to ensure identical treatment

        vis_inputs = self.vision_processor(images=[img_a, img_b], return_tensors="pt")
        
        # 5. Prepare Text
        prompt = "Describe the change between these two satellite images: "
        
        
        sentences = sample['sentences']
        
        # Map string numbers to their corresponding list index
        index_map = {
            "one": 0, 
            "two": 1, 
            "three": 2, 
            "four": 3, 
            "five": 4
        }
        
        if self.caption_method in index_map:
            # Get the requested index
            target_idx = index_map[self.caption_method]
            # Safety check: ensure we don't go out of bounds if a sample has fewer than 5 captions
            safe_idx = min(target_idx, len(sentences) - 1)
            caption = sentences[safe_idx]['raw']
            
        elif self.caption_method == "random":
            # Randomly select one of the available captions
            caption_idx = random.randint(0, len(sentences) - 1)
            caption = sentences[caption_idx]['raw']
            
        elif self.caption_method == "merge":
            # Add a period between each sentence, and a final period at the very end
            caption = ". ".join([s['raw'].strip() for s in sentences]) + "."
            
        else:
            valid_methods = list(index_map.keys()) + ["random", "merge"]
            raise ValueError(f"Unknown caption_method: '{self.caption_method}'. Use one of: {valid_methods}")
        # -------------------------
        
        # Combine with Prompt and end of sentence token
        full_text = f"{prompt}{caption}{self.tokenizer.eos_token}"
        
        text_inputs = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Calculate prompt length for label masking later
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        return {
            "image_id": img_name,
            "pixel_values_a": vis_inputs['pixel_values'][0],
            "pixel_values_b": vis_inputs['pixel_values'][1],
            "input_ids": text_inputs['input_ids'].squeeze(),
            "attention_mask": text_inputs['attention_mask'].squeeze(),
            "prompt_len": prompt_len,
            "raw_caption": caption
        }
       