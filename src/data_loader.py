import json
import torch
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer, pipeline

def precompute_merge_captions(json_path, output_path, model_id="Qwen/Qwen2.5-1.5B-Instruct", limit=None):
    """
    Pre-computes summarized captions for all samples using a LLM.
    Run this once before training to generate merge_captions.json.
    
    Args:
        json_path: Path to the SECOND-CC-AUG.json file.
        output_path: Path to save the resulting JSON (e.g., 'merge_captions.json').
        model_id: HuggingFace model ID to use for summarization.
        limit: Optional integer to limit the number of samples processed (for testing).
    """
    with open(json_path) as f:
        full_data = json.load(f)

    summarizer = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        max_new_tokens=35,
    )

    results = {}
    all_samples = full_data.get("images", [])
    # Apply limit for testing
    if limit is not None:
        all_samples = all_samples[:limit]
        print(f"  (limited to {limit} samples for testing)")
    
    print(f"Pre-computing merge captions for {len(all_samples)} samples...")

    for i, sample in enumerate(all_samples):
        numbered = "\n".join(
            f"{i+1}. {s['raw'].strip()}" for i, s in enumerate(sample["sentences"])
        )
        prompt = (
            "You are given several descriptions of changes observed between two satellite images. "
            "Write a single, concise sentence (maximum 15 words) that summarizes all changes mentioned.\n\n"
            f"Descriptions:\n{numbered}\n\nSummary:"
        )
        output = summarizer(prompt, do_sample=False, temperature=None, top_p=None)[0]["generated_text"]
        caption = output.split("Summary:")[-1].strip().split("\n")[0].strip()
        results[sample["filename"]] = caption

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_samples)} samples...")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} captions to {output_path}")

class SecondCCDataset(Dataset):
    """
    Dataset for Remote Sensing Change Captioning.
    Links two satellite images (Time 1 and Time 2) to a change description.
    """
    def __init__(self, json_path, data_root, llm_model="qwen2.5-7B", split="train", 
                max_length=128, caption_method="random", merge_captions_path=None, 
                llm_prompt="Describe the changes between the first image and the second image.",
                tags_cache_name="marine_tags.json"):
        """
        Args:
            json_path: Path to the SECOND-CC-AUG.json file.
            data_root: Path to the 'data' directory.
            llm_model: String identifying the LLM (used to select tokenizer).
            split: 'train', 'val', or 'test'.
            max_length: Maximum sentence length for the LLM.
            caption_method: Method for selecting captions ('random', 'one', 'five').
            merge_captions_path: Path to the pre-computed merge captions JSON (required if caption_method='merge').
            llm_prompt : Prompt given to the LLM to perform the task
            tags_cache_name : Name of the file containing the tags for the images
        """
        self.data_root = Path(data_root) / split
        self.max_length = max_length
        self.caption_method = caption_method
        self.llm_prompt = llm_prompt
        self.split = split
        
        # 1. Initialize tools
        llm_registry = {
            "qwen2.5-7B": "Qwen/Qwen2.5-7B",
            "vicuna-7b":  "lmsys/vicuna-7b-v1.5",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "llama2-7b":  "meta-llama/Llama-2-7b-hf",
        }
        
        if llm_model not in llm_registry:
            raise ValueError(f"LLM {llm_model} not in registry. Use: {list(llm_registry.keys())}")
        
        tokenizer_id = llm_registry[llm_model]
        print(f"Loading processor and tokenizer for: {llm_model} ({split})")

        self.vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

        # Ensure padding token exists 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load pre-computed merge captions if needed
        if caption_method == "merge":
            if merge_captions_path is None:
                raise ValueError(
                    "caption_method='merge' requires a pre-computed captions file. "
                    "Please provide 'merge_captions_path', or run precompute_merge_captions() first."
                )
            with open(merge_captions_path) as f:
                self.merge_captions = json.load(f)
            print(f"Loaded pre-computed merge captions from: {merge_captions_path}")
        else:
            self.merge_captions = None

        # Load the pre-computed tags
        project_root = Path(json_path).resolve().parent.parent
        tags_json_path = project_root / "data" / tags_cache_name
        if tags_json_path.exists():
            print(f"[{split.upper()}] Loading visual guidance tags cache: {tags_json_path}")
            with open(tags_json_path, "r") as f:
                self.tag_cache = json.load(f)
        else:
            print(f"WARNING: Guidance cache missing at {tags_json_path}. Falling back to default baseline configurations.")
            self.tag_cache = {}
                                                               
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
        
        # 1. Get filename from JSON file of dataset
        img_name = sample['filename']
        
        # 2. Construct paths to RGB images (A = Before, B = After)
        path_a = self.data_root / "rgb" / "A" / img_name
        path_b = self.data_root / "rgb" / "B" / img_name
        
        # 3. Load images
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        
        # 4. Prepare images : we process both as a list to ensure identical treatment
        vis_inputs = self.vision_processor(images=[img_a, img_b], return_tensors="pt")
        
        sentences = sample['sentences']
       
        # Map string numbers to their corresponding list index
        index_map = {"one": 0, "two": 1, "three": 2, "four": 3, "five": 4}
        if self.caption_method in index_map:
            target_idx = index_map[self.caption_method]
            safe_idx = min(target_idx, len(sentences) - 1)
            caption = sentences[safe_idx]['raw']

        # Randomly select one of the available captions    
        elif self.caption_method == "random":
            caption_idx = random.randint(0, len(sentences) - 1)
            caption = sentences[caption_idx]['raw']
        
        # Select the caption from the merged file
        elif self.caption_method == "merge":
            caption = self.merge_captions.get(img_name)
            if caption is None:
                raise KeyError(
                    f"No pre-computed merge caption found for '{img_name}'. "
                    "Re-run precompute_merge_captions() to regenerate the file."
                )
            
        else:
            valid_methods = list(index_map.keys()) + ["random", "merge"]
            raise ValueError(f"Unknown caption_method: '{self.caption_method}'. Use one of: {valid_methods}")
        
        # 5. Handle visual + text tokens fusion
        # Combine with prompt and end of sentence token
        full_text = f"{self.llm_prompt}{caption}{self.tokenizer.eos_token}"
        
        text_inputs = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Calculate prompt length for label masking later
        prompt_len = len(self.tokenizer(self.llm_prompt, add_special_tokens=True)["input_ids"])
        
        # 6. MARINE 
        # Fetch pre-computed tags
        if img_name in self.tag_cache:
            tags_a_str = ", ".join(self.tag_cache[img_name]["A"]) if self.tag_cache[img_name]["A"] else "no features"
            tags_b_str = ", ".join(self.tag_cache[img_name]["B"]) if self.tag_cache[img_name]["B"] else "no features"
        else:
            tags_a_str, tags_b_str = "unknown features", "unknown features"

        # Construct the contrastive companion prompt string
        guided_prompt_str = f"Context: Time A shows {tags_a_str}. Time B shows {tags_b_str}. {self.llm_prompt}"
        
        # Tokenize prompt token inputs cleanly for validation generation trackers
        input_ids_base = self.tokenizer(self.llm_prompt, return_tensors="pt")["input_ids"].squeeze(0)
        input_ids_guided = self.tokenizer(guided_prompt_str, return_tensors="pt")["input_ids"].squeeze(0)

        return {
            "image_id": img_name,
            "pixel_values_a": vis_inputs['pixel_values'][0],
            "pixel_values_b": vis_inputs['pixel_values'][1],
            "input_ids": text_inputs['input_ids'].squeeze(),
            "attention_mask": text_inputs['attention_mask'].squeeze(),
            "prompt_len": prompt_len,
            "raw_caption": caption,
            "input_ids_base": input_ids_base,
            "input_ids_guided": input_ids_guided
        }
       