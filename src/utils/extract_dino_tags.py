import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 1. SETUP ALL ABSOLUTE SERVER PATHS
PROJECT_ROOT = Path("/home/apolline/project-2-team-1")
JSON_PATH = PROJECT_ROOT / "data" / "SECOND-CC-AUG.json"  
OUTPUT_PATH = PROJECT_ROOT / "data" / "tags" / "dino_tags.json"

# 2. CHOOSE DEVICE AND LOAD NATIVE HUGGING FACE MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Targeting compute device: {device}")

model_id = "IDEA-Research/grounding-dino-base"
print(f"Loading {model_id} from Hugging Face Hub...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# 3. DEFINE YOUR GEOGRAPHICAL TARGET VOCABULARY
VOCABULARY = ["building", "road", "water", "forest", "vegetation", "vehicle", "ship", "bridge", "solar panel"]
# Grounding DINO expects lower-case items separated by a period and a space
TEXT_PROMPT = " . ".join(VOCABULARY) + " ."

# 4. LOAD THE DATASET INDEX
with open(JSON_PATH, "r") as f:
    full_data = json.load(f)
all_samples = full_data.get("images", [])

tag_cache = {}

def get_image_tags(image_path, text_prompt, conf_threshold=0.35):
    """Loads a single image, runs grounding inference, and returns unique text tags."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return [] # Skip gracefully if file corrupted

    # Prepare inputs natively
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Format target size as a tensor of (height, width)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    
    # Post-process to filter predictions by confidence thresholds
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=conf_threshold,
        text_threshold=0.25,
        target_sizes=target_sizes
    )
    
    # Extract textual labels strings directly from the predictions dictionary
    detected_labels = results[0].get("labels", [])
    return list(set(detected_labels))

# 5. EXECUTE THE IMAGE PROCESSING LOOP
print(f"Beginning zero-shot tag extraction for {len(all_samples)} image pairs...")
model.eval()

for sample in tqdm(all_samples):
    img_name = sample['filename']
    split = sample.get('split', 'train')
    
    path_a = PROJECT_ROOT / "data" / split / "rgb" / "A" / img_name
    path_b = PROJECT_ROOT / "data" / split / "rgb" / "B" / img_name
    
    # Skip processing pair if either time frame image is missing on disk
    if not (path_a.exists() and path_b.exists()):
        continue
        
    tags_a = get_image_tags(path_a, TEXT_PROMPT)
    tags_b = get_image_tags(path_b, TEXT_PROMPT)
    
    tag_cache[img_name] = {
        "A": tags_a,
        "B": tags_b
    }

# 6. SAVE CACHE LAYER TO DISK
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(tag_cache, f, indent=4)

print(f"\nSuccessfully generated pre-computed tags! Cache saved at: {OUTPUT_PATH}")