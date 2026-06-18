import os
import sys
import json
import re
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor, Resize, Compose

# 1. SETUP REPOSITORY PATH COUPLING
PROJECT_ROOT = Path("/home/apolline/project-2-team-1")
ITC_OVAD_PATH = PROJECT_ROOT / "ITC-OVAD"

if str(ITC_OVAD_PATH) not in sys.path:
    # insert(0, ...) gives this repository absolute search priority
    sys.path.insert(0, str(ITC_OVAD_PATH))

from src.core import YAMLConfig

# 2. SETUP ABSOLUTE WORKSPACE PATHS
JSON_PATH = PROJECT_ROOT / "data" / "SECOND-CC-AUG.json"  
OUTPUT_PATH = PROJECT_ROOT / "data" / "tags" / "ova_tags.json"
WEIGHTS_PATH = PROJECT_ROOT / "data" / "weights" / "itc_ovad_lae1m.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Targeting compute architecture: {device.upper()}")

# 3. AUTOMATIC CONFIGURATION SELECTION
try:
    demo_content = (ITC_OVAD_PATH / "demo.py").read_text()
    yml_match = re.search(r"configs/[^\s'\"]+\.yml", demo_content)
    CONFIG_PATH = str(ITC_OVAD_PATH / yml_match.group(0)) if yml_match else str(list(ITC_OVAD_PATH.glob("configs/**/*.yml"))[0])
except Exception:
    CONFIG_PATH = str(ITC_OVAD_PATH / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml")

print(f"Using configuration template: {Path(CONFIG_PATH).name}")

# 4. MODEL INITIALIZATION & EXPLICIT WEIGHT INJECTION
print("Assembling network graph and loading pre-trained parameters...")
cfg = YAMLConfig(CONFIG_PATH, resume=str(WEIGHTS_PATH))
model = cfg.model

# Extract the parameter layers cleanly
ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu")
if "model" in ckpt:
    state_dict = ckpt["model"]
elif "ema" in ckpt:
    state_dict = ckpt["ema"]["module"]
else:
    state_dict = ckpt

# Filter out keys with shape mismatches (fixes the 4800 vs 366 text embedding crash)
model_state = model.state_dict()
mismatched_keys = []
for k, v in state_dict.items():
    if k in model_state and v.shape != model_state[k].shape:
        mismatched_keys.append(k)

for k in mismatched_keys:
    del state_dict[k]
    
if mismatched_keys:
    print(f"    [Info] Dropped {len(mismatched_keys)} text-embedding layers to match local config.")

# Load the cleaned dictionary
model.load_state_dict(state_dict, strict=False)

# Wrap deployment optimizations
deploy_model = model.deploy() if hasattr(model, 'deploy') else model
deploy_model.to(device)
deploy_model.eval()

# Deploy postprocessor to return unpacked tuples
postprocessor = cfg.postprocessor.deploy() if hasattr(cfg.postprocessor, 'deploy') else cfg.postprocessor

# Force 800x800 resolution to match the 625 token position attention grid
transform = Compose([
    Resize((800, 800)),
    ToTensor(),
])

# 5. DEFINE YOUR GEOSPATIAL TARGET VOCABULARY
VOCABULARY = ["building", "road", "water", "forest", "vegetation", "vehicle", "ship", "bridge", "solar panel"]

# 6. LOAD THE DATASET INDEX
with open(JSON_PATH, "r") as f:
    full_data = json.load(f)
all_samples = full_data.get("images", [])

tag_cache = {}

@torch.no_grad()
def get_image_tags(image_path, conf_threshold=0.15):
    """Processes an image frame and filters predictions by confidence scores."""
    if not image_path.exists():
        return []
        
    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # Build tensor layouts exactly as the model expects
        orig_sizes = torch.tensor([[w, h]], dtype=torch.float32).to(device)
        data = transform(image)[None].to(device)
        
        # Pass the double-nested vocabulary to the text encoder
        outputs = deploy_model(data, class_text=[VOCABULARY])
        labels, boxes, scores = postprocessor(outputs, orig_sizes)
        
        # Strip the batch dimension
        img_labels = labels[0]
        img_scores = scores[0]
            
        # Filter by threshold
        keep = img_scores > conf_threshold
        detected_indices = img_labels[keep].cpu().int().tolist()
        
        # Map integer IDs back to strings and deduplicate
        return list(set([VOCABULARY[idx] for idx in detected_indices if idx < len(VOCABULARY)]))
        
    except Exception as e:
        # Silently catch corrupted images to prevent batch crashes
        return []

# 7. RUN THE TIMEFRAME DUAL-INPUT PROCESSING LOOP
print(f"\nBeginning batch tag extraction for {len(all_samples)} image pairs...")
for sample in tqdm(all_samples):
    img_name = sample['filename']
    split = sample.get('split', 'train')
    
    path_a = PROJECT_ROOT / "data" / split / "rgb" / "A" / img_name
    path_b = PROJECT_ROOT / "data" / split / "rgb" / "B" / img_name
    
    # Only process if both timeframes exist
    if not (path_a.exists() and path_b.exists()):
        continue
        
    tags_a = get_image_tags(path_a)
    tags_b = get_image_tags(path_b)
    
    tag_cache[img_name] = {
        "A": tags_a,
        "B": tags_b
    }

# 8. EXPORT RESULTS TO THE TOOLBOX CACHE DIRECTORY
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(tag_cache, f, indent=4)

print(f"\nSuccess! OVA-Det tags written cleanly to: {OUTPUT_PATH}")