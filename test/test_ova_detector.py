import os
import sys
import json
import re
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose

# 1. SETUP REPOSITORY PATH COUPLING
PROJECT_ROOT = Path("/home/apolline/project-2-team-1")
ITC_OVAD_PATH = PROJECT_ROOT / "ITC-OVAD"

if str(ITC_OVAD_PATH) not in sys.path:
    sys.path.insert(0, str(ITC_OVAD_PATH))

from src.core import YAMLConfig

# 2. FILE AND DEVICE RESOLUTION
JSON_PATH = PROJECT_ROOT / "data" / "SECOND-CC-AUG.json"  
WEIGHTS_PATH = PROJECT_ROOT / "data" / "weights" / "itc_ovad_lae1m.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print(f"RUNNING ITC-OVAD LOCAL DIAGNOSTIC TEST")
print(f"Targeting compute device: {device.upper()}")
print("=" * 60)

# 3. AUTOMATIC CONFIGURATION SELECTION
try:
    demo_content = (ITC_OVAD_PATH / "demo.py").read_text()
    yml_match = re.search(r"configs/[^\s'\"]+\.yml", demo_content)
    CONFIG_PATH = str(ITC_OVAD_PATH / yml_match.group(0)) if yml_match else str(list(ITC_OVAD_PATH.glob("configs/**/*.yml"))[0])
except Exception:
    CONFIG_PATH = str(ITC_OVAD_PATH / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml")

print(f"[1/4] Loading configuration template: {Path(CONFIG_PATH).name}")
cfg = YAMLConfig(CONFIG_PATH, resume=str(WEIGHTS_PATH))

# 4. MODEL INITIALIZATION & EXPLICIT WEIGHT LOADING
print("[2/4] Assembling network graph and injecting weights...")
model = cfg.model

# Extract the parameter layers cleanly
ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu")
if "model" in ckpt:
    state_dict = ckpt["model"]
elif "ema" in ckpt:
    state_dict = ckpt["ema"]["module"]
else:
    state_dict = ckpt

# --- NEW: Filter out keys with shape mismatches ---
model_state = model.state_dict()
mismatched_keys = []
for k, v in state_dict.items():
    if k in model_state and v.shape != model_state[k].shape:
        # Track the keys that don't fit the local config sizes
        mismatched_keys.append(k)

for k in mismatched_keys:
    del state_dict[k]
    
if mismatched_keys:
    print(f"    [Info] Skipped {len(mismatched_keys)} mismatched layers (e.g., {mismatched_keys[0]})")
# ------------------------------------------------

# Now load the cleaned dictionary
model.load_state_dict(state_dict, strict=False)

# Wrap deployment optimizations
deploy_model = model.deploy() if hasattr(model, 'deploy') else model
deploy_model.to(device)
deploy_model.eval()

postprocessor = cfg.postprocessor.deploy() if hasattr(cfg.postprocessor, 'deploy') else cfg.postprocessor

transform = Compose([
    Resize((800, 800)),
    ToTensor(),
])

# 5. TEST CONTEXT AND VOCABULARY
VOCABULARY = ["building", "road", "water", "forest", "vegetation", "vehicle", "ship", "bridge", "solar panel"]

with open(JSON_PATH, "r") as f:
    all_samples = json.load(f).get("images", [])

@torch.no_grad()
def test_inference(image_path, conf_threshold=0.15):
    if not image_path.exists():
        return f"File missing at: {image_path.name}"
    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        orig_sizes = torch.tensor([[w, h]], dtype=torch.float32).to(device)
        data = transform(image)[None].to(device)
        
        outputs = deploy_model(data, class_text=[VOCABULARY])
        labels, boxes, scores = postprocessor(outputs, orig_sizes)
        
        img_labels = labels[0]
        img_scores = scores[0]
        
        # Monitor confidence scores after weights are loaded
        if len(img_scores) > 0:
            max_score = img_scores.max().item()
            print(f"    [Diagnostic] Highest confidence score found: {max_score:.4f}")
            
        keep = img_scores > conf_threshold
        detected_indices = img_labels[keep].cpu().int().tolist()
        
        return list(set([VOCABULARY[idx] for idx in detected_indices if idx < len(VOCABULARY)]))
    except Exception as e:
        return f"Inference Error: {str(e)}"

# 6. RUN EXACTLY 2 VALID SAMPLES
print("\n[3/4] Searching for test targets on disk...")
tested_count = 0

for sample in all_samples:
    if tested_count >= 2:
        break
        
    img_name = sample['filename']
    split = sample.get('split', 'train')
    path_a = PROJECT_ROOT / "data" / split / "rgb" / "A" / img_name
    path_b = PROJECT_ROOT / "data" / split / "rgb" / "B" / img_name
    
    if not (path_a.exists() and path_b.exists()):
        continue
        
    tested_count += 1
    print(f"\n--- Test Sample #{tested_count}: {img_name} ---")
    
    print("Processing Image A...")
    tags_a = test_inference(path_a)
    print(f"  Result A: {tags_a}")
    
    print("Processing Image B...")
    tags_b = test_inference(path_b)
    print(f"  Result B: {tags_b}")

print("\n" + "=" * 60)
print("[4/4] DIAGNOSTIC PIPELINE COMPLETE")
print("=" * 60)