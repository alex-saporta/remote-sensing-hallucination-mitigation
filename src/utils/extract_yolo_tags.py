import json
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLOWorld

# 1. SETUP ALL ABSOLUTE SERVER PATHS
PROJECT_ROOT = Path("/home/apolline/project-2-team-1")
JSON_PATH = PROJECT_ROOT / "data" / "SECOND-CC-AUG.json"  
OUTPUT_PATH = PROJECT_ROOT / "data" / "tags" / "yolo_tags.json"
WEIGHTS_PATH = PROJECT_ROOT / "data" / "weights" / "yolov8s-worldv2.pt"

# 2. INITIALIZE THE OPEN-VOCABULARY YOLO MODEL
print(f"Loading YOLO-World architecture from: {WEIGHTS_PATH}")
model = YOLOWorld(str(WEIGHTS_PATH))

# 3. DEFINE TARGET CLEAN VOCABULARY AND CONTEXTUAL PROMPTS
RAW_VOCABULARY = ["building", "road", "water", "vegetation", "vehicle"]

# Contextual modifiers guide the text encoder to match top-down satellite shapes
CONTEXT_PROMPTS = [
    "aerial view of a building roof",
    "satellite view of a road or street",
    "body of water from above",
    "dense forest trees from above",
    "green vegetation field or grassland",
    "small car or vehicle parked from above",
    "boat or ship in the water from above",
    "bridge over water or road from above",
    "solar panels on a rooftop from above"
]

# Bind the descriptive prompts to the open-vocabulary model layers
model.set_classes(CONTEXT_PROMPTS)

# 4. LOAD THE DATASET INDEX
with open(JSON_PATH, "r") as f:
    full_data = json.load(f)
all_samples = full_data.get("images", [])

tag_cache = {}

def get_image_tags(image_path, conf_threshold=0.05):
    """Runs YOLO-World with descriptive prompts and returns matching raw vocabulary terms."""
    if not image_path.exists():
        return []
        
    try:
        # Run inference with a calibrated confidence threshold for satellite data
        results = model.predict(
            str(image_path), 
            conf=conf_threshold, 
            device="cuda:0", 
            verbose=False
        )
        
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            detected_indices = boxes.cls.cpu().int().tolist()
            # Map the prompt indices directly back to your clean raw vocabulary terms
            detected_labels = [RAW_VOCABULARY[idx] for idx in detected_indices]
            return list(set(detected_labels))
            
    except Exception as e:
        print(f"\n[Warning] Exception encountered on {image_path.name}: {e}")
        
    return []

# 5. EXECUTE THE ACCELERATED IMAGE PROCESSING LOOP
print(f"Beginning YOLO-World tag extraction for {len(all_samples)} image pairs...")

for sample in tqdm(all_samples):
    img_name = sample['filename']
    split = sample.get('split', 'train')
    
    path_a = PROJECT_ROOT / "data" / split / "rgb" / "A" / img_name
    path_b = PROJECT_ROOT / "data" / split / "rgb" / "B" / img_name
    
    if not (path_a.exists() and path_b.exists()):
        continue
        
    tags_a = get_image_tags(path_a)
    tags_b = get_image_tags(path_b)
    
    tag_cache[img_name] = {
        "A": tags_a,
        "B": tags_b
    }

# 6. SAVE PRECOMPUTED CACHE TO DISK
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(tag_cache, f, indent=4)

print(f"\nSuccessfully generated YOLO-World tags! Cache saved at: {OUTPUT_PATH}")