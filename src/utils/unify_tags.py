import json
from pathlib import Path

# 1. SETUP PATHS (Using YOLO as the prototype)
PROJECT_ROOT = Path("/home/apolline/project-2-team-1")
INPUT_JSON = PROJECT_ROOT / "data" / "tags" / "yolo_tags.json" 
OUTPUT_JSON = PROJECT_ROOT / "data" / "tags" / "unified_yolo_prompts.json"

def create_unified_caption(tags_a, tags_b):
    """Computes the delta between two tag lists and generates a guiding string."""
    set_a = set(tags_a)
    set_b = set(tags_b)

    # Calculate logical differences
    appeared = set_b - set_a
    disappeared = set_a - set_b
    unchanged = set_a & set_b

    phrases = []
    
    # You can customize these verbs based on what sounds best for RelaX
    if appeared:
        phrases.append(f"{', '.join(appeared)} appeared")
    if disappeared:
        phrases.append(f"{', '.join(disappeared)} disappeared")
    if unchanged:
        phrases.append(f"{', '.join(unchanged)} remained unchanged")

    # Fallback if both tag lists are completely empty
    if not phrases:
        return "no visible objects detected"

    return ", ".join(phrases)

def main():
    if not INPUT_JSON.exists():
        print(f"Error: Could not find {INPUT_JSON.name}. Check your path.")
        return

    print(f"Loading base tags from {INPUT_JSON.name}...")
    with open(INPUT_JSON, "r") as f:
        tag_data = json.load(f)

    unified_prompts = {}
    
    print("Generating unified change prompts...")
    for img_name, timeframes in tag_data.items():
        tags_a = timeframes.get("A", [])
        tags_b = timeframes.get("B", [])
        
        # Generate the sentence (e.g., "building disappeared, road remained unchanged")
        unified_string = create_unified_caption(tags_a, tags_b)
        
        # Store it mapped to the image name for the LLM loader
        unified_prompts[img_name] = unified_string

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(unified_prompts, f, indent=4)

    print(f"Success! {len(unified_prompts)} unified prompts saved to: {OUTPUT_JSON.name}")
    
    # Print a quick preview of the first 3 items so you can verify the phrasing
    print("\n--- Preview ---")
    for key in list(unified_prompts.keys())[:3]:
        print(f"{key}: {unified_prompts[key]}")

if __name__ == "__main__":
    main()