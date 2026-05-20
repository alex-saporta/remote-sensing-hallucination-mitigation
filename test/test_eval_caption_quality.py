import sys
import argparse
import torch
from pathlib import Path

current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from eval_merge_quality import eval_merge_quality

def test_eval_merge_quality(merge_json_path, original_json_path):
    output_txt = "test_merge_quality_results.txt"
    
    print(f"Testing eval_merge_quality with:\n  Merged JSON: {merge_json_path}\n  Original JSON: {original_json_path}")
    
    # Execute the evaluation
    results = eval_merge_quality(merge_json_path, original_json_path, output_file=output_txt)
    
    print("Evaluation Results:")
    print(results)
    
    # Handle the text file: read its content, then delete it
    file_path = Path(output_txt)
    if file_path.exists():
        print(f"\n--- Content of {output_txt} ---")
        print(file_path.read_text(encoding="utf-8"))
        print("---------------------------------" + "-" * len(output_txt))
        
        # Delete the text file
        file_path.unlink()
        print(f"(The temporary file {output_txt} was read and then deleted)")
    else:
        print(f"(The temporary file {output_txt} was not generated)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for evaluating merged caption quality.")
    parser.add_argument(
        "--merge_json",
        type=str,
        required=True,
        help="Path to the merged captions JSON file (e.g., merge_captions.json)"
    )
    parser.add_argument(
        "--original_json",
        type=str,
        required=True,
        help="Path to the original SECOND-CC-AUG.json file"
    )
    
    args = parser.parse_args()
    test_eval_merge_quality(args.merge_json, args.original_json)