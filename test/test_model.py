import sys
import torch
from pathlib import Path


current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from model import VLM

def test_vlm_forward():
    #  1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 2. Initialize Model
    print("Initializing model (this may take a minute)...")
    model = VLM(fusion_type="concat").to(device)
    model.eval() # Set to evaluation mode

    # 3. Create mock data : batch of 2, 3 channels, 224x224 pixels
    dummy_img_a = torch.randn(2, 3, 224, 224).to(device)
    dummy_img_b = torch.randn(2, 3, 224, 224).to(device)
    # Batch of 2, 10 tokens long
    dummy_input_ids = torch.randint(0, 1000, (2, 10)).to(device)
    
    print("Running forward pass")
    try:
        # 5. Run Forward Pass
        with torch.no_grad():
            outputs = model(
                pixel_values_a=dummy_img_a,
                pixel_values_b=dummy_img_b,
                input_ids=dummy_input_ids
            )
        
        # 6. Verify Outputs
        logits = outputs.logits
        print("-" * 30)
        print("SUCCESS!")
        print(f"Logits Shape: {logits.shape}") 
        # Expected: [2, 11, vocab_size] -> (1 visual token + 10 text tokens = 11)
        print("-" * 30)

    except Exception as e:
        print("-" * 30)
        print("Forward pass failed")
        print(f"Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    test_vlm_forward()