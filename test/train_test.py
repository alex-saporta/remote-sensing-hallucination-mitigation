import sys
import torch
import torch.optim as optim
from pathlib import Path
from data_loader import SecondCCDataset
from model import VLM
from torch.utils.data import DataLoader

current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

def test_training_loop():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing training on: {device}")

    # 2. Initialize Model
    model = VLM(fusion_type="concat").to(device)
    model.train() # Set to training mode

    # 3. Load a sample of the data
    dataset = SecondCCDataset(
        json_path="data/SECOND-CC-AUG.json", 
        data_root="data", 
        split="train"
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # 4. Setup Optimizer 
    optimizer = optim.AdamW(model.projector.parameters(), lr=1e-4)

    print("\nStarting Overfitting Test (Goal: Loss should decrease)...")
    
    # Move batch to device
    pixel_values_a = batch['pixel_values_a'].to(device)
    pixel_values_b = batch['pixel_values_b'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    prompt_lens = batch['prompt_len']

    # 5. Iterative training on one batch
    initial_loss = None
    
    for i in range(10): 
        optimizer.zero_grad()
        
        # Dynamic label masking
        labels = input_ids.clone()
        for j, p_len in enumerate(prompt_lens):
            labels[j, :p_len] = -100
        
        outputs = model(
            pixel_values_a=pixel_values_a,
            pixel_values_b=pixel_values_b,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
        
        print(f"Iteration {i+1}/10 | Loss: {loss.item():.4f}")

    # 6. Check loss
    if loss.item() < initial_loss:
        print("\n" + "="*30)
        print("SUCCESS: Loss decreased")
        print(f"Loss dropped from {initial_loss:.4f} to {loss.item():.4f}")
        print("="*30)
    else:
        print("\nERROR: Loss did not decrease")

if __name__ == "__main__":
    test_training_loop()