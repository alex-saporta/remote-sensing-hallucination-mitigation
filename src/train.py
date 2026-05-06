import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import os
from data_loader import SecondCCDataset 
from model import VLM 

def train():
    #1. Setup depends on device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Detected and using device: {device}")

    #2. Initialize model
    model = VLM(fusion_type="concat")
    model.to(device)

    #3. Data preparation
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    train_dataset = SecondCCDataset(
        json_path= project_root / "data" / "SECOND-CC-AUG.json", 
        data_root=project_root / "data", 
        split="train"
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
    )

    optimizer = optim.AdamW(model.projector.parameters(), lr=1e-4)

    # 4. Training loop
    model.train()
    epochs = 5
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            # Move all tensors to the detected device
            pixel_values_a = batch['pixel_values_a'].to(device)
            pixel_values_b = batch['pixel_values_b'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prompt_lens = batch['prompt_len'] # This is a list/tensor from DataLoader

            # Mask labels corresponding to the prompt, which we do not need the LLM to guess
            labels = input_ids.clone()
            for i, p_len in enumerate(prompt_lens):
                labels[i, :p_len] = -100
            
            optimizer.zero_grad()
            
            # Forward Pass
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
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save weights 
        save_path = f"checkpoints/projector_epoch_{epoch+1}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.projector.state_dict(), save_path)

if __name__ == "__main__":
    train()