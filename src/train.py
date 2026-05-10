import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import os
from data_loader import SecondCCDataset 
from model import VLM 

def train():

    # 1. Setup Arguments for Modular Training
    parser = argparse.ArgumentParser(description="Bi-temporal VLM training for change captioning")
    parser.add_argument("--vision_type", type=str, default="RemoteCLIP", choices=['RemoteCLIP', 'SkyCLIP', 'standard'])
    parser.add_argument("--llm_model", type=str, default="qwen2.5-3B", choices=['qwen2.5-3B', 'qwen2.5-7B', 'vicuna-7b'])
    parser.add_argument("--caption_method", type=str, default="random", choices=['one', 'two', 'three', 'four', 'five', 'random', 'merge'])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    #2. Setup depends on device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device} | Vision: {args.vision_type} | LLM: {args.llm_model} | Caption: {args.caption_method}")

    #3. Initialize model
    model = VLM(
        fusion_type="concat", 
        llm_model=args.llm_model, 
        vision_type=args.vision_type
    )
    model.to(device)

    #4. Data preparation
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    train_dataset = SecondCCDataset(
        json_path= project_root / "data" / "SECOND-CC-AUG.json", 
        data_root=project_root / "data", 
        llm_model=args.llm_model,
        split="train",
        caption_method=args.caption_method
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
    )

    optimizer = optim.AdamW(model.projector.parameters(), lr=args.lr)

    # 4. Training loop
    model.train()
    
    for epoch in range(args.epochs):
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
        
        # Save weights depending on models used
        checkpoint_dir = Path("checkpoints") / f"{args.vision_type}_{args.llm_model}_{args.caption_method}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = checkpoint_dir / f"projector_epoch_{epoch+1}.pt"
        torch.save(model.projector.state_dict(), save_path)

if __name__ == "__main__":
    train()