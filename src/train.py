import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import os
from data_loader import SecondCCDataset 
from model import VLM 
from utils.metrics import Metrics
from transformers import get_cosine_schedule_with_warmup

def train(args):
    # 1. Setup depends on device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device} | Vision: {args.vision_type} | LLM: {args.llm_model} | Caption: {args.caption_method}")

    # 2. Initialize model
    model = VLM(
        fusion_type=args.fusion_type, 
        llm_model=args.llm_model, 
        vision_type=args.vision_type,
        prompt=args.prompt
    )
    model.to(device)
    evaluator = Metrics(device)


    # 3. Data preparation
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    train_dataset = SecondCCDataset(
        json_path= project_root / "data" / "SECOND-CC-AUG.json", 
        data_root=project_root / "data", 
        llm_model=args.llm_model,
        split="train",
        caption_method=args.caption_method,
        merge_captions_path=args.merge_captions_path,
        llm_prompt=args.prompt
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
    )

    val_dataset = SecondCCDataset(
        json_path=project_root / "data" / "SECOND-CC-AUG.json",
        data_root=project_root / "data",
        llm_model=args.llm_model,
        split="val", 
        caption_method=args.caption_method,
        merge_captions_path=args.merge_captions_path,
        llm_prompt=args.prompt
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(model.projector.parameters(), lr=args.lr)

    # 4. Initialize dynamic learning rate
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 5. Prepare evaluator
    experiment_results = {
        "config": vars(args), # save command line args
        "history": [] # store metrics for every epoch
    }
    best_bleu = -1.0

    # 6. Save the experiment setup/arguments
    checkpoint_dir = project_root / "data" / "checkpoints" / f"{args.caption_method}_{args.vision_type}_{args.fusion_type}_{args.llm_model}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "setup_args.json" 
    with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    # 7. Training loop
    for epoch in range(args.epochs):
        # ----- Training Phase ------
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
         
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
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())  
        # ---- Evaluation phase -----
        model.eval()
        all_preds = []
        all_targets = []
        print("Calculating quality metrics...")
        print(f"Generating captions for validation (MARINE active: {args.marine})...")

        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                if args.marine :
                    pred_caption = model.generate_marine(
                        pixel_values_a=val_batch['pixel_values_a'].to(device),
                        pixel_values_b=val_batch['pixel_values_b'].to(device),
                        input_ids_base=val_batch['input_ids_base'].to(device),
                        input_ids_guided=val_batch['input_ids_guided'].to(device),
                        tokenizer=val_dataset.tokenizer,
                        guidance_strength=args.guidance_strength,
                        device=device
                    )
                else :
                    pred_caption = model.generate(
                        val_batch['pixel_values_a'].to(device),
                        val_batch['pixel_values_b'].to(device),
                        tokenizer=val_dataset.tokenizer,
                        device=device
                    )
                all_preds.append(pred_caption)
                all_targets.append([val_batch['raw_caption'][0]])

        # Evaluate the captions
        scores = evaluator.compute_caption_quality(all_preds, all_targets)
        evaluator.reset()
        avg_loss = epoch_loss / len(train_loader)
        epoch_record = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "bleu4": scores.get('bleu4', 0.0),
            "rougeL": scores.get('rougeL_fmeasure', 0.0),
            "cider": scores.get('cider', 0.0),
            "meteor": scores.get('meteor', 0.0)
        }
        experiment_results["history"].append(epoch_record)
        

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"BLEU-4:   {scores.get('bleu4', 0.0):.4f}")
        print(f"ROUGE-L:  {scores.get('rougeL_fmeasure', 0.0):.4f}")
        print(f"CIDEr:    {scores.get('cider', 0.0):.4f}")
        print(f"METEOR:   {scores.get('meteor', 0.0):.4f}")
        
        # 8. Save weights, metrics and captions
        
        # Weights
        save_path = checkpoint_dir / f"projector_epoch_{epoch+1}.pt"
        torch.save(model.projector.state_dict(), save_path)

        if scores['bleu4'] > best_bleu:
            best_bleu = scores['bleu4']
            torch.save(model.projector.state_dict(), checkpoint_dir / "best_projector.pt")
        
        # Metrics file
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(experiment_results, f, indent=4)

        # Generated captions for this epoch
        epoch_predictions = []
        for i in range(len(all_preds)):
            epoch_predictions.append({
                "sample_index": i,
                "ground_truth": all_targets[i][0],
                "generated_caption": all_preds[i]
            })
            
        preds_save_path = checkpoint_dir / f"predictions_epoch_{epoch+1}.json"
        with open(preds_save_path, "w") as f:
            json.dump(epoch_predictions, f, indent=4)