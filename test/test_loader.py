from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import SecondCCDataset


dataset = SecondCCDataset(
    json_path="data/SECOND-CC-AUG.json", 
    data_root="data", 
    split="train"
)
loader = DataLoader(dataset, batch_size=2)

# Get one batch
batch = next(iter(loader))

print(f"Image A shape: {batch['pixel_values_a'].shape}") # Expected: [2, 3, 224, 224]
print(f"Input IDs shape: {batch['input_ids'].shape}")   # Expected: [2, 128]
print(f"Prompt Lengths: {batch['prompt_len']}")         # Expected: tensor([size, size])
print(f"Sample Caption: {batch['raw_caption'][0]}")