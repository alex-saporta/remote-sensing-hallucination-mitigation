import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO : write the training loops
def train_projector(model, dataloader, epochs=5, lr=1e-4, device="cuda"):
    """
    Main training loop to fine tune the Projector.
    """