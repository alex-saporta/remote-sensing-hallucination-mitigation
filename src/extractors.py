from abc import ABC, abstractmethod
import os
from pathlib import Path
import torch
import open_clip
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor

# Automatically locate the data/extractors target directory relative to src/
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
EXTRACTORS_CACHE_DIR = PROJECT_ROOT / "data" / "extractors"
EXTRACTORS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class BaseTagExtractor(ABC):
    """Abstract base class for all MARINE tag extractors."""
    @abstractmethod
    def __init__(self, device: torch.device):
        self.device = device
        self.processor = None

    @abstractmethod
    def extract_batch_tags(self, pixel_values_batch, vocabulary: list, top_k: int) -> list:
        """Takes a batch of processed image inputs and returns a list of text tags."""
        pass

class SkyClipExtractor(BaseTagExtractor):
    def __init__(self, device: torch.device, model_id: str = "BiliSakura/SkyCLIP-ViT-L-14"):
        super().__init__(device)
        
        # 1. Setup the SkyCLIP VLM
        # Download the model from hugging face and store it in /data
        skyclip_cache_path = EXTRACTORS_CACHE_DIR / "SkyCLIP"
        skyclip_cache_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing SkyCLIP via OpenCLIP...")
        print(f"Targeting storage tracking directory: {skyclip_cache_path}")
        
        # Build standard ViT-L-14 OpenCLIP framework
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", 
            pretrained="openai", 
            cache_dir=str(skyclip_cache_path)
        )
        
        # Download the specialized remote sensing visual weights
        checkpoint_path = hf_hub_download(
            repo_id=model_id, 
            filename="SkyCLIP-ViT-L-14.pth",
            cache_dir=str(skyclip_cache_path)
        )

        # Splice the remote sensing vision tower into the OpenCLIP framework
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sky_state_dict = ckpt.get("state_dict", ckpt)
        visual_state_dict = {k: v for k, v in sky_state_dict.items() if k.startswith("visual.")}
        
        missing, unexpected = self.model.load_state_dict(visual_state_dict, strict=False)
        print("SkyCLIP visual weights successfully loaded into OpenCLIP backbone framework.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=str(skyclip_cache_path)
        )

    def extract_batch_tags(self, pixel_values_batch: torch.Tensor, vocabulary: list, top_k: int) -> list:
        """
        Receives standard pre-computed multi-dimensional PyTorch vision tensors.
        """
        prompt_phrases = [f"a satellite image showing {item}" for item in vocabulary]
        text_tokens = self.tokenizer(prompt_phrases).to(self.device)
        
        with torch.no_grad():
            # Encode text
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Encode image
            image_features = self.model.encode_image(pixel_values_batch.to(self.device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Cross-modal matrix similarity dot product calculation
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            top_k_indices = similarity.topk(top_k).indices.tolist()
            
        return [[vocabulary[idx] for idx in sample_indices] for sample_indices in top_k_indices]


class ExtractorFactory:
    """Factory class to generate the requested extractor model configuration."""
    @staticmethod
    def get_extractor(extractor_type: str, device: torch.device) -> BaseTagExtractor:
        registry = {
            "SkyCLIP": lambda: SkyClipExtractor(device, "BiliSakura/SkyCLIP-ViT-L-14"),
        }
        
        if extractor_type not in registry:
            raise ValueError(f"Unknown extractor: {extractor_type}. Choose from {list(registry.keys())}")
        return registry[extractor_type]()