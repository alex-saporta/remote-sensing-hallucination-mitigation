import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM


class Projector(nn.Module):
    """
    The projection matrix we need to fine tune, implemented as a simple MLP with GELU activation function 
    """
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.model(x)
    

class VLM(nn.Module):
    """
    The bi-temporal VLM, follows a Llava architecture : Vision Encoder (RemoteCLIP) --> Projector --> LLM (Qwen2.5-3B)
    """
    def __init__(self, fusion_type="concat"):
        super().__init__()
        # Need to handle the fact that our dataset has dual input, 
        # include fusion type to be able to choose between concatenation ('concat') or  subtraction ()'subtract')
        self.fusion_type = fusion_type 
        
        # 1. Vision Encoder (Shared weights)
        self.vision_encoder = CLIPVisionModel.from_pretrained("chendelong/RemoteCLIP")
        
        # 2. Projector (Input dim depends on fusion strategy)
        # If concat: 1024 + 1024 = 2048. If subtract: 1024
        self.proj_in_dim = 2048 if fusion_type == "concat" else 1024
        self.projector = Projector(vision_dim=self.proj_in_dim, llm_dim=2560)
        
        # 3. LLM
        self.language_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B", 
            trust_remote_code=True,
            device_map="auto"
        )

        # Freeze backbones : pre-trained models, we do not want to modify the weights
        for param in self.vision_encoder.parameters(): param.requires_grad = False
        for param in self.language_model.parameters(): param.requires_grad = False

    # TODO : Implement the forward & generate_caption methods of the VLM
