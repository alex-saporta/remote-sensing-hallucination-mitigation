import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPModel, AutoModelForCausalLM, AutoConfig

class Projector(nn.Module):
    def __init__(self, vision_dim, llm_dim=2048):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.model(x)

class VLM(nn.Module):
    def __init__(self, fusion_type="concat", llm_model="qwen2.5-3B"):
        super().__init__()
        self.fusion_type = fusion_type 
        
        # 1. Vision Encoder
        print("Loading Vision Encoder architecture")
        # Load the base ViT-L/14 architecture
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        try:
            print("Attempting to load RemoteCLIP weights")
            # We load the full CLIPModel to get the weights, then transfer them
            remote_clip = CLIPModel.from_pretrained("chendelong/RemoteCLIP")
            self.vision_encoder.load_state_dict(remote_clip.vision_model.state_dict())
            print("Successfully loaded RemoteCLIP weights")
        except Exception as e:
            print(f"Note: RemoteCLIP weights not applied ({e}), using standard CLIP instead")

        # 2. Projector logic
        self.proj_in_dim = 2048 if fusion_type == "concat" else 1024

        llm_registry = {
            "qwen2.5-3B": "Qwen/Qwen2.5-3B",
            "qwen2.5-7B": "Qwen/Qwen2.5-7B",
            "vicuna-7b": "lmsys/vicuna-7b-v1.5"
        }

        if llm_model not in llm_registry:
            raise ValueError(f"Unrecognized LLM model: '{llm_model}'. Available options: {list(llm_registry.keys())}")

        hf_model_id = llm_registry[llm_model]

        # Dynamically fetching the LLM's hidden size
        print(f"Fetching config for {hf_model_id} to get hidden dimensions...")
        config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        llm_dim = config.hidden_size
        
        self.projector = Projector(vision_dim=self.proj_in_dim, llm_dim=llm_dim)
        
        # 3. LLM
        print(f"Loading LLM: {hf_model_id}")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            hf_model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )

        # Freeze model weights : only want to train Projector
        for param in self.vision_encoder.parameters(): param.requires_grad = False
        for param in self.language_model.parameters(): param.requires_grad = False

    def forward(self, pixel_values_a, pixel_values_b, input_ids, attention_mask=None, labels=None):
        
        # 1. Encode
        pixel_values_a = pixel_values_a.to(self.vision_encoder.device)
        pixel_values_b = pixel_values_b.to(self.vision_encoder.device)

        with torch.no_grad():
            feat_a = self.vision_encoder(pixel_values_a).last_hidden_state[:, 0, :] 
            feat_b = self.vision_encoder(pixel_values_b).last_hidden_state[:, 0, :]

        # 2. Fusion
        if self.fusion_type == "concat":
            fused_feat = torch.cat([feat_a, feat_b], dim=-1)
        else:
            fused_feat = feat_b - feat_a

        # 3. Project to LLM space (Batch, 1, 2048)
        visual_embeds = self.projector(fused_feat).unsqueeze(1)
        visual_embeds = visual_embeds.to(self.language_model.dtype)

        # 4. Text embeddings (Batch, Seq, 2048)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 5. Concatenate
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 6. Mask Handling
        if attention_mask is not None:
            visual_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            full_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            full_mask = None

        if labels is not None:
            ignore_node = torch.full((labels.size(0), 1), -100, device=labels.device)
            full_labels = torch.cat([ignore_node, labels], dim=1)
        else:
            full_labels = None

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            return_dict=True
        )