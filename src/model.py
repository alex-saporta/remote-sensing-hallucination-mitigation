import torch
import open_clip
import timm
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoConfig

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
    def __init__(self, fusion_type="concat", llm_model="qwen2.5-3B", vision_type="standard"):
        super().__init__()
        self.fusion_type = fusion_type

        # ViT-L-14 is the default backbone; swap here to try other OpenCLIP variants
        model_name = "ViT-L-14"

        # 1. Vision Encoder
        print("Loading Vision Encoder architecture via OpenCLIP")
        self.vision_encoder = self._load_vision_encoder(vision_type, model_name)
        vision_out_dim = getattr(self.vision_encoder, "output_dim", 768)

        # 2. Projector
        # Concatenation fusion doubles the input dimension to the projector
        self.proj_in_dim = vision_out_dim * 2 if fusion_type == "concat" else vision_out_dim

        llm_registry = {
            "qwen2.5-7B": "Qwen/Qwen2.5-7B",
            "vicuna-7b":  "lmsys/vicuna-7b-v1.5",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "llama2-7b":  "meta-llama/Llama-2-7b-hf",
        }
        if llm_model not in llm_registry:
            raise ValueError(
                f"Unrecognized LLM model: '{llm_model}'. "
                f"Available options: {list(llm_registry.keys())}"
            )
        hf_model_id = llm_registry[llm_model]

        # Fetch hidden size from the model config instead of hard-coding it
        print(f"Fetching config for {hf_model_id} to determine hidden dimension …")
        llm_dim = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True).hidden_size

        self.projector = Projector(vision_dim=self.proj_in_dim, llm_dim=llm_dim)

        # 3. LLM
        print(f"Loading LLM: {hf_model_id}")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        # Freeze vision encoder and LLM — only the projector is trained
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False


    # Vision-encoder factory (extracted from __init__ for readability)

    def _load_vision_encoder(self, vision_type: str, model_name: str):
        """Instantiate and return the requested vision encoder."""

        if vision_type == "standard":
            return self._load_openclip(model_name)

        if vision_type == "RemoteCLIP":
            return self._load_remoteclip(model_name)

        if vision_type == "SkyCLIP":
            return self._load_skyclip(model_name)

        if vision_type == "GeoRSCLIP":
            return self._load_georsclip(model_name)

        raise ValueError(
            f"Unrecognized vision_type: '{vision_type}'. "
            f"Available options: ['RemoteCLIP', 'SkyCLIP', 'GeoRSCLIP', 'standard']"
        )


    def _load_openclip(self, model_name: str):
        """Load standard OpenCLIP ViT weights (OpenAI pretrained)."""
        clip_model = open_clip.create_model(model_name, pretrained="openai")
        return clip_model.visual


    def _load_remoteclip(self, model_name: str):
        """
        Attempt to load RemoteCLIP weights from the Hub.
        Falls back to standard OpenCLIP on failure.
        """
        try:
            print(f"Attempting to load RemoteCLIP weights via OpenCLIP …")
            checkpoint_path = hf_hub_download(
                "chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt"
            )
            clip_model = open_clip.create_model(model_name)
            clip_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print("Successfully loaded RemoteCLIP weights.")
            return clip_model.visual

        except Exception as e:
            print(f"RemoteCLIP weights unavailable ({e}). Falling back to standard OpenCLIP.")
            return self._load_openclip(model_name)


    def _load_skyclip(self, model_name: str):
        """
        Attempt to load SkyCLIP weights from the Hub via timm.
        Falls back to standard OpenCLIP on failure.
        """
        try:
            print("Attempting to load SkyCLIP weights via HuggingFace Hub …")
            checkpoint_path = hf_hub_download(
                "BiliSakura/SkyCLIP-ViT-L-14", "SkyCLIP-ViT-L-14.pth"
            )

            # Build a timm ViT-L/14 backbone (no classification head; returns CLS token, dim=1024)
            timm_model = timm.create_model(
                "vit_large_patch14_224",
                pretrained=False,
                num_classes=0,
            )

            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = {
                k.replace("visual.", ""): v
                for k, v in ckpt.get("state_dict", ckpt).items()
            }

            missing, unexpected = timm_model.load_state_dict(state_dict, strict=False)
            print(f"SkyCLIP checkpoint loaded — missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

            # Expose output_dim so the rest of __init__ can query it uniformly
            timm_model.output_dim = timm_model.num_features
            print("Successfully loaded SkyCLIP weights.")
            return timm_model

        except Exception as e:
            print(f"SkyCLIP weights unavailable ({e}). Falling back to standard OpenCLIP.")
            return self._load_openclip(model_name)
        
    def _load_georsclip(self, model_name: str):
        """
        Attempt to load GeoRSCLIP weights from the Hub.
        Falls back to standard OpenCLIP on failure.
        """
        try:
            print(f"Attempting to load GeoRSCLIP weights via OpenCLIP …")
            checkpoint_path = hf_hub_download(
                "Zilun/GeoRSCLIP", f"RS5M_{model_name}.pt",
                subfolder="ckpt"
            )
            clip_model = open_clip.create_model(model_name)
            clip_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print("Successfully loaded GeoRSCLIP weights.")
            return clip_model.visual

        except Exception as e:
            print(f"GeoRSCLIP weights unavailable ({e}). Falling back to standard OpenCLIP.")
            return self._load_openclip(model_name)

    def forward(self, pixel_values_a, pixel_values_b, input_ids, attention_mask=None, labels=None):
        
        device = next(self.vision_encoder.parameters()).device
        
        # 1. Encode
        pixel_values_a = pixel_values_a.to(device)
        pixel_values_b = pixel_values_b.to(device)

        with torch.no_grad():
            feat_a = self.vision_encoder(pixel_values_a) 
            feat_b = self.vision_encoder(pixel_values_b)

        # 2. Fusion
        if self.fusion_type == "concat":
            fused_feat = torch.cat([feat_a, feat_b], dim=-1)
        else:
            fused_feat = feat_b - feat_a

        # 3. Project to LLM space (Batch, 1, llm_dim)
        visual_embeds = self.projector(fused_feat).unsqueeze(1)
        visual_embeds = visual_embeds.to(self.language_model.dtype)

        # 4. Text embeddings (Batch, Seq, llm_dim)
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

    def generate(self, pixel_values_a, pixel_values_b, tokenizer, max_new_tokens=50, device="cuda"):
            """
            Inference method to generate a caption for a given image pair.
            """
            # 1. Encode and fusion images
            with torch.no_grad():
                feat_a = self.vision_encoder(pixel_values_a.to(device))
                feat_b = self.vision_encoder(pixel_values_b.to(device))

                if self.fusion_type == "concat":
                    fused_feat = torch.cat([feat_a, feat_b], dim=-1)
                else:
                    fused_feat = feat_b - feat_a

                # 2. Project to LLM space
                visual_embeds = self.projector(fused_feat).unsqueeze(1) # [Batch, 1, llm_dim]
                visual_embeds = visual_embeds.to(self.language_model.dtype)

                # 3. Prepare the Prompt
                prompt = "Describe the change between these two satellite images: "
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                text_embeds = self.language_model.get_input_embeddings()(input_ids)

                # 4. Concatenate visual prefix with prompt
                inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
                inputs_embeds = inputs_embeds.to(self.language_model.dtype)

                # 5. Generate tokens
                output_ids = self.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, # greedy decoding for consistent metrics
                    num_beams=1, # speed up validation
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

                # 6. Decode back to string : only decode the new tokens (=the generated caption)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                return caption.strip()