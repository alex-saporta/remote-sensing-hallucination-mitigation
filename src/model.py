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
    def __init__(self, fusion_type="concat", llm_model="qwen2.5-7B", vision_type="standard", 
                 prompt="Describe the change between these two satellite images in one sentence."):
        super().__init__()
        self.fusion_type = fusion_type
        self.prompt = prompt

        # 1. Vision Encoder : ViT-L-14 is the default; swap here to try other OpenCLIP variants
        model_name = "ViT-L-14"
        print("Loading Vision Encoder architecture...")
        self.vision_encoder = self._load_vision_encoder(vision_type, model_name)
        vision_out_dim = getattr(self.vision_encoder, "output_dim", 1024)

        # 2. Projector : concatenation fusion doubles the input dimension to the projector
        self.proj_in_dim = vision_out_dim * 2 if fusion_type == "concat" else vision_out_dim

        # 3. LLM
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

        print(f"Loading LLM: {hf_model_id}")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        # Freeze vision encoder and LLM, only the projector is trained
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False


    # Vision-encoder factory
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
    
    def _apply_patch_to_openclip(self, vision_model):
        """Applies the 256-token monkey patch to any OpenCLIP visual model."""
        def patched_forward(x):
            x = vision_model.conv1(x)  
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  
            x = torch.cat([vision_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
            x = x + vision_model.positional_embedding.to(x.dtype)
            x = vision_model.ln_pre(x)
            x = x.permute(1, 0, 2)  
            x = vision_model.transformer(x)
            x = x.permute(1, 0, 2)  
            x = vision_model.ln_post(x)
            return x
            
        vision_model.forward = patched_forward
        vision_model.output_dim = 1024 # Hardcoded ViT-L dimension
        return vision_model


    def _load_openclip(self, model_name: str):
        """Load standard OpenCLIP ViT weights (OpenAI pretrained)."""
        """clip_model = open_clip.create_model(model_name, pretrained="openai")
        return clip_model.visual"""
        clip_model = open_clip.create_model(model_name, pretrained="openai")
        
        return self._apply_patch_to_openclip(clip_model.visual)

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
            return self._apply_patch_to_openclip(clip_model.visual)

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
                global_pool=''
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
            return self._apply_patch_to_openclip(clip_model.visual)

        except Exception as e:
            print(f"GeoRSCLIP weights unavailable ({e}). Falling back to standard OpenCLIP.")
            return self._load_openclip(model_name)

    def forward(self, pixel_values_a, pixel_values_b, input_ids, attention_mask=None, labels=None):
        
        device = next(self.vision_encoder.parameters()).device
        
        # 1. Encode
        pixel_values_a = pixel_values_a.to(device)
        pixel_values_b = pixel_values_b.to(device)

        with torch.no_grad():
            feat_a = self.vision_encoder(pixel_values_a)[:, 1:, :] 
            feat_b = self.vision_encoder(pixel_values_b)[:, 1:, :]

        # 2. Fusion
        if self.fusion_type == "concat":
            fused_feat = torch.cat([feat_a, feat_b], dim=-1)
        else:
            fused_feat = feat_b - feat_a

        # 3. Project to LLM space (Batch, 1, llm_dim)
        visual_embeds = self.projector(fused_feat)
        visual_embeds = visual_embeds.to(self.language_model.dtype)

        # 4. Text embeddings (Batch, Seq, llm_dim)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 5. Concatenate
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 6. Mask Handling
        if attention_mask is not None:
            visual_mask = torch.ones((attention_mask.size(0), 256), device=attention_mask.device)
            full_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            full_mask = None

        if labels is not None:
            ignore_node = torch.full((labels.size(0), 256), -100, device=labels.device)
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
            self.eval()
            # 1. Encode and fusion images
            with torch.no_grad():
                feat_a = self.vision_encoder(pixel_values_a.to(device))[:, 1:, :]
                feat_b = self.vision_encoder(pixel_values_b.to(device))[:, 1:, :]

                if self.fusion_type == "concat":
                    fused_feat = torch.cat([feat_a, feat_b], dim=-1)
                else:
                    fused_feat = feat_b - feat_a

                # 2. Project to LLM space
                visual_embeds = self.projector(fused_feat)
                visual_embeds = visual_embeds.to(self.language_model.dtype)

                # 3. Prepare the prompt
                input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids.to(device)
                text_embeds = self.language_model.get_input_embeddings()(input_ids)

                # 4. Concatenate visual prefix with prompt
                inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
                inputs_embeds = inputs_embeds.to(self.language_model.dtype)

                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

                # 5. Generate tokens
                output_ids = self.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=15,
                    do_sample=False, # greedy decoding for consistent metrics
                    num_beams=1, # speed up validation
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

                # 6. Decode back to string : only decode the new tokens (=the generated caption)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                return caption.strip()
    
    @torch.no_grad()
    def generate_marine(self, pixel_values_a, pixel_values_b, input_ids_base, input_ids_guided, tokenizer, guidance_strength: float = 0.7, max_new_tokens=50, device="cuda"):
        """
        Inference method to generate captions using Image Grounded Guidance (MARINE).
        """
        self.eval()
        # 1. Encode and fuse images
        feat_a = self.vision_encoder(pixel_values_a.to(device))[:, 1:, :]
        feat_b = self.vision_encoder(pixel_values_b.to(device))[:, 1:, :]

        if self.fusion_type == "concat":
            fused_feat = torch.cat([feat_a, feat_b], dim=-1)
        else:
            fused_feat = feat_b - feat_a

        visual_embeds = self.projector(fused_feat).to(self.language_model.dtype)

        # 2. Clean padding from the dataloader tensors
        # Dataloader pads to max_length (128), which breaks inference. We strip it here.
        pad_id = tokenizer.pad_token_id
        
        # Squeeze in case dataloader batched it, then remove padding, then add batch dim back
        uncond_ids = input_ids_base.squeeze()
        uncond_ids = uncond_ids[uncond_ids != pad_id].unsqueeze(0)
        
        cond_ids = input_ids_guided.squeeze()
        cond_ids = cond_ids[cond_ids != pad_id].unsqueeze(0)

        # 3. Get text embeddings
        uncond_text_embeds = self.language_model.get_input_embeddings()(uncond_ids)
        cond_text_embeds = self.language_model.get_input_embeddings()(cond_ids)

        uncond_inputs = torch.cat([visual_embeds, uncond_text_embeds], dim=1)
        cond_inputs = torch.cat([visual_embeds, cond_text_embeds], dim=1)

        # 4. Initialize KV Caches for fast generation
        uncond_past = None
        cond_past = None
        uncond_curr_input = uncond_inputs
        cond_curr_input = cond_inputs
        
        generated_ids = []

        # 5. The MARINE Autoregressive Loop
        for t in range(max_new_tokens):
            uncond_outputs = self.language_model(
                inputs_embeds=uncond_curr_input if t == 0 else None,
                input_ids=uncond_curr_input if t > 0 else None,
                past_key_values=uncond_past,
                use_cache=True
            )
            uncond_logits = uncond_outputs.logits[:, -1, :]
            uncond_past = uncond_outputs.past_key_values
            
            cond_outputs = self.language_model(
                inputs_embeds=cond_curr_input if t == 0 else None,
                input_ids=cond_curr_input if t > 0 else None,
                past_key_values=cond_past,
                use_cache=True
            )
            cond_logits = cond_outputs.logits[:, -1, :]
            cond_past = cond_outputs.past_key_values
            
            # The CFG Formula
            final_logits = (guidance_strength * cond_logits) + ((1.0 - guidance_strength) * uncond_logits)
            
            next_token = torch.argmax(final_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token.item())
            
            # Stop if the model generates EOS or Qwen's <|im_end|> token
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if next_token.item() == tokenizer.eos_token_id or next_token.item() == im_end_id:
                break
                
            uncond_curr_input = next_token
            cond_curr_input = next_token

        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()