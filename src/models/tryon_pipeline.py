"""
TryOn Pipeline — MeshVTON: IDM-VTON backbone + ControlNet3D.

Uses IDM-VTON's ORIGINAL hacked UNet classes (unet_hacked_tryon,
unet_hacked_garmnet) for proper weight loading and forward pass.
The only trainable module is ControlNet3D.

Architecture (from architecture_diagram.png):
    3D garment mesh → SMPL-X drape → PyTorch3D render → ControlNet3D ──┐
    Person image → agnostic mask + pose + DensePose → VAE encode ───── TryonNet
    Garment image → frozen GarmentNet → IP-Adapter → cross-attention ──┘
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.controlnet_3d import ControlNet3D


class TryOnPipeline(nn.Module):
    """
    MeshVTON Pipeline: IDM-VTON backbone (frozen) + ControlNet3D (trainable).

    Components:
        - tryon_net: IDM-VTON's hacked TryonNet UNet (frozen)
        - garment_net: IDM-VTON's hacked GarmentNet UNet encoder (frozen)
        - vae: SDXL AutoencoderKL (frozen)
        - image_encoder: CLIP Vision encoder for IP-Adapter (frozen)
        - image_proj_model: IP-Adapter Resampler (frozen)
        - noise_scheduler: DDPM/DDIM scheduler
        - controlnet_3d: ControlNet3D — NOVEL (trainable)
    """

    def __init__(
        self,
        tryon_net: nn.Module,
        garment_net: nn.Module,
        vae: nn.Module,
        noise_scheduler,
        controlnet_3d: Optional[ControlNet3D] = None,
        image_encoder: Optional[nn.Module] = None,
        image_proj_model: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        text_encoder_2: Optional[nn.Module] = None,
        tokenizer=None,
        tokenizer_2=None,
        scaling_factor: float = 0.13025,
    ):
        super().__init__()
        self.tryon_net = tryon_net
        self.garment_net = garment_net
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.controlnet_3d = controlnet_3d or ControlNet3D()
        self.image_encoder = image_encoder
        self.image_proj_model = image_proj_model
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scaling_factor = scaling_factor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_id: str = "yisol/IDM-VTON",
        controlnet_3d_channels: int = 9,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> "TryOnPipeline":
        """
        Load IDM-VTON using its ORIGINAL hacked UNet classes.
        """
        try:
            # IDM-VTON's own UNet classes
            from src.idm_vton.unet_hacked_tryon import UNet2DConditionModel as TryonUNet
            from src.idm_vton.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet

            from diffusers import AutoencoderKL, DDIMScheduler
            from transformers import (
                CLIPVisionModelWithProjection,
                CLIPTextModel,
                CLIPTextModelWithProjection,
                AutoTokenizer,
            )
            from huggingface_hub import hf_hub_download
            import json

            print(f"Loading IDM-VTON from: {pretrained_model_id}")

            # ============================================================
            # 1. TryonNet — IDM-VTON's hacked UNet with IP-Adapter support
            # ============================================================
            print("  Loading TryonNet...")
            tryon_net = TryonUNet.from_pretrained(
                pretrained_model_id,
                subfolder="unet",
                torch_dtype=torch_dtype,
            ).to(device)
            print(f"  ✅ TryonNet: {sum(p.numel() for p in tryon_net.parameters()):,} params")

            # ============================================================
            # 2. GarmentNet — IDM-VTON's hacked garment encoder
            # ============================================================
            print("  Loading GarmentNet...")
            # GarmentNet loads from SDXL base (not IDM-VTON's unet)
            # and has addition_embed_type removed
            try:
                garment_net = GarmentUNet.from_pretrained(
                    pretrained_model_id,
                    subfolder="unet_encoder",
                    torch_dtype=torch_dtype,
                ).to(device)
            except Exception as e:
                print(f"  GarmentNet from unet_encoder failed: {e}")
                print("  Trying alternative: load from SDXL base...")
                # Fallback: load from SDXL base and remove addition_embed_type
                garment_net = GarmentUNet.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    subfolder="unet",
                    torch_dtype=torch_dtype,
                ).to(device)

            garment_net.config.addition_embed_type = None
            garment_net.config["addition_embed_type"] = None
            print(f"  ✅ GarmentNet: {sum(p.numel() for p in garment_net.parameters()):,} params")

            # ============================================================
            # 3. VAE
            # ============================================================
            print("  Loading VAE...")
            try:
                vae = AutoencoderKL.from_pretrained(
                    pretrained_model_id,
                    subfolder="vae",
                    torch_dtype=torch_dtype,
                ).to(device)
            except Exception:
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch_dtype,
                ).to(device)
            print(f"  ✅ VAE: {sum(p.numel() for p in vae.parameters()):,} params")

            # ============================================================
            # 4. Image Encoder (CLIP Vision) + IP-Adapter Resampler
            # ============================================================
            print("  Loading Image Encoder...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                pretrained_model_id,
                subfolder="image_encoder",
                torch_dtype=torch_dtype,
            ).to(device)
            print(f"  ✅ Image Encoder: {sum(p.numel() for p in image_encoder.parameters()):,} params")

            # IP-Adapter Resampler
            from ip_adapter.ip_adapter import Resampler
            image_proj_model = Resampler(
                dim=1280,
                depth=4,
                dim_head=64,
                heads=20,
                num_queries=16,
                embedding_dim=image_encoder.config.hidden_size,
                output_dim=tryon_net.config.cross_attention_dim,
                ff_mult=4,
            ).to(device=device, dtype=torch_dtype)
            print(f"  ✅ IP-Adapter Resampler: {sum(p.numel() for p in image_proj_model.parameters()):,} params")

            # ============================================================
            # 5. Text Encoders + Tokenizers
            # ============================================================
            print("  Loading Text Encoders...")
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_id, subfolder="tokenizer"
            )
            tokenizer_2 = AutoTokenizer.from_pretrained(
                pretrained_model_id, subfolder="tokenizer_2"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_id,
                subfolder="text_encoder",
                torch_dtype=torch_dtype,
            ).to(device)
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_id,
                subfolder="text_encoder_2",
                torch_dtype=torch_dtype,
            ).to(device)
            print(f"  ✅ Text Encoders loaded")

            # ============================================================
            # 6. Scheduler
            # ============================================================
            noise_scheduler = DDIMScheduler.from_pretrained(
                pretrained_model_id, subfolder="scheduler"
            )

            # ============================================================
            # 7. Freeze ALL backbone components
            # ============================================================
            for module in [tryon_net, garment_net, vae, image_encoder,
                           image_proj_model, text_encoder, text_encoder_2]:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False

            total_frozen = sum(
                sum(p.numel() for p in m.parameters())
                for m in [tryon_net, garment_net, vae, image_encoder,
                          image_proj_model, text_encoder, text_encoder_2]
                if m is not None
            )
            print(f"  ✅ All backbone frozen: {total_frozen:,} params")

        except Exception as e:
            print(f"FATAL: Could not load IDM-VTON: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to placeholder modules")
            tryon_net, garment_net, vae, noise_scheduler = cls._create_placeholders(device)
            image_encoder = None
            image_proj_model = None
            text_encoder = None
            text_encoder_2 = None
            tokenizer = None
            tokenizer_2 = None

        # ============================================================
        # 8. ControlNet3D — the ONLY trainable component
        # ============================================================
        controlnet_3d = ControlNet3D(
            conditioning_channels=controlnet_3d_channels,
        ).to(device)

        pipeline = cls(
            tryon_net=tryon_net,
            garment_net=garment_net,
            vae=vae,
            noise_scheduler=noise_scheduler,
            controlnet_3d=controlnet_3d,
            image_encoder=image_encoder,
            image_proj_model=image_proj_model,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scaling_factor=0.13025,
        )

        trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pipeline.parameters())
        frozen = total - trainable
        print(f"\nModel Summary:")
        print(f"  Total parameters:         {total:,}")
        print(f"  Trainable (ControlNet3D): {trainable:,}")
        print(f"  Frozen (IDM-VTON):        {frozen:,}")
        print(f"  Trainable ratio:          {100 * trainable / max(total, 1):.1f}%")

        return pipeline

    @classmethod
    def from_config(cls, config: dict) -> "TryOnPipeline":
        """Build pipeline from config dict."""
        pretrained_id = config.get("pretrained_idm_vton")
        if pretrained_id:
            return cls.from_pretrained(
                pretrained_model_id=pretrained_id,
                controlnet_3d_channels=config.get("controlnet_3d_channels", 9),
            )
        # Fallback for testing
        tryon_net, garment_net, vae, scheduler = cls._create_placeholders("cpu")
        controlnet_3d = ControlNet3D(
            conditioning_channels=config.get("controlnet_3d_channels", 9),
        )
        return cls(tryon_net, garment_net, vae, scheduler, controlnet_3d)

    @staticmethod
    def _create_placeholders(device: str):
        """Create lightweight placeholder modules for dev/testing."""
        from src.models.noise_scheduler import create_noise_scheduler

        tryon_net = _PlaceholderUNet(in_ch=13, out_ch=4).to(device)
        garment_net = _PlaceholderUNet(in_ch=4, out_ch=4).to(device)
        for p in garment_net.parameters():
            p.requires_grad = False

        vae = _PlaceholderVAE().to(device)
        for p in vae.parameters():
            p.requires_grad = False

        scheduler = create_noise_scheduler("ddpm", num_train_timesteps=1000)
        return tryon_net, garment_net, vae, scheduler

    # ================================================================
    # FORWARD PASS — Training
    # ================================================================

    def forward(
        self,
        person_image: torch.Tensor,
        garment_image: torch.Tensor,
        agnostic_mask: torch.Tensor,
        pose_map: torch.Tensor,
        densepose_map: Optional[torch.Tensor] = None,
        conditioning_3d: Optional[torch.Tensor] = None,
        garment_text: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            person_image: (B, 3, H, W) target person image
            garment_image: (B, 3, H, W) garment image
            agnostic_mask: (B, 3, H, W) clothing-agnostic person
            pose_map: (B, C, H, W) pose/densepose map
            densepose_map: (B, C, H, W) optional DensePose UV map
            conditioning_3d: (B, 9, H, W) optional 3D conditioning
            garment_text: optional garment description text
        """
        device = person_image.device
        dtype = person_image.dtype
        b = person_image.shape[0]

        # --- Encode images to latent space (frozen VAE) ---
        with torch.no_grad():
            person_latent = self._vae_encode(person_image)
            garment_latent = self._vae_encode(garment_image)
            agnostic_latent = self._vae_encode(agnostic_mask)

            # Create inpainting mask in latent space
            # IDM-VTON uses 13-channel input: noisy(4) + agnostic(4) + mask(1) + warped_cloth(4)
            inpaint_mask = torch.ones(b, 1, person_latent.shape[2], person_latent.shape[3],
                                      device=device, dtype=dtype)

        # --- Sample noise and timesteps ---
        noise = torch.randn_like(person_latent)
        timesteps = torch.randint(
            0, self._num_train_timesteps(),
            (b,), device=device, dtype=torch.long,
        )
        noisy_latent = self.noise_scheduler.add_noise(person_latent, noise, timesteps)

        # --- Get garment features (frozen) ---
        with torch.no_grad():
            # CLIP image features for IP-Adapter
            ip_features = self._get_ip_adapter_features(garment_image)

            # GarmentNet encoder features for self-attention fusion
            garment_down_features, garment_ref_features = self._get_garment_features(
                garment_latent, timesteps
            )

            # Text embeddings
            prompt_embeds, pooled_prompt_embeds = self._encode_text(garment_text, b)

        # --- 3D ControlNet conditioning (TRAINABLE) ---
        controlnet_residuals = None
        if conditioning_3d is not None:
            controlnet_residuals = self.controlnet_3d(conditioning_3d, timesteps)

        # --- Build TryonNet input (13 channels) ---
        # noisy_latent(4) + agnostic_latent(4) + mask(1) + garment_latent(4) = 13ch
        model_input = torch.cat([
            noisy_latent,       # 4ch - noised target
            agnostic_latent,    # 4ch - clothing-agnostic person
            inpaint_mask,       # 1ch - inpainting mask
            garment_latent,     # 4ch - garment (warped cloth in original)
        ], dim=1)  # (B, 13, H/8, W/8)

        # --- SDXL added conditions ---
        add_time_ids = torch.zeros(b, 6, device=device, dtype=dtype)
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # --- Predict noise via TryonNet ---
        pred_noise = self._tryon_forward(
            model_input=model_input,
            timesteps=timesteps,
            encoder_hidden_states=prompt_embeds,
            ip_features=ip_features,
            garment_down_features=garment_down_features,
            garment_ref_features=garment_ref_features,
            added_cond_kwargs=added_cond_kwargs,
            controlnet_residuals=controlnet_residuals,
        )

        # Ensure shape match
        if pred_noise.shape != noise.shape:
            pred_noise = pred_noise[:, :noise.shape[1]]

        loss = F.mse_loss(pred_noise, noise)

        return {"loss": loss, "pred_noise": pred_noise, "target_noise": noise}

    # ================================================================
    # GENERATE — Inference
    # ================================================================

    @torch.no_grad()
    def generate(
        self,
        garment_image: torch.Tensor,
        agnostic_mask: torch.Tensor,
        pose_map: torch.Tensor,
        densepose_map: Optional[torch.Tensor] = None,
        conditioning_3d: Optional[torch.Tensor] = None,
        garment_text: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate try-on image."""
        device = garment_image.device
        dtype = garment_image.dtype
        b = garment_image.shape[0]

        garment_latent = self._vae_encode(garment_image)
        agnostic_latent = self._vae_encode(agnostic_mask)

        # Garment features
        ip_features = self._get_ip_adapter_features(garment_image)
        garment_down_features, garment_ref_features = self._get_garment_features(
            garment_latent,
            torch.zeros(b, device=device, dtype=torch.long)
        )
        prompt_embeds, pooled_prompt_embeds = self._encode_text(garment_text, b)

        # Inpainting mask
        h, w = agnostic_latent.shape[2], agnostic_latent.shape[3]
        inpaint_mask = torch.ones(b, 1, h, w, device=device, dtype=dtype)

        # Start from noise
        latent = torch.randn(b, 4, h, w, generator=generator, device=device, dtype=dtype)

        # DDIM loop
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(b).to(device)

            # 3D ControlNet
            controlnet_residuals = None
            if conditioning_3d is not None:
                controlnet_residuals = self.controlnet_3d(conditioning_3d, t_batch)

            # 13-channel input
            model_input = torch.cat([
                latent, agnostic_latent, inpaint_mask, garment_latent
            ], dim=1)

            add_time_ids = torch.zeros(b, 6, device=device, dtype=dtype)
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            noise_pred = self._tryon_forward(
                model_input=model_input,
                timesteps=t_batch,
                encoder_hidden_states=prompt_embeds,
                ip_features=ip_features,
                garment_down_features=garment_down_features,
                garment_ref_features=garment_ref_features,
                added_cond_kwargs=added_cond_kwargs,
                controlnet_residuals=controlnet_residuals,
            )

            step_out = self.noise_scheduler.step(noise_pred, t, latent)
            latent = step_out.prev_sample

        # Decode
        image = self.vae.decode(latent / self.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # ================================================================
    # INTERNAL HELPERS
    # ================================================================

    def _vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent via VAE."""
        if hasattr(self.vae, "encode") and hasattr(self.vae, "config"):
            dist = self.vae.encode(image).latent_dist
            return dist.sample() * self.scaling_factor
        elif hasattr(self.vae, "encode"):
            enc = self.vae.encode(image)
            if isinstance(enc, dict):
                return enc["latent"]
            return enc * self.scaling_factor
        return image

    def _get_ip_adapter_features(self, garment_image: torch.Tensor) -> torch.Tensor:
        """Get IP-Adapter features from CLIP vision encoder."""
        if self.image_encoder is None or self.image_proj_model is None:
            b = garment_image.shape[0]
            # Return dummy features matching expected dim
            cross_attn_dim = getattr(
                getattr(self.tryon_net, 'config', None),
                'cross_attention_dim', 2048
            )
            return torch.zeros(b, 16, cross_attn_dim,
                             device=garment_image.device, dtype=garment_image.dtype)

        # CLIP vision encoding
        clip_output = self.image_encoder(garment_image)
        clip_features = clip_output.hidden_states[-2]  # penultimate layer

        # Project through IP-Adapter Resampler
        ip_tokens = self.image_proj_model(clip_features)
        return ip_tokens

    def _get_garment_features(self, garment_latent, timesteps):
        """
        Get GarmentNet encoder features for self-attention fusion.

        IDM-VTON's GarmentNet returns:
          - down_features: list of intermediate encoder outputs
          - reference_features: list of reference features for self-attention
        """
        if isinstance(self.garment_net, _PlaceholderUNet):
            b = garment_latent.shape[0]
            device = garment_latent.device
            dtype = garment_latent.dtype
            # Return empty lists — no garment feature fusion
            return [], []

        b = garment_latent.shape[0]
        device = garment_latent.device
        dtype = garment_latent.dtype

        # GarmentNet expects encoder_hidden_states
        # Use zeros since it doesn't use text conditioning
        dummy_text = torch.zeros(b, 77, 2048, device=device, dtype=dtype)

        try:
            down_features, reference_features = self.garment_net(
                garment_latent,
                timesteps,
                dummy_text,
                return_dict=False,
            )
            return down_features, reference_features
        except Exception as e:
            print(f"GarmentNet forward failed: {e}")
            return [], []

    def _encode_text(self, text, batch_size):
        """Encode text prompt for SDXL conditioning."""
        device = next(self.tryon_net.parameters()).device
        dtype = next(self.tryon_net.parameters()).dtype

        if self.text_encoder is None or self.tokenizer is None:
            # Return dummy embeddings
            prompt_embeds = torch.zeros(batch_size, 77, 2048, device=device, dtype=dtype)
            pooled_embeds = torch.zeros(batch_size, 1280, device=device, dtype=dtype)
            return prompt_embeds, pooled_embeds

        if text is None:
            text = ""

        # Tokenize
        if isinstance(text, str):
            text = [text] * batch_size

        # CLIP text encoder 1
        tokens_1 = self.tokenizer(
            text, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        text_embeds_1 = self.text_encoder(tokens_1).last_hidden_state

        # CLIP text encoder 2
        tokens_2 = self.tokenizer_2(
            text, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        text_output_2 = self.text_encoder_2(tokens_2)
        text_embeds_2 = text_output_2.last_hidden_state
        pooled_prompt_embeds = text_output_2.text_embeds

        # Concatenate text encoder outputs (SDXL style)
        prompt_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)

        return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)

    def _tryon_forward(
        self,
        model_input,
        timesteps,
        encoder_hidden_states,
        ip_features,
        garment_down_features,
        garment_ref_features,
        added_cond_kwargs,
        controlnet_residuals=None,
    ):
        """
        Run TryonNet forward pass.

        IDM-VTON's hacked TryonNet accepts additional args:
          - down_block_additional_residuals: ControlNet residuals
          - mid_block_additional_residual: ControlNet mid block
          - garment_features: from GarmentNet for self-attention fusion
        """
        try:
            kwargs = {
                "added_cond_kwargs": added_cond_kwargs,
            }

            # ControlNet3D residuals
            if controlnet_residuals is not None:
                kwargs["down_block_additional_residuals"] = controlnet_residuals[:-1]
                kwargs["mid_block_additional_residual"] = controlnet_residuals[-1]

            # GarmentNet features for self-attention (IDM-VTON specific)
            if garment_down_features:
                kwargs["garment_features"] = garment_ref_features

            out = self.tryon_net(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
                **kwargs,
            )

            # IDM-VTON returns tuple
            if isinstance(out, tuple):
                return out[0]
            if hasattr(out, "sample"):
                return out.sample
            return out

        except Exception as e:
            # Fallback: try without extra kwargs
            print(f"TryonNet forward with full kwargs failed: {e}")
            print("Trying minimal forward...")
            try:
                out = self.tryon_net(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                if isinstance(out, tuple):
                    return out[0]
                return out
            except Exception as e2:
                print(f"Minimal forward also failed: {e2}")
                raise

    def _num_train_timesteps(self) -> int:
        if hasattr(self.noise_scheduler, "config"):
            return self.noise_scheduler.config.num_train_timesteps
        return 1000

    # ================================================================
    # FREEZE / TRAINABLE PARAMS
    # ================================================================

    def freeze_backbone(self):
        """Freeze all pre-trained components, keep only ControlNet3D trainable."""
        for name, module in self.named_children():
            if name == "controlnet_3d":
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Backbone frozen — Trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / max(total, 1):.1f}%)")

    def get_trainable_params(self):
        """Return only ControlNet3D parameters for optimizer."""
        return [p for p in self.controlnet_3d.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Placeholders for dev / CPU testing
# ---------------------------------------------------------------------------

class _PlaceholderUNet(nn.Module):
    def __init__(self, in_ch: int = 13, out_ch: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.config = type('Config', (), {
            'in_channels': in_ch,
            'cross_attention_dim': 2048,
        })()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

    def forward(self, x, timesteps=None, encoder_hidden_states=None,
                return_dict=False, **kwargs):
        if x.shape[1] != self.in_ch:
            if x.shape[1] > self.in_ch:
                x = x[:, :self.in_ch]
            else:
                pad = torch.zeros(
                    x.shape[0], self.in_ch - x.shape[1],
                    x.shape[2], x.shape[3],
                    device=x.device, dtype=x.dtype,
                )
                x = torch.cat([x, pad], dim=1)
        out = self.net(x)
        if return_dict:
            return type('Output', (), {'sample': out})()
        return (out,)


class _PlaceholderVAE(nn.Module):
    def __init__(self, scale: float = 0.13025):
        super().__init__()
        self.scale = scale
        self.config = type('Config', (), {'scaling_factor': scale})()
        self.enc = nn.Conv2d(3, 4, 4, stride=4, padding=0)
        self.dec = nn.ConvTranspose2d(4, 3, 4, stride=4, padding=0)

    def encode(self, x):
        latent = self.enc(x) * self.scale
        return type('Output', (), {
            'latent_dist': type('Dist', (), {'sample': lambda: latent})()
        })()

    def decode(self, z):
        return type('Output', (), {'sample': self.dec(z / self.scale)})()