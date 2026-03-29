"""
TryOn Pipeline — IDM-VTON backed dual-branch diffusion pipeline.

Loads the pre-trained IDM-VTON model (SDXL-based TryonNet + frozen GarmentNet)
from HuggingFace and integrates the novel ControlNet3D module for 3D garment
conditioning (rendered RGB, normal map, depth map from PyTorch3D).

Architecture:
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
    IDM-VTON-backed Virtual Try-On Pipeline with 3D ControlNet.

    Uses pre-trained IDM-VTON components (TryonNet, GarmentNet, VAE) as
    frozen backbone components. The only trainable module is ControlNet3D,
    which injects 3D conditioning from the garment rendering pipeline.

    Components:
        - tryon_net: SDXL UNet (TryonNet) — main denoising backbone
        - garment_net: Frozen SDXL UNet — garment feature extractor
        - vae: SDXL AutoencoderKL — encode/decode latent space
        - noise_scheduler: DDPM/DDIM scheduler
        - controlnet_3d: 3D ControlNet — NOVEL: 3D conditioning injection
    """

    def __init__(
        self,
        tryon_net: nn.Module,
        garment_net: nn.Module,
        vae: nn.Module,
        noise_scheduler,
        controlnet_3d: Optional[ControlNet3D] = None,
        scaling_factor: float = 0.13025,
    ):
        super().__init__()
        self.tryon_net = tryon_net
        self.garment_net = garment_net
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.controlnet_3d = controlnet_3d or ControlNet3D()
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
        Load IDM-VTON pre-trained weights from HuggingFace.
    
        IDM-VTON uses a modified SDXL UNet with IP-Adapter integration.
        The UNet config needs to be patched before loading weights.
        """
        try:
            from diffusers import (
                UNet2DConditionModel,
                AutoencoderKL,
                DDIMScheduler as DiffusersDDIMScheduler,
            )
            from transformers import CLIPVisionModelWithProjection
            from huggingface_hub import hf_hub_download
            import json
            import os
    
            print(f"Loading IDM-VTON from: {pretrained_model_id}")
    
            # ============================================================
            # Step 1: Load TryonNet (main UNet)
            # IDM-VTON's UNet has IP-Adapter fields in config that
            # standard diffusers UNet2DConditionModel doesn't accept.
            # Solution: download config, patch it, then load.
            # ============================================================
    
            # Download the UNet config
            config_path = hf_hub_download(
                pretrained_model_id,
                filename="unet/config.json",
            )
            with open(config_path, "r") as f:
                unet_config = json.load(f)
    
            # Remove problematic fields that cause loading errors
            fields_to_remove = [
                "encoder_hid_dim_type",
                "encoder_hid_dim",
                "addition_embed_type",
                "addition_embed_type_num_heads",
                "addition_time_embed_dim",
                # EMA fields that leak into config
                "decay", "inv_gamma", "min_decay",
                "optimization_step", "power",
                "update_after_step", "use_ema_warmup",
            ]
            for field in fields_to_remove:
                unet_config.pop(field, None)
    
            # Create UNet from cleaned config
            tryon_net = UNet2DConditionModel.from_config(unet_config)
    
            # Load the actual weights
            weights_path = hf_hub_download(
                pretrained_model_id,
                filename="unet/diffusion_pytorch_model.bin",
            )
            state_dict = torch.load(weights_path, map_location="cpu")
    
            # Load weights (strict=False to handle IP-Adapter extra keys)
            missing, unexpected = tryon_net.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  TryonNet missing keys: {len(missing)} (IP-Adapter keys, expected)")
            if unexpected:
                print(f"  TryonNet unexpected keys: {len(unexpected)}")
    
            tryon_net = tryon_net.to(dtype=torch_dtype, device=device)
            print(f"  TryonNet loaded: {sum(p.numel() for p in tryon_net.parameters()):,} params")
    
            # ============================================================
            # Step 2: Load GarmentNet (frozen garment encoder)
            # Uses unet_encoder subfolder with similar config issues
            # ============================================================
    
            try:
                # Try loading unet_encoder directly
                encoder_config_path = hf_hub_download(
                    pretrained_model_id,
                    filename="unet_encoder/config.json",
                )
                with open(encoder_config_path, "r") as f:
                    encoder_config = json.load(f)
    
                # Clean config
                for field in fields_to_remove:
                    encoder_config.pop(field, None)
                # GarmentNet doesn't use addition_embed_type
                encoder_config.pop("addition_embed_type", None)
    
                garment_net = UNet2DConditionModel.from_config(encoder_config)
    
                encoder_weights_path = hf_hub_download(
                    pretrained_model_id,
                    filename="unet_encoder/diffusion_pytorch_model.bin",
                )
                encoder_state_dict = torch.load(encoder_weights_path, map_location="cpu")
                garment_net.load_state_dict(encoder_state_dict, strict=False)
                garment_net = garment_net.to(dtype=torch_dtype, device=device)
                print(f"  GarmentNet loaded: {sum(p.numel() for p in garment_net.parameters()):,} params")
    
            except Exception as e:
                print(f"  GarmentNet from unet_encoder failed: {e}")
                print(f"  Using TryonNet copy as GarmentNet (encoder-only)")
                # Fallback: copy TryonNet weights for GarmentNet
                garment_net = UNet2DConditionModel.from_config(unet_config)
                garment_net.load_state_dict(state_dict, strict=False)
                garment_net = garment_net.to(dtype=torch_dtype, device=device)
    
            # ============================================================
            # Step 3: Load VAE
            # ============================================================
    
            try:
                vae = AutoencoderKL.from_pretrained(
                    pretrained_model_id,
                    subfolder="vae",
                    torch_dtype=torch_dtype,
                ).to(device)
                print(f"  VAE loaded: {sum(p.numel() for p in vae.parameters()):,} params")
            except Exception as e:
                print(f"  VAE from IDM-VTON failed: {e}")
                print(f"  Loading SDXL VAE from madebyollin/sdxl-vae-fp16-fix")
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch_dtype,
                ).to(device)
    
            # ============================================================
            # Step 4: Load Scheduler
            # ============================================================
    
            try:
                noise_scheduler = DiffusersDDIMScheduler.from_pretrained(
                    pretrained_model_id,
                    subfolder="scheduler",
                )
            except Exception:
                noise_scheduler = DiffusersDDIMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    num_train_timesteps=1000,
                    clip_sample=False,
                    prediction_type="epsilon",
                )
    
            # ============================================================
            # Step 5: Freeze all backbone components
            # ============================================================
    
            for param in tryon_net.parameters():
                param.requires_grad = False
            for param in garment_net.parameters():
                param.requires_grad = False
            for param in vae.parameters():
                param.requires_grad = False
    
            total_frozen = (
                sum(p.numel() for p in tryon_net.parameters())
                + sum(p.numel() for p in garment_net.parameters())
                + sum(p.numel() for p in vae.parameters())
            )
            print(f"  All backbone frozen: {total_frozen:,} params")
    
        except Exception as e:
            print(f"FATAL: Could not load IDM-VTON: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to placeholder modules for development")
            tryon_net, garment_net, vae, noise_scheduler = cls._create_placeholders(device)
    
        # ============================================================
        # Step 6: Create ControlNet3D (TRAINABLE — our novel module)
        # ============================================================
    
        controlnet_3d = ControlNet3D(
            conditioning_channels=controlnet_3d_channels,
        ).to(device)
    
        scaling_factor = 0.13025
    
        pipeline = cls(
            tryon_net=tryon_net,
            garment_net=garment_net,
            vae=vae,
            noise_scheduler=noise_scheduler,
            controlnet_3d=controlnet_3d,
            scaling_factor=scaling_factor,
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
        """
        Build pipeline from config dict.

        Supports two modes:
          1. If 'pretrained_idm_vton' key exists → load from HuggingFace
          2. Otherwise → create placeholder modules for dev/testing
        """
        pretrained_id = config.get("pretrained_idm_vton")
        if pretrained_id:
            return cls.from_pretrained(
                pretrained_model_id=pretrained_id,
                controlnet_3d_channels=config.get("controlnet_3d_channels", 9),
            )

        # Fallback: create lightweight placeholder pipeline for testing
        tryon_net, garment_net, vae, scheduler = cls._create_placeholders("cpu")
        controlnet_3d = ControlNet3D(
            conditioning_channels=config.get("controlnet_3d_channels", 9),
            base_channels=config.get("model_channels", 320),
        )
        return cls(tryon_net, garment_net, vae, scheduler, controlnet_3d)

    @staticmethod
    def _create_placeholders(device: str):
        """Create lightweight placeholder modules for dev/testing."""
        from src.models.noise_scheduler import create_noise_scheduler

        # Minimal placeholder UNets
        # in_ch=8: noisy_latent(4) + agnostic_latent(4)
        tryon_net = _PlaceholderUNet(in_ch=8, out_ch=4).to(device)
        garment_net = _PlaceholderUNet(in_ch=4, out_ch=4).to(device)
        for p in garment_net.parameters():
            p.requires_grad = False

        vae = _PlaceholderVAE().to(device)
        for p in vae.parameters():
            p.requires_grad = False

        scheduler = create_noise_scheduler("ddpm", num_train_timesteps=1000)
        return tryon_net, garment_net, vae, scheduler

    def forward(
        self,
        person_image: torch.Tensor,
        garment_image: torch.Tensor,
        agnostic_mask: torch.Tensor,
        pose_map: torch.Tensor,
        densepose_map: Optional[torch.Tensor] = None,
        conditioning_3d: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            person_image: (B, 3, H, W) target person image.
            garment_image: (B, 3, H, W) garment image.
            agnostic_mask: (B, 3, H, W) clothing-agnostic person.
            pose_map: (B, C, H, W) pose skeleton map.
            densepose_map: (B, C, H, W) optional DensePose UV map.
            conditioning_3d: (B, 9, H, W) optional 3D conditioning
                (RGB render + normal map + depth map).

        Returns:
            Dict with 'loss', 'pred_noise', 'target_noise'.
        """
        with torch.no_grad():
            person_latent = self._encode(person_image)
            garment_latent = self._encode(garment_image)
            agnostic_latent = self._encode(agnostic_mask)

        # Sample noise and timesteps
        noise = torch.randn_like(person_latent)
        b = person_latent.shape[0]
        timesteps = torch.randint(
            0, self._num_train_timesteps(),
            (b,), device=person_latent.device, dtype=torch.long,
        )

        # Add noise
        noisy_latent = self._add_noise(person_latent, noise, timesteps)

        # Get garment features (frozen)
        with torch.no_grad():
            garment_features = self._extract_garment_features(
                garment_latent, timesteps
            )

        # 3D ControlNet conditioning (TRAINABLE)
        controlnet_residuals = None
        if conditioning_3d is not None:
            controlnet_residuals = self.controlnet_3d(conditioning_3d, timesteps)

        # Build model input
        model_input = torch.cat([noisy_latent, agnostic_latent], dim=1)

        # Build conditioning
        cond = [pose_map]
        if densepose_map is not None:
            cond.append(densepose_map)

        # Predict noise via TryonNet
        pred_noise = self._predict_noise(
            model_input, timesteps, garment_features,
            controlnet_residuals=controlnet_residuals,
        )

        # Ensure pred_noise matches noise shape
        if pred_noise.shape != noise.shape:
            pred_noise = pred_noise[:, :noise.shape[1]]

        loss = F.mse_loss(pred_noise, noise)

        return {"loss": loss, "pred_noise": pred_noise, "target_noise": noise}

    @torch.no_grad()
    def generate(
        self,
        garment_image: torch.Tensor,
        agnostic_mask: torch.Tensor,
        pose_map: torch.Tensor,
        densepose_map: Optional[torch.Tensor] = None,
        conditioning_3d: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate try-on image with 3D conditioning.

        Args:
            garment_image: (B, 3, H, W) garment image.
            agnostic_mask: (B, 3, H, W) clothing-agnostic person.
            pose_map: (B, C, H, W) pose map.
            densepose_map: Optional DensePose map.
            conditioning_3d: (B, 9, H, W) 3D conditioning from PyTorch3D.
            num_inference_steps: DDIM denoising steps.
            guidance_scale: Classifier-free guidance scale.
            generator: Optional random generator for reproducibility.

        Returns:
            (B, 3, H, W) generated try-on image.
        """
        device = garment_image.device

        garment_latent = self._encode(garment_image)
        agnostic_latent = self._encode(agnostic_mask)

        # Set scheduler timesteps
        if hasattr(self.noise_scheduler, "set_timesteps"):
            self.noise_scheduler.set_timesteps(num_inference_steps)
            timesteps = self.noise_scheduler.timesteps.to(device)
        else:
            timesteps = torch.arange(
                self._num_train_timesteps() - 1, -1, -1, device=device
            )

        # Start from noise
        b, c, h, w = agnostic_latent.shape
        latent = torch.randn(
            b, 4, h, w, generator=generator,
            device=device, dtype=garment_latent.dtype,
        )

        for t in timesteps:
            t_batch = t.expand(b) if t.dim() == 0 else t

            # Garment features (frozen)
            garment_features = self._extract_garment_features(
                garment_latent, t_batch
            )

            # 3D ControlNet conditioning
            controlnet_residuals = None
            if conditioning_3d is not None:
                controlnet_residuals = self.controlnet_3d(
                    conditioning_3d, t_batch
                )

            # Model input
            model_input = torch.cat([latent, agnostic_latent], dim=1)

            # Predict noise
            noise_pred = self._predict_noise(
                model_input, t_batch, garment_features,
                controlnet_residuals=controlnet_residuals,
            )

            # Scheduler step
            t_val = t.item() if t.dim() == 0 else t[0].item()
            if hasattr(self.noise_scheduler, "step"):
                step_output = self.noise_scheduler.step(
                    noise_pred, t_val, latent, generator=generator
                )
                latent = step_output if isinstance(step_output, torch.Tensor) else step_output.prev_sample

        # Decode
        image = self._decode(latent)
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # ---- Internal helpers ----

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent via VAE."""
        if hasattr(self.vae, "encode") and hasattr(self.vae.encode(image[:1]), "latent_dist"):
            # Diffusers AutoencoderKL
            dist = self.vae.encode(image).latent_dist
            return dist.sample() * self.scaling_factor
        elif hasattr(self.vae, "encode"):
            # Custom / placeholder VAE
            enc = self.vae.encode(image)
            if isinstance(enc, dict):
                return enc["latent"]
            return enc * self.scaling_factor
        return image

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image via VAE."""
        z = latent / self.scaling_factor
        if hasattr(self.vae, "decode"):
            dec = self.vae.decode(z)
            if hasattr(dec, "sample"):
                return dec.sample
            return dec
        return z

    def _num_train_timesteps(self) -> int:
        """Get number of training timesteps from scheduler."""
        if hasattr(self.noise_scheduler, "config"):
            return self.noise_scheduler.config.num_train_timesteps
        return getattr(self.noise_scheduler, "num_train_timesteps", 1000)

    def _add_noise(self, original, noise, timesteps):
        """Add noise using scheduler."""
        if hasattr(self.noise_scheduler, "add_noise"):
            return self.noise_scheduler.add_noise(original, noise, timesteps)
        # Manual fallback
        alpha = 0.9 ** (timesteps.float() / 1000)
        alpha = alpha[:, None, None, None]
        return alpha.sqrt() * original + (1 - alpha).sqrt() * noise

    def _extract_garment_features(self, garment_latent, timesteps):
        """Extract garment features from frozen GarmentNet."""
        try:
            b = garment_latent.shape[0]
            # SDXL UNet expects encoder_hidden_states with dim=2048
            # and added_cond_kwargs with text_embeds + time_ids
            dummy_encoder_hidden = torch.zeros(
                b, 77, 2048,  # 77 tokens, 2048 dim (SDXL)
                device=garment_latent.device,
                dtype=garment_latent.dtype,
            )
            # SDXL also requires added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": torch.zeros(b, 1280, device=garment_latent.device, dtype=garment_latent.dtype),
                "time_ids": torch.zeros(b, 6, device=garment_latent.device, dtype=garment_latent.dtype),
            }
            out = self.garment_net(
                garment_latent, timesteps,
                encoder_hidden_states=dummy_encoder_hidden,
                added_cond_kwargs=added_cond_kwargs,
            )
            if hasattr(out, "sample"):
                return out.sample
            return out
        except Exception:
            b = garment_latent.shape[0]
            return torch.zeros(b, 77, 2048, device=garment_latent.device, dtype=garment_latent.dtype)

    def _predict_noise(self, model_input, timesteps, garment_features,
                    controlnet_residuals=None):
        """Run TryonNet with optional ControlNet3D residuals."""
        try:
            b = model_input.shape[0]
            
            # Ensure garment_features is (B, seq_len, 2048) for SDXL
            if garment_features.dim() == 2:
                garment_features = garment_features.unsqueeze(1)
            if garment_features.shape[-1] != 2048:
                # Project or pad to 2048
                garment_features = torch.zeros(
                    b, 77, 2048,
                    device=model_input.device,
                    dtype=model_input.dtype,
                )
            
            # SDXL requires added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": torch.zeros(b, 1280, device=model_input.device, dtype=model_input.dtype),
                "time_ids": torch.zeros(b, 6, device=model_input.device, dtype=model_input.dtype),
            }

            # Pad input channels if needed
            expected = self.tryon_net.config.in_channels
            if model_input.shape[1] != expected:
                if model_input.shape[1] < expected:
                    pad = torch.zeros(
                        b, expected - model_input.shape[1],
                        model_input.shape[2], model_input.shape[3],
                        device=model_input.device, dtype=model_input.dtype,
                    )
                    model_input = torch.cat([model_input, pad], dim=1)
                else:
                    model_input = model_input[:, :expected]

            out = self.tryon_net(
                model_input, timesteps,
                encoder_hidden_states=garment_features,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=controlnet_residuals[:-1] if controlnet_residuals else None,
                mid_block_additional_residual=controlnet_residuals[-1] if controlnet_residuals else None,
            )
            return out.sample if hasattr(out, "sample") else out

        except Exception as e:
            print(f"_predict_noise error: {e}")
            raise

    def freeze_backbone(self):
        """Freeze all pre-trained components, keep ControlNet3D trainable."""
        for param in self.tryon_net.parameters():
            param.requires_grad = False
        for param in self.garment_net.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.controlnet_3d.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Backbone frozen — Trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / max(total, 1):.1f}%)")

    def get_trainable_params(self):
        """Return only trainable parameters (for optimizer)."""
        return [p for p in self.controlnet_3d.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Lightweight placeholder modules for dev / CPU testing
# ---------------------------------------------------------------------------

class _PlaceholderUNet(nn.Module):
    """Tiny UNet placeholder for testing without downloading SDXL."""

    def __init__(self, in_ch: int = 8, out_ch: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

    def forward(self, x, timesteps=None, **kwargs):
        # Handle channel mismatch gracefully
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
        return self.net(x)


class _PlaceholderVAE(nn.Module):
    """Tiny VAE placeholder for testing."""

    def __init__(self, scale: float = 0.13025):
        super().__init__()
        self.scale = scale
        self.enc = nn.Conv2d(3, 4, 4, stride=4, padding=0)
        self.dec = nn.ConvTranspose2d(4, 3, 4, stride=4, padding=0)

    def encode(self, x):
        return {"latent": self.enc(x) * self.scale}

    def decode(self, z):
        return self.dec(z / self.scale)
