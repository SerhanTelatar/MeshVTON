"""
TryOn Pipeline — Main dual-branch diffusion pipeline.

Orchestrates the Person UNet and Garment UNet branches, coordinating
the full diffusion forward/reverse process for virtual try-on.
"""

from typing import Optional
import torch
import torch.nn as nn
from src.models.person_unet import PersonUNet
from src.models.garment_unet import GarmentUNet
from src.models.vae import VAEWrapper
from src.models.noise_scheduler import create_noise_scheduler, DDPMScheduler, DDIMScheduler


class TryOnPipeline(nn.Module):
    """
    Dual-branch Latent Diffusion Pipeline for Virtual Try-On.

    Combines Person UNet (body preservation) and Garment UNet (texture transfer)
    with cross-attention fusion. Supports both training and inference modes.
    """

    def __init__(self, person_unet: PersonUNet, garment_unet: GarmentUNet,
                 vae: VAEWrapper, noise_scheduler: DDPMScheduler | DDIMScheduler):
        super().__init__()
        self.person_unet = person_unet
        self.garment_unet = garment_unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler

    @classmethod
    def from_config(cls, config: dict) -> "TryOnPipeline":
        """Build pipeline from a configuration dictionary."""
        person_unet = PersonUNet(
                    in_channels=config.get("person_in_channels", 9),
                    model_channels=config.get("model_channels", 320),
                    out_channels=config.get("out_channels", 4),
                    context_dim=config.get("context_dim", 768),
                    controlnet_channels=9,  # ← bunu ekle/düzelt
        )
        garment_unet = GarmentUNet(
            in_channels=config.get("garment_in_channels", 4),
            model_channels=config.get("model_channels", 320),
            out_channels=config.get("out_channels", 4),
            context_dim=config.get("context_dim", 768),
        )
        vae = VAEWrapper(
            latent_channels=config.get("latent_channels", 4),
            scaling_factor=config.get("scaling_factor", 0.18215),
        )
        scheduler = create_noise_scheduler(
            scheduler_type=config.get("scheduler_type", "ddpm"),
            num_train_timesteps=config.get("num_train_timesteps", 1000),
        )
        return cls(person_unet, garment_unet, vae, scheduler)

    def forward(self, person_image: torch.Tensor, garment_image: torch.Tensor,
                agnostic_mask: torch.Tensor, pose_map: torch.Tensor,
                densepose_map: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            person_image: (B, 3, H, W) target person image.
            garment_image: (B, 3, H, W) garment image.
            agnostic_mask: (B, 3, H, W) clothing-agnostic person.
            pose_map: (B, C_pose, H, W) pose skeleton map.
            densepose_map: (B, 1, H, W) optional DensePose UV map.

        Returns:
            Dict with 'loss', 'pred_noise', 'target_noise'.
        """
        with torch.no_grad():
            person_latent = self.vae.encode(person_image)["latent"]
            garment_latent = self.vae.encode(garment_image)["latent"]

        # Sample noise and timesteps
        noise = torch.randn_like(person_latent)
        batch_size = person_latent.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,), device=person_latent.device, dtype=torch.long,
        )

        # Add noise to person latent
        noisy_latent = self.noise_scheduler.add_noise(person_latent, noise, timesteps)

        # Get garment features via garment UNet
        garment_result = self.garment_unet(
            garment_latent, timesteps, return_features=True,
        )
        garment_features = garment_result["features"]

        # Build conditioning input
        cond_inputs = [agnostic_mask, pose_map]
        if densepose_map is not None:
            cond_inputs.append(densepose_map)
        controlnet_cond = torch.cat(cond_inputs, dim=1)

        # Concat agnostic mask with noisy latent
        agnostic_latent = self.vae.encode(agnostic_mask)["latent"]
        model_input = torch.cat([noisy_latent, agnostic_latent], dim=1)

        # Pad to expected channels if needed
        expected_ch = self.person_unet.input_conv.in_channels
        current_ch = model_input.shape[1]
        if current_ch < expected_ch:
            padding = torch.zeros(
                batch_size, expected_ch - current_ch,
                model_input.shape[2], model_input.shape[3],
                device=model_input.device, dtype=model_input.dtype,
            )
            model_input = torch.cat([model_input, padding], dim=1)

        # Predict noise
        pred_noise = self.person_unet(
            model_input, timesteps, context=garment_features,
            controlnet_cond=controlnet_cond,
        )

        # MSE loss on predicted noise
        loss = nn.functional.mse_loss(pred_noise, noise)

        return {"loss": loss, "pred_noise": pred_noise, "target_noise": noise}

    @torch.no_grad()
    def generate(self, garment_image: torch.Tensor, agnostic_mask: torch.Tensor,
                 pose_map: torch.Tensor, densepose_map: Optional[torch.Tensor] = None,
                 num_inference_steps: int = 50, guidance_scale: float = 7.5,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Inference: generate try-on image.

        Args:
            garment_image: (B, 3, H, W) garment image.
            agnostic_mask: (B, 3, H, W) clothing-agnostic person.
            pose_map: (B, C, H, W) pose map.
            densepose_map: Optional DensePose map.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            generator: Optional random generator for reproducibility.

        Returns:
            (B, 3, H, W) generated try-on image.
        """
        device = garment_image.device

        # Encode garment
        garment_latent = self.vae.encode(garment_image)["latent"]
        agnostic_latent = self.vae.encode(agnostic_mask)["latent"]

        # Build conditioning
        cond_inputs = [agnostic_mask, pose_map]
        if densepose_map is not None:
            cond_inputs.append(densepose_map)
        controlnet_cond = torch.cat(cond_inputs, dim=1)

        # Get scheduler timesteps
        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(num_inference_steps)
            timesteps = self.noise_scheduler.timesteps.to(device)
        else:
            timesteps = torch.arange(
                self.noise_scheduler.num_train_timesteps - 1, -1, -1, device=device
            )

        # Start from random noise
        b, c, h, w = agnostic_latent.shape
        latent = torch.randn(b, 4, h, w, generator=generator, device=device, dtype=garment_latent.dtype)

        # Denoising loop
        for t in timesteps:
            t_batch = t.expand(b) if t.dim() == 0 else t

            # Get garment features
            garment_result = self.garment_unet(garment_latent, t_batch, return_features=True)
            garment_features = garment_result["features"]

            # Build model input
            model_input = torch.cat([latent, agnostic_latent], dim=1)
            expected_ch = self.person_unet.input_conv.in_channels
            current_ch = model_input.shape[1]
            if current_ch < expected_ch:
                padding = torch.zeros(b, expected_ch - current_ch, h, w, device=device, dtype=latent.dtype)
                model_input = torch.cat([model_input, padding], dim=1)

            # Predict noise
            noise_pred = self.person_unet(
                model_input, t_batch, context=garment_features, controlnet_cond=controlnet_cond,
            )

            # Scheduler step
            t_val = t.item() if t.dim() == 0 else t[0].item()
            latent = self.noise_scheduler.step(noise_pred, t_val, latent, generator=generator)

        # Decode
        image = self.vae.decode(latent)
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
