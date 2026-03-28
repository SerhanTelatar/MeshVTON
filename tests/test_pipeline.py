"""Tests for the TryOnPipeline with IDM-VTON backbone and ControlNet3D."""

import pytest
import torch
from src.models.tryon_pipeline import TryOnPipeline
from src.models.controlnet_3d import ControlNet3D
from src.models.noise_scheduler import create_noise_scheduler, DDPMScheduler, DDIMScheduler


class TestControlNet3D:
    """Tests for the 3D ControlNet module."""

    def test_output_is_list(self):
        """ControlNet3D should return a list of multi-scale residuals."""
        cn = ControlNet3D(conditioning_channels=9, base_channels=32,
                          channel_mult=(1, 1), num_res_blocks=1, time_emb_dim=128)
        cond = torch.randn(1, 9, 64, 64)
        t = torch.tensor([100])
        residuals = cn(cond, t)
        assert isinstance(residuals, list)
        assert len(residuals) > 0

    def test_zero_initialization(self):
        """All outputs should be zero at initialization (no perturbation)."""
        cn = ControlNet3D(conditioning_channels=9, base_channels=32,
                          channel_mult=(1, 1), num_res_blocks=1, time_emb_dim=128)
        cond = torch.randn(1, 9, 64, 64)
        t = torch.tensor([100])
        residuals = cn(cond, t)
        for r in residuals:
            assert torch.allclose(r, torch.zeros_like(r), atol=1e-5), \
                "Zero-initialized ControlNet should produce zero outputs"

    def test_prepare_conditioning(self):
        """prepare_conditioning should concatenate 3 maps into 9 channels."""
        rgb = torch.randn(2, 3, 64, 64)
        normal = torch.randn(2, 3, 64, 64)
        depth = torch.randn(2, 3, 64, 64)
        cond = ControlNet3D.prepare_conditioning(rgb, normal, depth)
        assert cond.shape == (2, 9, 64, 64)

    def test_batch_size(self):
        """ControlNet3D should handle various batch sizes."""
        cn = ControlNet3D(conditioning_channels=9, base_channels=32,
                          channel_mult=(1, 1), num_res_blocks=1, time_emb_dim=128)
        for bs in [1, 2, 4]:
            cond = torch.randn(bs, 9, 64, 64)
            t = torch.tensor([100] * bs)
            residuals = cn(cond, t)
            assert residuals[0].shape[0] == bs


class TestNoiseScheduler:
    def test_ddpm_creation(self):
        scheduler = create_noise_scheduler("ddpm", num_train_timesteps=100)
        assert isinstance(scheduler, DDPMScheduler)
        assert len(scheduler.betas) == 100

    def test_ddim_creation(self):
        scheduler = create_noise_scheduler("ddim", num_train_timesteps=1000, num_inference_steps=50)
        assert isinstance(scheduler, DDIMScheduler)
        assert len(scheduler.timesteps) == 50

    def test_add_noise(self):
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        x = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x)
        t = torch.tensor([100, 500])
        noisy = scheduler.add_noise(x, noise, t)
        assert noisy.shape == x.shape
        assert not torch.allclose(noisy, x)


class TestPipelineCreation:
    def test_from_config_placeholder(self):
        """Pipeline should build with placeholder modules (no HuggingFace)."""
        config = {
            "model_channels": 32,
            "controlnet_3d_channels": 9,
        }
        pipeline = TryOnPipeline.from_config(config)
        assert isinstance(pipeline, TryOnPipeline)
        assert isinstance(pipeline.controlnet_3d, ControlNet3D)

    def test_trainable_params_only_controlnet(self):
        """Only ControlNet3D params should be trainable after freeze."""
        config = {"model_channels": 32, "controlnet_3d_channels": 9}
        pipeline = TryOnPipeline.from_config(config)
        pipeline.freeze_backbone()

        trainable = [n for n, p in pipeline.named_parameters() if p.requires_grad]
        assert all("controlnet_3d" in n for n in trainable), \
            f"Non-ControlNet3D params are trainable: {[n for n in trainable if 'controlnet_3d' not in n]}"

    def test_forward_with_3d_conditioning(self):
        """Pipeline forward should work with 3D conditioning."""
        config = {"model_channels": 32, "controlnet_3d_channels": 9}
        pipeline = TryOnPipeline.from_config(config)

        b, h, w = 1, 64, 64
        person = torch.randn(b, 3, h, w)
        garment = torch.randn(b, 3, h, w)
        agnostic = torch.randn(b, 3, h, w)
        pose = torch.randn(b, 3, h, w)
        cond_3d = torch.randn(b, 9, h, w)

        outputs = pipeline(
            person_image=person,
            garment_image=garment,
            agnostic_mask=agnostic,
            pose_map=pose,
            conditioning_3d=cond_3d,
        )
        assert "loss" in outputs
        assert outputs["loss"].ndim == 0  # scalar

    def test_forward_without_3d_conditioning(self):
        """Pipeline should also work WITHOUT 3D conditioning (traditional mode)."""
        config = {"model_channels": 32, "controlnet_3d_channels": 9}
        pipeline = TryOnPipeline.from_config(config)

        b, h, w = 1, 64, 64
        outputs = pipeline(
            person_image=torch.randn(b, 3, h, w),
            garment_image=torch.randn(b, 3, h, w),
            agnostic_mask=torch.randn(b, 3, h, w),
            pose_map=torch.randn(b, 3, h, w),
            conditioning_3d=None,
        )
        assert "loss" in outputs
