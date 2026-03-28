"""
ControlNet3D — 3D Conditioning Module for DiffFit-3D.

Processes 3D rendering outputs (RGB render, normal map, depth map) from the
PyTorch3D pipeline and produces multi-scale conditioning features that are
injected into the TryonNet (Person UNet) via residual connections.

This is the NOVEL CONTRIBUTION of DiffFit-3D: unlike 2D-only try-on methods,
this module provides the diffusion model with explicit 3D geometric cues
rendered from actual garment meshes draped onto the estimated body.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlNet3DConditioningEncoder(nn.Module):
    """
    Encodes raw 3D conditioning signals (RGB render + normal map + depth map)
    into a feature map matching the UNet's base channel dimension.

    Input: (B, 9, H, W) — 3ch RGB render + 3ch normal map + 3ch depth map
    Output: (B, base_channels, H, W)
    """

    def __init__(self, conditioning_channels: int = 9, base_channels: int = 320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, base_channels, 3, padding=1, stride=2),
            nn.SiLU(),
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.net(conditioning)


class ZeroConv(nn.Module):
    """Zero-initialized 1×1 convolution for residual injection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ControlNet3DResBlock(nn.Module):
    """Simplified residual block for the ControlNet encoder."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class ControlNet3D(nn.Module):
    """
    3D ControlNet for DiffFit-3D.

    Takes 3D rendering outputs (RGB render + normal map + depth map from
    PyTorch3D) and produces multi-scale conditioning features that are
    injected into the TryonNet's encoder blocks via zero-initialized
    residual connections.

    Architecture mirrors the encoder side of the SDXL UNet but with
    zero-initialized output projections so that the pre-trained model
    is not disturbed at initialization.

    Args:
        conditioning_channels: Number of input conditioning channels
            (default 9 = 3 RGB + 3 normal + 3 depth).
        base_channels: Base channel count (matches UNet).
        channel_mult: Channel multipliers per level.
        num_res_blocks: ResBlocks per level.
        time_emb_dim: Timestep embedding dimension.
    """

    def __init__(
        self,
        conditioning_channels: int = 9,
        base_channels: int = 320,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 1280,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.channel_mult = channel_mult

        # Timestep embedding (shared structure with UNet)
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Conditioning input encoder: (B, 9, H, W) → (B, base_ch, H/8, W/8)
        self.cond_encoder = ControlNet3DConditioningEncoder(
            conditioning_channels=conditioning_channels,
            base_channels=base_channels,
        )

        # Input conv (same as UNet — takes conditioned features)
        self.input_conv = nn.Conv2d(base_channels, base_channels, 1)

        # Encoder blocks (mirrors UNet encoder)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.zero_convs = nn.ModuleList()

        ch = base_channels
        # Zero conv for input level
        self.zero_convs.append(ZeroConv(ch))

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ControlNet3DResBlock(ch, out_ch, time_emb_dim)
                )
                ch = out_ch
                self.zero_convs.append(ZeroConv(ch))

            if level < len(channel_mult) - 1:
                self.downsamples.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
                self.zero_convs.append(ZeroConv(ch))

        # Middle block
        self.mid_block = ControlNet3DResBlock(ch, ch, time_emb_dim)
        self.mid_zero_conv = ZeroConv(ch)

    def _sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal timestep embeddings."""
        half_dim = self.base_channels // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        conditioning_3d: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Process 3D conditioning and produce multi-scale residuals.

        Args:
            conditioning_3d: (B, 9, H, W) — concatenated RGB render,
                normal map, and depth map from the 3D pipeline.
            timesteps: (B,) diffusion timesteps.

        Returns:
            List of tensors — one per encoder level + mid block — to be
            added as residuals to the TryonNet's corresponding levels.
            All outputs are zero-initialized so the pre-trained model
            starts unperturbed.
        """
        # Timestep embedding
        t_emb = self._sinusoidal_embedding(timesteps)
        t_emb = self.time_embed(t_emb)

        # Encode conditioning
        h = self.cond_encoder(conditioning_3d)
        h = self.input_conv(h)

        # Collect residuals
        residuals = [self.zero_convs[0](h)]
        zero_idx = 1
        ds_idx = 0

        for level, mult in enumerate(self.channel_mult):
            for block_i in range(2):  # num_res_blocks
                block_idx = level * 2 + block_i
                if block_idx < len(self.encoder_blocks):
                    h = self.encoder_blocks[block_idx](h, t_emb)
                    residuals.append(self.zero_convs[zero_idx](h))
                    zero_idx += 1

            if level < len(self.channel_mult) - 1:
                if ds_idx < len(self.downsamples):
                    h = self.downsamples[ds_idx](h)
                    residuals.append(self.zero_convs[zero_idx](h))
                    zero_idx += 1
                    ds_idx += 1

        # Mid block
        h = self.mid_block(h, t_emb)
        residuals.append(self.mid_zero_conv(h))

        return residuals

    @staticmethod
    def prepare_conditioning(
        render_rgb: torch.Tensor,
        normal_map: torch.Tensor,
        depth_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate 3D pipeline outputs into a single conditioning tensor.

        Args:
            render_rgb: (B, 3, H, W) RGB render from PyTorch3D.
            normal_map: (B, 3, H, W) surface normal map.
            depth_map: (B, 3, H, W) depth map (3ch for ControlNet compat).

        Returns:
            (B, 9, H, W) conditioning tensor.
        """
        return torch.cat([render_rgb, normal_map, depth_map], dim=1)
