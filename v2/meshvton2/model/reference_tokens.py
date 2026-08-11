"""FLUX latent packing + Kontext-style reference tokens.

FLUX packs a (B,C,H,W) latent into a (B, H/2·W/2, 4C) token sequence with 2×2 patches;
each token has a 3-channel position id: [frame_idx, row, column]. Kontext-dev appends the
reference frame to the same sequence with frame_idx=1 — we append the appearance reference
(the latent of the textured garment render) the same way:

    target tokens: ids[...,0]=0     reference tokens: ids[...,0]=1

The loss covers the target tokens only (token_mask); the reference is given clean at every step.
"""

from __future__ import annotations

import torch


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B, H/2·W/2, 4C). H and W must be even."""
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"H,W must be even: {(h, w)}")
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The inverse of pack_latents. height/width: the ORIGINAL latent dimensions."""
    b, l, d = tokens.shape
    h2, w2 = height // 2, width // 2
    if l != h2 * w2 or d % 4:
        raise ValueError(f"Mismatch: L={l}!={h2 * w2} or D={d}%4")
    c = d // 4
    x = tokens.view(b, h2, w2, c, 2, 2)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, height, width)


def make_img_ids(height: int, width: int, frame_idx: int = 0, device="cpu") -> torch.Tensor:
    """(H/2·W/2, 3) float: [frame_idx, row, column] — FLUX RoPE position ids."""
    h2, w2 = height // 2, width // 2
    ids = torch.zeros(h2, w2, 3, device=device)
    ids[..., 0] = frame_idx
    ids[..., 1] = torch.arange(h2, device=device).unsqueeze(1)
    ids[..., 2] = torch.arange(w2, device=device).unsqueeze(0)
    return ids.reshape(h2 * w2, 3)


def concat_reference(
    target_tokens: torch.Tensor,   # (B, Lt, D)
    target_ids: torch.Tensor,      # (Lt, 3) frame 0
    ref_tokens: torch.Tensor,      # (B, Lr, D)
    ref_ids: torch.Tensor,         # (Lr, 3) frame 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (tokens (B,Lt+Lr,D), ids (Lt+Lr,3), target_mask (Lt+Lr,)).
    target_mask: loss/sampling applies to the target tokens only."""
    tokens = torch.cat([target_tokens, ref_tokens], dim=1)
    ids = torch.cat([target_ids, ref_ids], dim=0)
    mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
    mask[: target_tokens.shape[1]] = True
    return tokens, ids, mask


def pack_pixel_mask(mask: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
    """FLUX Fill's mask channels: a pixel mask (B,1,H,W) -> (B, L, 256).

    Fill does not downsample the mask to the latent; it unfolds each latent pixel's 8x8 pixel
    block into 64 channels, then packs it into 256 channels with the 2x2 latent pack (identical
    to FluxFillPipeline.prepare_mask_latents). H must be 8*latent_height, W 8*latent_width.
    """
    b, c, h, w = mask.shape
    if c != 1 or h != 8 * latent_height or w != 8 * latent_width:
        raise ValueError(f"mask must be (B,1,{8*latent_height},{8*latent_width}), got {tuple(mask.shape)}")
    m = mask.view(b, latent_height, 8, latent_width, 8)
    m = m.permute(0, 2, 4, 1, 3).reshape(b, 64, latent_height, latent_width)
    return pack_latents(m)  # (B, L, 256)
