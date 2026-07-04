"""FLUX latent paketleme + Kontext-tarzı referans token'ları.

FLUX, (B,C,H,W) latent'i 2×2 patch'lerle (B, H/2·W/2, 4C) token dizisine paketler;
her token'ın 3 kanallı konum id'si vardır: [frame_idx, satır, sütun]. Kontext-dev
referans kareyi frame_idx=1 ile aynı diziye ekler — biz de görünüm referansını
(dokulu giysi render'ının latent'i) böyle ekliyoruz:

    hedef token'lar: ids[...,0]=0     referans token'lar: ids[...,0]=1

Kayıp yalnız hedef token'larda (token_mask); referans her adımda temiz verilir.
"""

from __future__ import annotations

import torch


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B, H/2·W/2, 4C). H ve W çift olmalı."""
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"H,W çift olmalı: {(h, w)}")
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """pack_latents'in tersi. height/width: ORİJİNAL latent boyutları."""
    b, l, d = tokens.shape
    h2, w2 = height // 2, width // 2
    if l != h2 * w2 or d % 4:
        raise ValueError(f"Uyumsuz: L={l}!={h2 * w2} veya D={d}%4")
    c = d // 4
    x = tokens.view(b, h2, w2, c, 2, 2)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, height, width)


def make_img_ids(height: int, width: int, frame_idx: int = 0, device="cpu") -> torch.Tensor:
    """(H/2·W/2, 3) float: [frame_idx, satır, sütun] — FLUX RoPE konum id'leri."""
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
    target_mask: kayıp/örnekleme yalnız hedef token'larında."""
    tokens = torch.cat([target_tokens, ref_tokens], dim=1)
    ids = torch.cat([target_ids, ref_ids], dim=0)
    mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
    mask[: target_tokens.shape[1]] = True
    return tokens, ids, mask


def pack_pixel_mask(mask: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
    """FLUX Fill'in maske kanalları: piksel maskesi (B,1,H,W) -> (B, L, 256).

    Fill, maskeyi latent'e indirmez; her latent pikselin 8x8 piksel bloğunu 64
    kanala açar, sonra 2x2 latent-pack ile 256 kanala paketler (FluxFillPipeline.
    prepare_mask_latents ile aynı). H = 8*latent_height, W = 8*latent_width olmalı.
    """
    b, c, h, w = mask.shape
    if c != 1 or h != 8 * latent_height or w != 8 * latent_width:
        raise ValueError(f"mask (B,1,{8*latent_height},{8*latent_width}) olmalı, gelen {tuple(mask.shape)}")
    m = mask.view(b, latent_height, 8, latent_width, 8)
    m = m.permute(0, 2, 4, 1, 3).reshape(b, 64, latent_height, latent_width)
    return pack_latents(m)  # (B, L, 256)
