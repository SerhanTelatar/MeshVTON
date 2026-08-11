"""Rectified flow (FLUX-native) training maths.

Contract (the same as the FLUX/SD3 family):
    x_t   = (1 - t)·x0 + t·ε            t ∈ (0,1); t=0 clean, t=1 pure noise
    target v = ε - x0                    (velocity)
    x0    = x_t - t·v                    (solved back for preview/consistency loss)

Time sampling: logit-normal (the FLUX training recipe) + a resolution-dependent shift —
for long token sequences it shifts the t distribution towards noise (the SD3 §5.3.2 recipe:
t' = s·t / (1 + (s-1)·t)).
"""

from __future__ import annotations

import math

import torch


def sample_logit_normal_t(
    batch: int,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """(B,) t ~ sigmoid(N(mean, std)) — little mass at the ends (0/1), a lot in the middle."""
    z = torch.randn(batch, generator=generator, device=device) * std + mean
    return torch.sigmoid(z)


def resolution_shift(seq_len: int, base_seq_len: int = 256, base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    """FLUX's dynamic shift: mu grows linearly with the token count.
    (Same form as diffusers FluxPipeline.calculate_shift; s = exp(mu))."""
    max_seq_len = 4096
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = base_shift + m * (seq_len - base_seq_len)
    return math.exp(mu)


def apply_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """t' = s·t / (1 + (s-1)·t). s=1 → identity; s>1 → pushes t towards 1 (towards noise)."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def _bcast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, *([1] * (like.dim() - 1))).to(like.dtype)


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    tb = _bcast(t, x0)
    return (1.0 - tb) * x0 + tb * noise


def rf_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - x0


def x0_from_v(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return x_t - _bcast(t, x_t) * v


def rf_loss(v_pred: torch.Tensor, x0: torch.Tensor, noise: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """MSE(v_pred, ε−x0). If token_mask (B,L) is given, only over those tokens —
    the reference tokens do NOT enter the loss (they are given clean at every step)."""
    err = (v_pred - rf_target(x0, noise)) ** 2
    if token_mask is None:
        return err.mean()
    m = token_mask.to(err.dtype)
    while m.dim() < err.dim():
        m = m.unsqueeze(-1)
    denom = (m.expand_as(err)).sum().clamp_min(1.0)
    return (err * m).sum() / denom


def make_sigma_schedule(steps: int, seq_len: int) -> torch.Tensor:
    """(steps+1,) a decreasing sigma schedule: 1 → 0, with the resolution shift.
    The sampling Euler step: x_{i+1} = x_i + (σ_{i+1} − σ_i)·v(x_i, σ_i)."""
    t = torch.linspace(1.0, 0.0, steps + 1)
    return apply_shift(t, resolution_shift(seq_len))
