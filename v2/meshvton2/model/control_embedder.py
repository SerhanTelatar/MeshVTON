"""Zero-init geometry control embedder (the BFL Depth/Canny "Control" recipe).

FLUX Fill's x_embedder is a Linear projecting the packed input (384 channels: latent 64 +
masked 64 + mask 256) to 3072. The control images (normal, depth+silhouette) are VAE-encoded,
packed and appended as channels. Instead of widening the weight matrix we use a PARALLEL
zero-init Linear:

    out = x_embedder(x_orig)  +  control_proj(x_control)      # control_proj W=0, no bias

Why parallel (rather than widening):
- At init the output is bit-identical to stock Fill ≡ v1's "the adapter breaks it" failure is
  structurally impossible (test_control_zero_init locks this in).
- The original columns stay a separate tensor → freezing them is trivial (no grad mask needed).
- The checkpoint is only control_proj.state_dict() (a ~0.8M param sidecar).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ControlXEmbedder(nn.Module):
    """Replaces FluxTransformer2DModel.x_embedder.

    forward(x): x (B, L, orig_in + control_in) — the last control_in channels are the control.
    If the control channels are missing (a stock call, B,L,orig_in) it behaves exactly like the original.
    """

    def __init__(self, original: nn.Linear, control_in_features: int):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)  # the original columns are frozen; adaptation is LoRA's job
        self.control_in_features = control_in_features
        self.control_proj = nn.Linear(
            control_in_features, original.out_features, bias=False,
            dtype=original.weight.dtype, device=original.weight.device,
        )
        nn.init.zeros_(self.control_proj.weight)

    @property
    def orig_in_features(self) -> int:
        return self.original.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.orig_in_features:  # call without control (stock behaviour)
            return self.original(x)
        expected = self.orig_in_features + self.control_in_features
        if x.shape[-1] != expected:
            raise ValueError(
                f"x_embedder input has {x.shape[-1]} channels; it must be {self.orig_in_features} "
                f"(stock) or {expected} (with control)"
            )
        # .contiguous() is required: a sliced input falls into a different matmul kernel and
        # creates a ~5e-7 deviation — that would break the "bit-identical stock Fill at init" guarantee.
        base = x[..., : self.orig_in_features].contiguous()
        ctrl = x[..., self.orig_in_features :].contiguous()
        return self.original(base) + self.control_proj(ctrl)

    def trainable_parameters(self):
        return self.control_proj.parameters()

    def save_sidecar(self, path: str) -> None:
        torch.save({"control_proj.weight": self.control_proj.weight.detach().cpu()}, path)

    def load_sidecar(self, path: str) -> None:
        w = torch.load(path, map_location="cpu", weights_only=True)["control_proj.weight"]
        with torch.no_grad():
            self.control_proj.weight.copy_(w.to(self.control_proj.weight.dtype))


def attach_control_embedder(transformer, control_in_features: int) -> ControlXEmbedder:
    """Replaces FluxTransformer2DModel's x_embedder with the wrapper."""
    if isinstance(transformer.x_embedder, ControlXEmbedder):
        return transformer.x_embedder
    wrapped = ControlXEmbedder(transformer.x_embedder, control_in_features)
    transformer.x_embedder = wrapped
    return wrapped
