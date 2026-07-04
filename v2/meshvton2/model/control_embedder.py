"""Zero-init geometri kontrol embedder'ı (BFL Depth/Canny "Control" tarifi).

FLUX Fill'in x_embedder'ı packed girdiyi (384 kanal: latent 64 + masked 64 + mask 256)
3072'ye projekte eden bir Linear'dır. Kontrol görüntüleri (normal, depth+silüet)
VAE-encode + pack edilip kanal olarak eklenir. Ağırlık matrisini genişletmek yerine
PARALEL zero-init bir Linear kullanıyoruz:

    out = x_embedder(x_orig)  +  control_proj(x_control)      # control_proj W=0, b yok

Neden paralel (genişletme değil):
- Init'te çıktı bit-eş stok Fill ≡ v1'in "adapter bozuyor" hatası yapısal imkânsız
  (test_control_zero_init bunu kilitler).
- Orijinal kolonlar ayrı tensör → dondurmak trivial (grad maskesi gerekmez).
- Checkpoint = yalnız control_proj.state_dict() (~0.8M param sidecar).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ControlXEmbedder(nn.Module):
    """FluxTransformer2DModel.x_embedder'ın yerine geçer.

    forward(x): x (B, L, orig_in + control_in) — son control_in kanal kontrol.
    Kontrol kanalları eksikse (stok çağrı, B,L,orig_in) aynen orijinal davranır.
    """

    def __init__(self, original: nn.Linear, control_in_features: int):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)  # orijinal kolonlar donuk; adaptasyon LoRA'nın işi
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
        if x.shape[-1] == self.orig_in_features:  # kontrolsüz çağrı (stok davranış)
            return self.original(x)
        expected = self.orig_in_features + self.control_in_features
        if x.shape[-1] != expected:
            raise ValueError(
                f"x_embedder girdisi {x.shape[-1]} kanal; {self.orig_in_features} (stok) "
                f"veya {expected} (kontrollü) olmalı"
            )
        # .contiguous() şart: dilimlenmiş girdi farklı matmul çekirdeğine düşüp
        # ~5e-7 sapma yaratıyor — "init'te bit-eş stok Fill" garantisi bozulurdu.
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
    """FluxTransformer2DModel'in x_embedder'ını sarmalayıcıyla değiştirir."""
    if isinstance(transformer.x_embedder, ControlXEmbedder):
        return transformer.x_embedder
    wrapped = ControlXEmbedder(transformer.x_embedder, control_in_features)
    transformer.x_embedder = wrapped
    return wrapped
