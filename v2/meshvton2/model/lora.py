"""LoRA bağlama + eğitilebilir-durum yardımcıları (peft; import'lar tembel).

Eğitilebilir küme = LoRA katmanları + ControlXEmbedder.control_proj. Checkpoint
sözlüğü transformer.named_parameters() üzerinden requires_grad filtresiyle
toplanır — LoRA ve kontrol sidecar'ı tek dosyada, isim uzayı doğal.
"""

from __future__ import annotations

import torch

# FLUX blokları için standart hedefler: dikkat projeksiyonları + FF (iki akış da
# to_q/to_k/to_v/to_out kullanır; diffusers FLUX LoRA eğitim tarifleriyle uyumlu)
DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")


def attach_lora(transformer, *, rank: int = 64, alpha: int = 64, targets=DEFAULT_TARGETS):
    """FluxTransformer2DModel'e LoRA ekler (diffusers PeftAdapterMixin.add_adapter)."""
    from peft import LoraConfig

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=list(targets)
    )
    transformer.add_adapter(cfg)
    return cfg


def trainable_parameters(module: torch.nn.Module):
    return (p for p in module.parameters() if p.requires_grad)


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad}


def load_trainable_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = dict(module.named_parameters())
    missing = [n for n in state if n not in own]
    if missing:
        raise KeyError(f"Checkpoint'te modelde olmayan {len(missing)} anahtar (ilk: {missing[0]})")
    with torch.no_grad():
        for n, w in state.items():
            own[n].copy_(w.to(own[n].dtype))
