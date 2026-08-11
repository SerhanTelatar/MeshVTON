"""LoRA attachment + trainable-state helpers (peft; the imports are lazy).

The trainable set = the LoRA layers + ControlXEmbedder.control_proj. The checkpoint dict is
gathered from transformer.named_parameters() with a requires_grad filter — LoRA and the
control sidecar live in one file, with a natural namespace.
"""

from __future__ import annotations

import torch

# Standard targets for the FLUX blocks: the attention projections + FF (both streams use
# to_q/to_k/to_v/to_out; consistent with the diffusers FLUX LoRA training recipes)
DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")


def _silence_torchao_check() -> None:
    """peft's torchao version check RAISES an ImportError when torchao is INSTALLED but <0.16
    (the Colab image) — it should have returned False instead. We do NOT USE torchao;
    at both hook points is_torchao_available -> False (permanent, not dependent on pip;
    no need to repeat an uninstall every session)."""
    import importlib

    for modname in ("peft.import_utils", "peft.tuners.lora.torchao"):
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "is_torchao_available"):
                m.is_torchao_available = lambda: False
        except Exception:
            pass


def attach_lora(transformer, *, rank: int = 64, alpha: int = 64, targets=DEFAULT_TARGETS):
    """Adds LoRA to a FluxTransformer2DModel (diffusers PeftAdapterMixin.add_adapter)."""
    _silence_torchao_check()
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
        raise KeyError(f"{len(missing)} keys in the checkpoint that the model does not have (first: {missing[0]})")
    with torch.no_grad():
        for n, w in state.items():
            own[n].copy_(w.to(own[n].dtype))
