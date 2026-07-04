"""EMA (v1 src/training/ema.py'den kopya — davranış aynı)."""

import copy
from typing import Optional
import torch
import torch.nn as nn


class EMAModel:
    """
    EMA tracker for model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (higher = slower update).
        update_after_step: Start EMA after this many steps.
        update_every: Update EMA every N steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 update_after_step: int = 100, update_every: int = 10):
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.shadow_params = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        self.collected_params = None

    def update(self, step: int, model: Optional[nn.Module] = None):
        """Update EMA parameters."""
        if step < self.update_after_step:
            return
        if step % self.update_every != 0:
            return
        if model is None:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params and param.requires_grad:
                    self.shadow_params[name].lerp_(param.data, 1 - self.decay)

    def apply_to(self, model: nn.Module):
        """Apply EMA weights to model (for inference)."""
        self.collected_params = {}
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                self.collected_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])

    def restore(self, model: nn.Module):
        """Restore original weights after EMA inference."""
        if self.collected_params is not None:
            for name, param in model.named_parameters():
                if name in self.collected_params:
                    param.data.copy_(self.collected_params[name])
            self.collected_params = None

    def state_dict(self) -> dict:
        return {"shadow_params": self.shadow_params, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow_params = state["shadow_params"]
        self.decay = state.get("decay", self.decay)


class EMAContextManager:
    """Context manager for temporarily applying EMA weights."""

    def __init__(self, ema: EMAModel, model: nn.Module):
        self.ema = ema
        self.model = model

    def __enter__(self):
        self.ema.apply_to(self.model)
        return self.model

    def __exit__(self, *args):
        self.ema.restore(self.model)
