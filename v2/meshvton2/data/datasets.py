"""Training datasets: single view (Stage 1), 4-view bundle (Stage 2), and a mix.

The mix ratio (real:synthetic ≈ 70:30) is the domain-gap insurance; the ratio comes from the
config and is deterministic (same seed → same order; safe for a Colab resume).
"""

from __future__ import annotations

import hashlib

import torch
from torch.utils.data import Dataset

from meshvton2.data.items import TrainItem, load_item


class SingleViewDataset(Dataset):
    """One item per (sample, view) — Stage 1. use_latents: the VAE-free fast path."""

    def __init__(self, items: list[TrainItem], size: tuple[int, int] | None = None,
                 use_latents: bool = False):
        if not items:
            raise ValueError("Empty dataset")
        self.items = items
        self.size = size
        self.use_latents = use_latents

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return load_item(self.items[i], self.size, use_latents=self.use_latents)


class BundleDataset(Dataset):
    """ALL views stacked per sample — Stage 2 (the views land in the same batch).
    items_by_sample: a view-ordered TrainItem list for each sample."""

    def __init__(self, items_by_sample: list[list[TrainItem]], size=None):
        if not items_by_sample:
            raise ValueError("Empty dataset")
        n = len(items_by_sample[0])
        if any(len(g) != n for g in items_by_sample):
            raise ValueError("Every sample must have the same number of views")
        self.groups = items_by_sample
        self.size = size

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        views = [load_item(it, self.size) for it in self.groups[i]]
        return {k: torch.stack([v[k] for v in views]) for k in views[0]}  # (V,C,H,W)


class MixedDataset(Dataset):
    """Mixes two datasets at a deterministic ratio. __len__ = steps_per_epoch;
    the source of item i comes from hash(i,seed) — the order does not change on resume."""

    def __init__(self, primary: Dataset, secondary: Dataset, *, primary_ratio: float = 0.7,
                 length: int | None = None, seed: int = 0):
        if not 0.0 < primary_ratio < 1.0:
            raise ValueError("primary_ratio must be in the (0,1) range")
        self.a, self.b = primary, secondary
        self.ratio = primary_ratio
        self.length = length or (len(primary) + len(secondary))
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def _pick(self, i: int) -> tuple[Dataset, int]:
        h = hashlib.sha256(f"{self.seed}:{i}".encode()).digest()
        u = int.from_bytes(h[:8], "little") / 2**64
        idx = int.from_bytes(h[8:16], "little")
        if u < self.ratio:
            return self.a, idx % len(self.a)
        return self.b, idx % len(self.b)

    def __getitem__(self, i: int) -> dict:
        ds, j = self._pick(i)
        return ds[j]
