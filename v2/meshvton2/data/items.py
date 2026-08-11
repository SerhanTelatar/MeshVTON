"""Training sample directory contract — ONE format, two sources:

    {item_dir}/gt.png agnostic.png mask.png normal.png depth_sil.png [appearance_ref.png]

- Synthetic: synth/generate.py writes under `{sample}/view_{deg}/`; appearance_ref lives at
  the sample root (the views share it).
- Real (VITON-HD): scripts/preprocess_vitonhd.py writes the same layout per person×garment
  (gt = the person's own photo; paired training).

The datasets read this contract — there is no source distinction (mixed training comes for free).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from meshvton2.utils.image_utils import load_image, load_mask, pil_to_tensor

FILES = ("gt", "agnostic", "mask", "normal", "depth_sil")


@dataclass(frozen=True)
class TrainItem:
    item_dir: Path
    appearance_ref: Path  # can be shared across views → a separate path

    def validate(self) -> bool:
        return all((self.item_dir / f"{f}.png").exists() for f in FILES) and self.appearance_ref.exists()


def load_item(item: TrainItem, size: tuple[int, int] | None = None, use_latents: bool = False) -> dict[str, torch.Tensor]:
    """-> a tensor dict matching the ConditioningBundle field names, all [-1,1] (mask {0,1}).

    use_latents=True: reads the precompute_latents.py output (gt_lat/masked_lat/
    normal_lat/depth_sil_lat/ref_lat + inpaint_mask) — no VAE in the training step.
    A missing latent file is an ERROR (it breaks mixed-batch collation; no silent fallback)."""
    if use_latents:
        lat_path = item.item_dir / "latents.pt"
        if not lat_path.exists():
            raise FileNotFoundError(
                f"{lat_path} is missing — run first: python v2/scripts/precompute_latents.py"
            )
        out = torch.load(lat_path, map_location="cpu", weights_only=True)
        out["inpaint_mask"] = load_mask(item.item_dir / "mask.png", size)
        return out
    out = {
        "gt_rgb": pil_to_tensor(load_image(item.item_dir / "gt.png", size)),
        "agnostic_rgb": pil_to_tensor(load_image(item.item_dir / "agnostic.png", size)),
        "inpaint_mask": load_mask(item.item_dir / "mask.png", size),
        "control_normal": pil_to_tensor(load_image(item.item_dir / "normal.png", size)),
        "control_depth_sil": pil_to_tensor(load_image(item.item_dir / "depth_sil.png", size)),
        "appearance_ref": pil_to_tensor(load_image(item.appearance_ref, size)),
    }
    if out["inpaint_mask"].sum() == 0:
        raise ValueError(f"Empty inpaint mask: {item.item_dir}")
    return out


def discover_synth_items(synth_root: str | Path, views: tuple[int, ...] = (0, 90, 180, 270)) -> list[TrainItem]:
    """All (sample, view) items from the synthetic root; an item with missing files is NOT skipped SILENTLY."""
    synth_root = Path(synth_root)
    items, broken = [], []
    for sd in sorted(synth_root.glob("s*_*/")):
        ref = sd / "appearance_ref.png"
        for v in views:
            it = TrainItem(sd / f"view_{v:03d}", ref)
            (items if it.validate() else broken).append(it)
    if broken:
        raise ValueError(f"{len(broken)} broken synthetic items (first: {broken[0].item_dir}) — data generation is incomplete")
    return items


def discover_flat_items(root: str | Path) -> list[TrainItem]:
    """Flat layout (the VITON-HD preprocessing output): {root}/{item_id}/*.png (+appearance_ref.png)."""
    root = Path(root)
    items, broken = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        it = TrainItem(d, d / "appearance_ref.png")
        (items if it.validate() else broken).append(it)
    if broken:
        raise ValueError(f"{len(broken)} broken items (first: {broken[0].item_dir})")
    return items
