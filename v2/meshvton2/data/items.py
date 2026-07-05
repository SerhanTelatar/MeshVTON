"""Eğitim örneği dizin sözleşmesi — TEK format, iki kaynak:

    {item_dir}/gt.png agnostic.png mask.png normal.png depth_sil.png [appearance_ref.png]

- Sentetik: synth/generate.py `{sample}/view_{deg}/` altına yazar; appearance_ref
  örnek kökünde (görüşler paylaşır).
- Gerçek (VITON-HD): scripts/preprocess_vitonhd.py aynı düzeni kişi×giysi başına
  yazar (gt = kişinin kendi fotoğrafı; paired eğitim).

Dataset'ler bu sözleşmeyi okur — kaynak ayrımı yoktur (mixed eğitim bedava).
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
    appearance_ref: Path  # görüşler arası paylaşılabilir → ayrı yol

    def validate(self) -> bool:
        return all((self.item_dir / f"{f}.png").exists() for f in FILES) and self.appearance_ref.exists()


def load_item(item: TrainItem, size: tuple[int, int] | None = None, use_latents: bool = False) -> dict[str, torch.Tensor]:
    """-> ConditioningBundle alan adlarıyla uyumlu tensör sözlüğü, hepsi [-1,1]/({0,1} maske).

    use_latents=True: precompute_latents.py çıktısını okur (gt_lat/masked_lat/
    normal_lat/depth_sil_lat/ref_lat + inpaint_mask) — eğitim adımında VAE yok.
    Latent dosyası eksikse HATA (karışık batch collate'i kırar; sessiz fallback yok)."""
    if use_latents:
        lat_path = item.item_dir / "latents.pt"
        if not lat_path.exists():
            raise FileNotFoundError(
                f"{lat_path} yok — önce: python v2/scripts/precompute_latents.py"
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
        raise ValueError(f"Boş inpaint maskesi: {item.item_dir}")
    return out


def discover_synth_items(synth_root: str | Path, views: tuple[int, ...] = (0, 90, 180, 270)) -> list[TrainItem]:
    """Sentetik kökten tüm (örnek, görüş) öğeleri; eksik dosyalı öğe SESSİZCE atlanmaz."""
    synth_root = Path(synth_root)
    items, broken = [], []
    for sd in sorted(synth_root.glob("s*_*/")):
        ref = sd / "appearance_ref.png"
        for v in views:
            it = TrainItem(sd / f"view_{v:03d}", ref)
            (items if it.validate() else broken).append(it)
    if broken:
        raise ValueError(f"{len(broken)} bozuk sentetik öğe (ilk: {broken[0].item_dir}) — veri üretimi eksik")
    return items


def discover_flat_items(root: str | Path) -> list[TrainItem]:
    """Düz düzen (VITON-HD ön-işleme çıktısı): {root}/{item_id}/*.png (+appearance_ref.png)."""
    root = Path(root)
    items, broken = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        it = TrainItem(d, d / "appearance_ref.png")
        (items if it.validate() else broken).append(it)
    if broken:
        raise ValueError(f"{len(broken)} bozuk öğe (ilk: {broken[0].item_dir})")
    return items
