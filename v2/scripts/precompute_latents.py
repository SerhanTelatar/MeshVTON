#!/usr/bin/env python3
"""VAE latent ön-hesabı — eğitim adımından VAE'yi ve PNG çözmeyi çıkarır.

Her eğitim öğesi için 5 görüntünün FLUX-VAE latent'i bir kez hesaplanıp
{item_dir}/latents.pt olarak yazılır (bf16, CPU):
    gt_lat, masked_lat (Fill sözleşmesi: maske içi SİYAH), normal_lat,
    depth_sil_lat, ref_lat
Sonuç eğitimle BİREBİR aynıdır (aynı VAE, aynı ölçekleme) — saf hız kazancı.
Resume-safe: mevcut latents.pt atlanır. A100'de ~10k öğe ≈ 20-30 dk.

Kullanım: python v2/scripts/precompute_latents.py [--batch 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import torch  # noqa: E402
import yaml  # noqa: E402

from meshvton2.data.items import discover_flat_items, discover_synth_items, load_item  # noqa: E402
from meshvton2.model.flux_tryon import mask_image_for_fill  # noqa: E402

KEYS = ("gt_lat", "masked_lat", "normal_lat", "depth_sil_lat", "ref_lat")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=8, help="öğe başına 5 görüntü; VAE batch = 5*bu")
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    repo = args.repo or base["model"]["flux_fill_repo"]

    items = []
    synth_root = REPO / base["paths"]["synth_root"]
    real_root = REPO / "v2/data/vitonhd_items"
    if synth_root.exists():
        items += discover_synth_items(synth_root)
    if real_root.exists():
        items += discover_flat_items(real_root)
    todo = [it for it in items if not (it.item_dir / "latents.pt").exists()]
    print(f"toplam öğe: {len(items)}, hesaplanacak: {len(todo)}")
    if not todo:
        return 0

    from diffusers import AutoencoderKL

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dtype).to("cuda").eval()
    scale, shift = vae.config.scaling_factor, vae.config.shift_factor

    @torch.no_grad()
    def encode(imgs: torch.Tensor) -> torch.Tensor:  # (N,3,H,W) [-1,1] -> (N,16,h,w)
        z = vae.encode(imgs.to("cuda", dtype)).latent_dist.sample()
        return ((z - shift) * scale).to(torch.bfloat16).cpu()

    done = 0
    for start in range(0, len(todo), args.batch):
        chunk = todo[start : start + args.batch]
        pixel_stack, metas = [], []
        for it in chunk:
            d = load_item(it, size=size)
            masked = mask_image_for_fill(d["agnostic_rgb"].unsqueeze(0), d["inpaint_mask"].unsqueeze(0))[0]
            pixel_stack += [d["gt_rgb"], masked, d["control_normal"], d["control_depth_sil"], d["appearance_ref"]]
            metas.append(it)
        lats = encode(torch.stack(pixel_stack))
        for i, it in enumerate(metas):
            group = lats[i * 5 : (i + 1) * 5]
            torch.save(dict(zip(KEYS, group)), it.item_dir / "latents.pt")
        done += len(chunk)
        if done % 400 < args.batch:
            print(f"[{done}/{len(todo)}]", flush=True)
    print(f"bitti: {done} öğe için latent yazıldı")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
