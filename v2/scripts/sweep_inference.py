#!/usr/bin/env python3
"""Inference lever sweep — looking for quality gains WITHOUT training.

Context (2026-08-11): the output is in the right place/size but flat and semi-transparent.
Ruled out: the training budget (the 4000-step run was washed out too, [[meshvton-v2-plan]]),
guidance (no effect between 1.0-15.0), compositing (the raw output is flat too), camera, alignment.

Two levers remain that have NEVER been tried:

1. control_scale > 1.0 — so far only 0.0 (ablation) and 1.0 (the training value) have been
   tried. The fold information is already in the input (control_normal is the draped garment's
   normal map); the model may not be using it, so amplifying the signal may help.

2. geometry_mask=False — the mask is currently parse ∪ dilate(silhouette) and reaches 48%; the
   model is forced to REGENERATE the arms/shoulders/neck (the skin problem comes from there).
   With False the mask is only the parser garment → skin and arms are preserved FROM THE PHOTO.
   The cost: a new garment wider than the old one cannot spill outside the mask, it gets clipped.

Diffusion does run, but the sweep is small (number of levers × variants). NO training.

Usage:
  python v2/scripts/sweep_inference.py --checkpoint <Drive>/stage1/final.pt \\
      --idm-repo /content/IDM-VTON [--control-scale 1.0 1.5 2.0 3.0]
      [--person 00000_00] [--garment upper_body__00047_Top] [--steps 28]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from meshvton2.conditioning.body import build_hmr2_backend  # noqa: E402
from meshvton2.conditioning.builder import (  # noqa: E402
    PHOTO_GARMENT_SCALE, PHOTO_HANG_PAD, PhotoView, assert_real_impl, build_conditioning)
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.model.flux_tryon import FluxTryOnSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--person", default=None)
    ap.add_argument("--garment", default=None)
    ap.add_argument("--control-scale", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_dir = REPO / base["paths"]["eval_results"] / "inference_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    pid = args.person or manifest.combos[0].person_id
    gid = args.garment or manifest.combos[0].garment_id
    person, garment = by_pid[pid], by_gid[gid]

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    pp = prep.process(manifest.root / person.image, size=size)
    params = hmr2(pp.image, bbox=person_square_bbox(pp))
    asset = load_garment_asset(
        garments_root / garment.mesh,
        texture_path=garments_root / garment.texture if garment.texture else None,
        garment_id=garment.id, allow_untextured=True,
    )
    sampler = FluxTryOnSampler(base["model"]["flux_fill_repo"], checkpoint=args.checkpoint,
                               prompt=base["model"]["prompt"])

    # Conditioning is built SEPARATELY for the two mask regimes (the mask is decided inside build_conditioning)
    bundles = {}
    for gm in (True, False):
        bundles[gm] = build_conditioning(
            pp.image, params, asset, PhotoView(), size=size, person_prep=pp,
            hang_pad=PHOTO_HANG_PAD, garment_scale=PHOTO_GARMENT_SCALE, geometry_mask=gm)

    print(f"person={pid} garment={gid} steps={args.steps} seed={args.seed}")
    for gm, b in bundles.items():
        print(f"  geometry_mask={gm}: mask area = {(b.inpaint_mask.numpy()[0] > 0.5).mean()*100:.1f}%")
    print()

    H, W = size
    TH = 210
    hh = round(TH * H / W)
    cols = 1 + len(args.control_scale)          # input + one per control_scale
    grid = Image.new("RGB", (TH * cols, hh * len(bundles)), "white")
    rows = []
    for r, (gm, b) in enumerate(bundles.items()):
        grid.paste(Image.fromarray(pp.image).resize((TH, hh), Image.LANCZOS), (0, r * hh))
        m = b.inpaint_mask.numpy()[0] > 0.5
        for c, cs in enumerate(args.control_scale):
            img = sampler.sample(b, steps=args.steps, seed=args.seed, control_scale=float(cs))
            Image.fromarray(img).save(out_dir / f"{pid}__{gid}__gm{int(gm)}_cs{cs:g}.png")
            # in-mask std: a texture/contrast proxy (the same measure as in the guidance sweep)
            std = float(np.asarray(img, np.float32)[m].std())
            rows.append((gm, cs, std))
            grid.paste(Image.fromarray(img).resize((TH, hh), Image.LANCZOS), ((c + 1) * TH, r * hh))
            print(f"  geometry_mask={gm} control_scale={cs:>4.1f} → in-mask std={std:6.2f}")

    sp = out_dir / f"{pid}__{gid}__sweep.png"
    grid.save(sp)
    print(f"\nGrid: {sp}")
    print(f"  rows: geometry_mask=True (top), False (bottom)")
    print(f"  columns: input, " + ", ".join(f"cs={c:g}" for c in args.control_scale))
    best = max(rows, key=lambda r: r[2])
    print(f"\nHIGHEST TEXTURE: geometry_mask={best[0]} control_scale={best[1]:g} → {best[2]:.2f}")
    print("NOTE: std is a PROXY (noise raises it too). Make the call BY LOOKING AT THE GRID:")
    print("      are the folds/edges getting sharper, or is it an artifact?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())