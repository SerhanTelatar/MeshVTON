#!/usr/bin/env python3
"""GEOMETRIC UPPER BOUND figure — a direct composite of the 3D render (NO diffusion).

Purpose: to visualize the question "how much geometric information does the conditioning carry?".
The draped garment's NORMAL map (control_normal) is shaded with a simple Lambert light and
pasted inside the depth-tested silhouette (control_depth_sil[2]). The result: a crisp,
folded, shaded grey garment on top of the photo — the geometric ceiling the model could
reach.

IMPORTANT, present it this way in the thesis: this is NOT a generation, it is a 3D composite.
It is placed next to the model output and read like this:
  - composite: the geometry is flawless, but the OLD GARMENT keeps showing at the edges
    (the composite cannot remove it) and the lighting does not match the photo;
  - model output: it REMOVES the old garment and matches the lighting, but cannot produce folds.
This contrast is direct evidence of why the generative step is necessary.

Diffusion is not run; if model outputs exist they are READ from the existing preds folder.

Usage:
  python v2/scripts/render_composite.py --idm-repo /content/IDM-VTON
  [--limit 5] [--pred-set ckpt_control_on_hp-012_sc125]
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

# Camera-space light direction (+z towards the camera), from the upper left. Chosen BY
# MEASUREMENT: the garment surface mostly faces the camera and folds deviate ~25° around
# (0,0,1); readability grows with the TANGENTIAL component. Intensity std in that 25° regime:
#   (-0.35,0.45,0.82) tangent 0.57 → std 0.072, grey 190-248 (nearly flat white)
#   (-0.60,0.60,0.53) tangent 0.85 → std 0.105, grey 136-222  ← chosen
#   (-0.65,0.70,0.30) tangent 0.95 → std 0.112 but the garment darkens noticeably
LIGHT = np.array([-0.60, 0.60, 0.53])
LIGHT /= np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE = 0.42, 0.58  # without an ambient floor the shadows clog and fall to black


def shade(normal_chw: np.ndarray, sil: np.ndarray) -> tuple[np.ndarray, dict]:
    """(3,H,W) [-1,1] normals → (H,W) [0,1] Lambert intensity + diagnostics.

    AUTOMATIC z-SIGN: because of the camera convention (+y down), the n_z sign of
    camera-facing surfaces is not fixed. With the wrong sign n·L goes negative
    everywhere, clip(0) drops every pixel to ambient and the composite comes out
    FLAT DARK (exactly what happened on the first run). We look at the mean n_z
    inside the silhouette and flip the light's z component accordingly.
    """
    n = normal_chw.transpose(1, 2, 0).astype(np.float64)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
    nz = float(n[sil][:, 2].mean()) if sil.any() else 0.0
    L = LIGHT.copy()
    if nz < 0:
        L[2] = -L[2]
    inten = AMBIENT + DIFFUSE * np.clip(n @ L, 0.0, 1.0)
    vals = inten[sil] if sil.any() else np.array([0.0])
    diag = {"nz": nz, "z_flipped": nz < 0,
            "grey_p5": int(np.percentile(vals, 5) * 255),
            "grey_p95": int(np.percentile(vals, 95) * 255),
            "std": float(vals.std())}
    return inten, diag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5, help="first N combos")
    ap.add_argument("--pred-set", default="ckpt_control_on_hp-012_sc125",
                    help="eval_results subfolder the model outputs are read from")
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]
    pred_dir = out_root / args.pred_set / "preds"
    out_dir = out_root / "render_composite"
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}

    person_cache: dict[str, tuple] = {}
    panels_all = []
    for combo in manifest.combos[: args.limit]:
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        if person.id not in person_cache:
            pp = prep.process(manifest.root / person.image, size=size)
            person_cache[person.id] = (pp, hmr2(pp.image, bbox=person_square_bbox(pp)))
        pp, params = person_cache[person.id]
        asset = load_garment_asset(
            garments_root / garment.mesh,
            texture_path=garments_root / garment.texture if garment.texture else None,
            garment_id=garment.id, allow_untextured=True)
        b = build_conditioning(pp.image, params, asset, PhotoView(), size=size,
                               person_prep=pp, hang_pad=PHOTO_HANG_PAD,
                               garment_scale=PHOTO_GARMENT_SCALE)

        sil = b.control_depth_sil[2].numpy() > 0
        inten, d = shade(b.control_normal.numpy(), sil)
        comp = pp.image.copy()
        comp[sil] = np.clip(inten[sil, None] * 255.0, 0, 255).astype(np.uint8)
        # Silhouette HOLE ratio: if the draped garment sinks into the body, the depth test
        # cuts pieces out → the old garment leaks through the middle of the composite (a real
        # finding, not an artifact). Solid body = the area after morphological closing.
        import cv2

        k = np.ones((max(size[1] // 40, 3),) * 2, np.uint8)
        closed = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        hole_ratio = 1.0 - sil.sum() / max(closed.sum(), 1)
        Image.fromarray(comp).save(out_dir / f"{person.id}__{garment.id}.composite.png")

        row = [("person", pp.image), ("3D RENDER COMPOSITE (upper bound)", comp)]
        p = pred_dir / cfg["pred_pattern"].format(
            person_id=person.id, garment_id=garment.id, angle=0)
        if p.exists():
            row.append(("model output", np.asarray(Image.open(p).convert("RGB"))))
        else:
            print(f"  NOTE: no model output ({p.name}) — that panel was skipped")
        panels_all.append((f"{person.id} × {garment.id}", row))
        print(f"OK {person.id} × {garment.id}  |  n_z={d['nz']:+.2f}"
              f"{' (light z FLIPPED)' if d['z_flipped'] else ''}"
              f"  grey {d['grey_p5']}-{d['grey_p95']} std={d['std']:.3f}"
              f"  silhouette hole ratio={hole_ratio*100:.1f}%")

    TH = 230
    hh = round(TH * size[0] / size[1])
    ncol = max(len(r) for _, r in panels_all)
    grid = Image.new("RGB", (TH * ncol, hh * len(panels_all)), "white")
    for r, (_, row) in enumerate(panels_all):
        for c, (_, a) in enumerate(row):
            grid.paste(Image.fromarray(a).convert("RGB").resize((TH, hh), Image.LANCZOS),
                       (c * TH, r * hh))
    sp = out_dir / "upper_bound_grid.png"
    grid.save(sp)
    print(f"\nGrid: {sp}")
    print("columns: " + " | ".join(l for l, _ in panels_all[0][1]))
    print("\nREADING: in the composite the garment is crisp and folded but the OLD GARMENT leaks at")
    print("         the edges and the light does not match the photo; the model output removes the")
    print("         old one and matches the light but cannot produce folds — the case for the generative step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())