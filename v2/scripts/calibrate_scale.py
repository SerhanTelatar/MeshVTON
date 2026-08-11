#!/usr/bin/env python3
"""Empirically calibrate the garment SCALE on the photo path (twin of the hang calibration).

Finding (2026-08-10): even after the hang was fixed (hang_pad=-0.12) the garment looks TOO
WIDE relative to the body — a poncho-like silhouette spilling over the arms (e.g. 00737_00).
`_prealign_garment` deliberately does no scaling at all (CLOTH3D garments are in metre scale
and modelled against the SMPL body). But on the photo path the body comes from HMR2; that
body does not have to match the body CLOTH3D assumed.

Method: sweep garment_scale and, at each value, measure the IoU between the mesh garment
silhouette and the REAL garment mask found by the parser. NO diffusion.

Reading:
  - If the peak is clearly BELOW 1.0 the garment really is too big; use that value.
  - If the peak is around 1.0 the scale is not to blame, the width gap comes from cut/drape.

Usage:
  python v2/scripts/calibrate_scale.py --idm-repo /content/IDM-VTON
  [--hang-pad -0.12] [--range 0.70 1.15 0.05] [--garment upper_body__00047_Top]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from meshvton2.conditioning.body import build_hmr2_backend  # noqa: E402
from meshvton2.conditioning.builder import (  # noqa: E402
    PHOTO_HANG_PAD, PhotoView, assert_real_impl, build_conditioning)
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402

ATR_GARMENT = (4, 7)
# The people validate_camera.py verified (calibrate with those whose alignment is proven)
DEFAULT_PERSONS = ("00000_00", "02935_00", "01455_00", "00737_00", "02199_00")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--persons", nargs="+", default=list(DEFAULT_PERSONS))
    ap.add_argument("--garment", default=None, help="default: the first garment in the manifest")
    ap.add_argument("--hang-pad", type=float, default=PHOTO_HANG_PAD)
    ap.add_argument("--range", nargs=3, type=float, default=[0.70, 1.15, 0.05],
                    metavar=("START", "END", "STEP"))
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]

    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    gid = args.garment or manifest.combos[0].garment_id
    garment = by_gid[gid]
    asset = load_garment_asset(
        garments_root / garment.mesh,
        texture_path=garments_root / garment.texture if garment.texture else None,
        garment_id=garment.id, allow_untextured=True,
    )

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()

    people = []
    for pid in args.persons:
        if pid not in by_pid:
            print(f"SKIP {pid}: not in the manifest", file=sys.stderr)
            continue
        pp = prep.process(manifest.root / by_pid[pid].image, size=size)
        params = hmr2(pp.image, bbox=person_square_bbox(pp))
        worn = np.isin(cv2.resize(np.asarray(pp.parse), (size[1], size[0]),
                                  interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
        people.append((pid, pp, params, worn))
    if not people:
        raise SystemExit("ERROR: no person could be prepared")

    lo, hi, step = args.range
    scales = np.arange(lo, hi + 1e-9, step)
    # TARGET area: the real garment found by the parser. silhouette% should peak around it —
    # too small and the overlap is short, too big and the union inflates; IoU drops at both ends.
    worn_pct = float(np.mean([w.mean() for *_, w in people])) * 100
    print(f"garment: {gid} | people: {len(people)} | hang_pad={args.hang_pad:+.2f} | "
          f"scale sweep: {lo:.2f} → {hi:.2f} step {step:.2f}")
    print(f"TARGET: real garment area (parser) = {worn_pct:.1f}% — silhouette% should approach it\n")
    print(f"{'scale':>7} {'mean IoU':>8} {'sil%':>8}   " + "  ".join(f"{p:>9}" for p, *_ in people))

    rows = []
    for sc in scales:
        ious, areas = [], []
        for pid, pp, params, worn in people:
            b = build_conditioning(pp.image, params, asset, PhotoView(),
                                   size=size, person_prep=pp,
                                   hang_pad=args.hang_pad, garment_scale=float(sc))
            sil = b.control_depth_sil[2].numpy() > 0
            union = (sil | worn).sum()
            ious.append(float((sil & worn).sum() / union) if union else 0.0)
            areas.append(float(sil.mean()))
        mean = float(np.mean(ious))
        rows.append((float(sc), mean, ious))
        print(f"{sc:>7.2f} {mean:>8.3f} {np.mean(areas)*100:>7.1f}%   "
              + "  ".join(f"{v:>9.3f}" for v in ious))

    best_sc, best_iou, _ = max(rows, key=lambda r: r[1])
    cur = next((r for r in rows if abs(r[0] - 1.0) < 1e-6), None)
    print(f"\nBEST: scale={best_sc:.2f} → mean IoU={best_iou:.3f}")
    # If the peak sits at an EDGE of the range the optimum is outside — do not use the number, widen the range
    if abs(best_sc - rows[-1][0]) < 1e-6 or abs(best_sc - rows[0][0]) < 1e-6:
        print(f"WARNING: the peak is at the sweep BOUNDARY ({best_sc:.2f}) — the true optimum is OUTSIDE the range.")
        print(f"  Widen the range and re-run, e.g.: --range {best_sc:.2f} {best_sc + 0.5:.2f} 0.05")
    if cur:
        print(f"CURRENT: scale=1.00 → mean IoU={cur[1]:.3f}  (difference {best_iou - cur[1]:+.3f})")
    if cur and best_iou - cur[1] < 0.02:
        print("\nCONCLUSION: the scale is NOT to blame — 1.00 is already near the peak. The width")
        print("  difference comes from cut/drape; do not change garment_scale.")
    else:
        print(f"\nCONCLUSION: the scale matters — use garment_scale={best_sc:.2f} on the photo path.")
        print("  CAREFUL: DO NOT TOUCH the SYNTHETIC path (there both body and garment come from the")
        print("  same SMPL-X; rescaling has always been a mistake there — see _prealign_garment).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())