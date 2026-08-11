#!/usr/bin/env python3
"""Empirically calibrate the garment hang height on the PHOTO path.

Finding (2026-08-10): for people that passed the camera gate (IoU 0.82) the body projects
correctly, but the garment mesh sits ~21 points HIGHER than the real garment (it rides over
the face). The synthetic path does not have this problem — the two paths contradict each
other, so instead of arguing about hang_pad we MEASURE it.

Method: sweep hang_pad and, at each value, compute the IoU between the mesh garment
silhouette and the REAL garment mask found by the parser. NO diffusion — only HMR2 +
parser + render.

Reading:
  - If the best IoU is HIGH (>0.5) and the peak is clear: the problem is a simple vertical
    offset, use that hang_pad.
  - If the best IoU is LOW at every value: the problem is not the offset (scale/pose/binding) —
    changing hang_pad will not fix it, you need to look deeper.

Usage:
  python v2/scripts/calibrate_hang.py --idm-repo /content/IDM-VTON
  [--persons 00000_00 02935_00] [--garment upper_body__00047_Top]
  [--range -0.30 0.12 0.03]
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
from meshvton2.conditioning.builder import PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402

ATR_GARMENT = (4, 7)  # upper body labels in the parser
# The people validate_camera.py verified (calibrate with those whose alignment is proven)
DEFAULT_PERSONS = ("00000_00", "02935_00", "01455_00", "00737_00", "02199_00")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--persons", nargs="+", default=list(DEFAULT_PERSONS))
    ap.add_argument("--garment", default=None, help="default: the first garment in the manifest")
    ap.add_argument("--range", nargs=3, type=float, default=[-0.30, 0.12, 0.03],
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

    # Person preprocessing + HMR2 ONCE (the sweep only changes hang_pad)
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
    pads = np.arange(lo, hi + 1e-9, step)
    print(f"garment: {gid} | people: {len(people)} | hang_pad sweep: "
          f"{lo:+.2f} → {hi:+.2f} step {step:.2f}\n")
    print(f"{'hang_pad':>9} {'mean IoU':>8}   " + "  ".join(f"{p:>9}" for p, *_ in people))

    rows = []
    for pad in pads:
        ious = []
        for pid, pp, params, worn in people:
            b = build_conditioning(pp.image, params, asset, PhotoView(),
                                   size=size, person_prep=pp, hang_pad=float(pad))
            sil = b.control_depth_sil[2].numpy() > 0
            union = (sil | worn).sum()
            ious.append(float((sil & worn).sum() / union) if union else 0.0)
        mean = float(np.mean(ious))
        rows.append((float(pad), mean, ious))
        print(f"{pad:>+9.2f} {mean:>8.3f}   " + "  ".join(f"{v:>9.3f}" for v in ious))

    best_pad, best_iou, _ = max(rows, key=lambda r: r[1])
    cur = next((r for r in rows if abs(r[0] - 0.06) < 1e-6), None)
    print(f"\nBEST: hang_pad={best_pad:+.2f} → mean IoU={best_iou:.3f}")
    if cur:
        print(f"CURRENT: hang_pad=+0.06 → mean IoU={cur[1]:.3f}  (difference {best_iou - cur[1]:+.3f})")
    if best_iou < 0.35:
        print("\nCONCLUSION: no offset gives adequate alignment — the problem is NOT a vertical shift.")
        print("  Look at scale/pose/binding instead; changing hang_pad will not fix it.")
    else:
        print("\nCONCLUSION: the peak is clear — the problem is a vertical offset. Use this value on the photo path.")
        print("  CAREFUL: the synthetic path passed QA with +0.06 (builder.py:36-43).")
        print("  Re-run the synthetic QA before changing the value GLOBALLY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
