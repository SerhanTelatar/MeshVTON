#!/usr/bin/env python3
"""PLACEMENT metric — does the generated garment REALLY sit on the body?

Why it is needed (the 2026-08-10 finding): the harness's `geo_iou` compares the output
against the silhouette WE GAVE as conditioning. So what it measures is not "is the result
correct" but "did the model follow the silhouette I gave it". If we put the silhouette in
the wrong place (over the face) the model draws there and geo_iou still comes out high —
the metric CANNOT SEE that the target itself is broken. That is exactly why geo_iou dropped
while the hang fix (hang_pad -0.12) raised the mesh↔parser alignment from 0.295 to 0.480.

This script uses an independent reference: the parser mask of the real garment the person is
WEARING. How well does the generated garment region sit on it?
  placement_iou = IoU(predsil, parser_worn_mask)

It cannot reach 1.0 because of cut differences (we are dressing a different garment mesh), but
it is comparable ACROSS runs — which hang_pad places the garment more realistically.

GPU: parser only (no diffusion).

Usage:
  python v2/scripts/placement_iou.py --idm-repo /content/IDM-VTON \\
      [--sets ckpt_control_on ckpt_control_on_hp-012]
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
from PIL import Image  # noqa: E402

from meshvton2.conditioning.person import PersonPreprocessor  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.eval.harness import _load_mask  # noqa: E402

ATR_GARMENT = (4, 7)
DEFAULT_SETS = ("phase1_fill_spatial", "ckpt_control_on", "ckpt_control_off",
                "ckpt_control_on_hp-012", "ckpt_control_off_hp-012")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    out_root = REPO / base["paths"]["eval_results"]
    H, W = size

    prep = PersonPreprocessor(args.idm_repo)
    by_pid = {p.id: p for p in manifest.persons}

    # The REAL garment mask per person, computed once (so every set uses the same reference)
    worn: dict[str, np.ndarray] = {}

    def worn_mask(pid: str) -> np.ndarray | None:
        if pid not in worn:
            try:
                pp = prep.process(manifest.root / by_pid[pid].image, size=size)
                worn[pid] = np.isin(cv2.resize(np.asarray(pp.parse), (W, H),
                                               interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
            except Exception as e:
                print(f"  {pid}: parser failed — {e}", file=sys.stderr)
                worn[pid] = None
        return worn[pid]

    print("=== PLACEMENT IoU (generated garment vs the person's real garment) ===")
    print("    DIFFERENCE from geo_iou: the target is not the silhouette we supplied, it is an independent parser.\n")
    rows = []
    for name in args.sets:
        pred_dir = out_root / name / "preds"
        if not pred_dir.exists():
            print(f"{name:28s} no folder — skipped")
            continue
        vals = []
        for combo in manifest.combos:
            for angle in args.angles:
                stem = (pred_dir / cfg["pred_pattern"].format(
                    person_id=combo.person_id, garment_id=combo.garment_id,
                    angle=angle)).with_suffix("")
                ps = Path(f"{stem}.predsil.png")
                if not ps.exists():
                    continue
                w = worn_mask(combo.person_id)
                if w is None:
                    continue
                p = _load_mask(ps, size=w.shape)
                union = (p | w).sum()
                if union:
                    vals.append(float((p & w).sum() / union))
        if not vals:
            print(f"{name:28s} no .predsil — skipped")
            continue
        rows.append((name, float(np.mean(vals)), len(vals)))
        print(f"{name:28s} placement IoU = {np.mean(vals):.4f}  (n={len(vals)})")

    if len(rows) >= 2:
        best = max(rows, key=lambda r: r[1])
        print(f"\nBEST PLACEMENT: {best[0]} → {best[1]:.4f}")
        print("This metric tells you which conditioning places the garment REALISTICALLY;")
        print("geo_iou only measures the model's fidelity to the conditioning. If the two")
        print("disagree, look at THIS metric for visual correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
