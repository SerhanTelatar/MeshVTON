#!/usr/bin/env python3
"""Mesh SPECIFICITY test — is the output really specific to the GIVEN mesh?

geo_iou says "did the output match the target silhouette", but on its own it is misleading:
even if the model produced the SAME plausible shirt for every garment, the mean IoU could be
high. The control_on/control_off gap measures the conditioning's STRENGTH, not its SPECIFICITY.

This test builds an N×N matrix per person: the output silhouette (predsil) of garment i vs
the target silhouette (sil) of garment j.
  diagonal     = the correct mesh match
  off-diagonal = the wrong mesh
If the diagonal is clearly higher, the output is mesh-specific (the thesis holds). If the two
are close, the model produces the same thing whichever mesh it is given.

NO GPU/diffusion — it only re-scores the existing .predsil.png / .sil.png files.

Usage:
  python v2/scripts/mesh_specificity.py            # every pred set it finds
  python v2/scripts/mesh_specificity.py --sets ckpt_control_on --angles 0
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from meshvton2.eval import metrics as M  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.eval.harness import _load_mask  # noqa: E402

# Root folders of the pred sets: eval_results/<name>/preds
DEFAULT_SETS = ("phase1_fill_spatial", "ckpt_control_on", "ckpt_control_off")


def analyse(pred_dir: Path, manifest, pattern: str, angles) -> dict | None:
    """Builds a garment×garment IoU matrix per person and returns the diagonal/off-diagonal means."""
    by_person: dict[str, list] = {}
    for combo in manifest.combos:
        by_person.setdefault(combo.person_id, []).append(combo.garment_id)

    diag_all: list[float] = []
    off_all: list[float] = []
    per_person: list[tuple[str, float, float, int]] = []

    for person_id, garment_ids in by_person.items():
        # Do not count the same garment twice (the manifest may contain duplicates)
        garments = list(dict.fromkeys(garment_ids))
        for angle in angles:
            stem = lambda g: (pred_dir / pattern.format(
                person_id=person_id, garment_id=g, angle=angle)).with_suffix("")
            predsil = {g: Path(f"{stem(g)}.predsil.png") for g in garments}
            sil = {g: Path(f"{stem(g)}.sil.png") for g in garments}
            usable = [g for g in garments if predsil[g].exists() and sil[g].exists()]
            if len(usable) < 2:  # a matrix needs at least 2 garments
                continue

            d, o = [], []
            for gi, gj in product(usable, usable):
                # predsil and sil come from the same generation → equal size; still, fit sil to predsil
                a = _load_mask(predsil[gi])
                b = _load_mask(sil[gj], size=a.shape)
                iou = M.silhouette_iou(a, b)
                if iou is None:
                    continue
                (d if gi == gj else o).append(iou)
            if d and o:
                diag_all += d
                off_all += o
                per_person.append((person_id, float(np.mean(d)), float(np.mean(o)), len(usable)))

    if not diag_all or not off_all:
        return None
    return {
        "diag": float(np.mean(diag_all)),
        "off": float(np.mean(off_all)),
        "n_diag": len(diag_all),
        "n_off": len(off_all),
        "per_person": per_person,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    ap.add_argument("--per-person", action="store_true", help="also print the per-person breakdown")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    out_root = REPO / base["paths"]["eval_results"]

    print("=== MESH SPECIFICITY (diagonal = correct mesh, higher is better) ===\n")
    any_done = False
    for name in args.sets:
        pred_dir = out_root / name / "preds"
        if not pred_dir.exists():
            print(f"{name:22s} no preds folder — skipped")
            continue
        res = analyse(pred_dir, manifest, cfg["pred_pattern"], args.angles)
        if res is None:
            print(f"{name:22s} not enough .predsil/.sil — skipped "
                  f"(run backfill_baseline_geo.py first)")
            continue
        any_done = True
        gap = res["diag"] - res["off"]
        rel = gap / res["off"] * 100 if res["off"] else float("nan")
        print(f"{name}")
        print(f"  correct mesh (diagonal)     = {res['diag']:.4f}  (n={res['n_diag']})")
        print(f"  wrong mesh   (off-diagonal) = {res['off']:.4f}  (n={res['n_off']})")
        print(f"  SPECIFICITY                 = {gap:+.4f}  ({rel:+.1f}%)")
        if args.per_person:
            for pid, d, o, k in sorted(res["per_person"]):
                print(f"    {pid:12s} {k} garments: correct={d:.4f} wrong={o:.4f} diff={d - o:+.4f}")
        print()

    if any_done:
        print("READING: if the difference is ~0 the model produces the same thing whichever mesh it gets —")
        print("         the improvement in geo_iou is not mesh-SPECIFIC, it is distribution fit.")
        print("         Use the difference on the baseline set as the anchor (untrained model).")
    return 0 if any_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
