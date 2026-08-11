#!/usr/bin/env python3
"""BACKFILL the geo_iou aux files for the baseline predictions + compare with a trained ckpt.

Why: zero_shot_baseline.py only writes .person/.mask/.ref, while harness.evaluate's geo_iou
needs .predsil + .sil (harness.py::evaluate_item). Because of that the question "did training
beat the baseline?" was unmeasurable — the ablation gate only compares control ON/OFF, not
trained vs untrained.

This script does not regenerate the EXISTING baseline PNGs (no diffusion — only HMR2 + parser
run on the GPU). It just writes the missing aux files and refreshes the report.

Usage (Colab, A100):
  python v2/scripts/backfill_baseline_geo.py --idm-repo /content/IDM-VTON
  [--variant fill_spatial] [--limit N] [--angles 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from meshvton2.conditioning.body import build_hmr2_backend  # noqa: E402
from meshvton2.conditioning.builder import OrbitView, PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval import harness  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402

from eval_checkpoint import _save_predsil  # noqa: E402  (the same parser path — consistency is mandatory)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--variant", default="fill_spatial")
    ap.add_argument("--limit", type=int, default=0, help="0 = all combos")
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]
    pred_dir = out_root / f"phase1_{args.variant}" / "preds"
    if not pred_dir.exists():
        raise SystemExit(f"ERROR: no baseline predictions: {pred_dir}")

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    combos = manifest.combos[: args.limit] if args.limit else manifest.combos

    # prep+HMR2 per person, asset per garment — no repeats inside the combo loop
    person_cache: dict[str, tuple] = {}
    asset_cache: dict[str, object] = {}
    done = skipped = missing = 0

    for i, combo in enumerate(combos):
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        stems = {
            a: (pred_dir / cfg["pred_pattern"].format(
                person_id=person.id, garment_id=garment.id, angle=a)).with_suffix("")
            for a in args.angles
        }
        # There is no point producing aux files without a prediction PNG
        todo = {a: s for a, s in stems.items() if s.with_suffix(".png").exists()}
        missing += len(stems) - len(todo)
        if not todo:
            continue
        # Skip this combo if both aux files exist (resume-safe)
        if all(Path(f"{s}.sil.png").exists() and Path(f"{s}.predsil.png").exists()
               for s in todo.values()):
            skipped += len(todo)
            print(f"[{i+1}/{len(combos)}] skipped (aux complete) {person.id}×{garment.id}")
            continue

        try:
            if person.id not in person_cache:
                pp = prep.process(manifest.root / person.image, size=size)
                # person_square_bbox is MANDATORY — a full-frame bbox shifts the mesh for an off-centre person
                person_cache[person.id] = (pp, hmr2(pp.image, bbox=person_square_bbox(pp)))
            pp, params = person_cache[person.id]
            if garment.id not in asset_cache:
                # RAW asset — the builder already applies force_textureless (builder.py:316,341);
                # applying it again here risks diverging from the eval_checkpoint path
                asset_cache[garment.id] = load_garment_asset(
                    garments_root / garment.mesh,
                    texture_path=garments_root / garment.texture if garment.texture else None,
                    garment_id=garment.id,
                    allow_untextured=True,
                )
            asset = asset_cache[garment.id]

            for angle, stem in todo.items():
                bundle = build_conditioning(
                    pp.image, params, asset,
                    PhotoView() if angle == 0 else OrbitView(angle),
                    size=size, person_prep=pp,
                )
                # TARGET silhouette — bit-identical source to eval_checkpoint.run_variant
                sil = (bundle.control_depth_sil[2].numpy() > 0).astype("uint8") * 255
                Image.fromarray(sil).save(f"{stem}.sil.png")
                _save_predsil(prep, stem)  # the garment region of the GENERATED image (parser)
                done += 1
            print(f"[{i+1}/{len(combos)}] OK {person.id}×{garment.id}")
        except Exception as e:
            print(f"[{i+1}/{len(combos)}] ERROR {person.id}×{garment.id}: {e}", file=sys.stderr)

    print(f"\naux written={done} skipped={skipped} no-prediction={missing}")

    # Refresh the report + compare against the trained checkpoint
    summary = harness.evaluate(manifest, pred_dir, cfg["pred_pattern"])
    report = out_root / f"phase1_{args.variant}.json"
    harness.write_report(summary, report)
    print(f"report updated: {report}")

    get = lambda s, k: (s["overall"].get(k) or {}).get("mean")
    rows = {(r["person"], r["garment"], r["angle"]): r for r in summary["rows"]}
    print("\n=== TRAINING vs BASELINE (geo_iou, higher is better) ===")
    base_all = get(summary, "geo_iou")
    print(f"baseline (all {summary['found']} combos): {base_all}")
    for tag in ("control_on", "control_off"):
        ck = out_root / f"ckpt_{tag}.json"
        if not ck.exists():
            print(f"{tag}: no report ({ck})")
            continue
        c = json.loads(ck.read_text())
        crows = {(r["person"], r["garment"], r["angle"]): r for r in c["rows"]}
        shared = sorted(set(rows) & set(crows))
        bv = [rows[k]["geo_iou"] for k in shared if rows[k].get("geo_iou") is not None]
        cv = [crows[k]["geo_iou"] for k in shared if crows[k].get("geo_iou") is not None]
        if len(bv) == len(cv) == len(shared) and shared:
            b_m, c_m = float(np.mean(bv)), float(np.mean(cv))
            print(f"{tag}: {len(shared)} shared combos — baseline={b_m:.4f} ckpt={c_m:.4f} "
                  f"Δ={c_m - b_m:+.4f}  {'TRAINING BETTER' if c_m > b_m else 'TRAINING WORSE'}")
        else:
            print(f"{tag}: no shared combos or geo_iou missing (baseline {len(bv)}/{len(shared)}, "
                  f"ckpt {len(cv)}/{len(shared)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
