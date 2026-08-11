#!/usr/bin/env python3
"""Phase 4 exit gate — golden set evaluation with a trained checkpoint + ablation.

It runs and compares two passes:
  1. control ON  (control_scale=1.0)
  2. control OFF (control_scale=0.0 — bit-identical stock behaviour thanks to zero-init)

GATE (the automated test of the v1 PHASE C lesson): if ON is worse than OFF, training did not
learn the geometry → do not move to Stage 2, diagnose it (first remedy: raise ref_dropout).

Usage (Colab, A100):
  python v2/scripts/eval_checkpoint.py --checkpoint <Drive>/stage1/latest-ckpt.pt \\
      --idm-repo /content/IDM-VTON [--limit 10] [--angles 0]
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
    PHOTO_GARMENT_SCALE, PHOTO_HANG_PAD, OrbitView, PhotoView, assert_real_impl,
    build_conditioning)
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval import harness  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.model.flux_tryon import FluxTryOnSampler  # noqa: E402
from meshvton2.utils.image_utils import tensor_to_pil  # noqa: E402


ATR_GARMENT = (4, 7)


def _save_predsil(prep, stem) -> None:
    """Extract the garment region of the GENERATED image with the parser → geo_iou's input.
    (Comparing the conditioning silhouette against the conditioning came out constant in the ablation — a bug.)"""
    import cv2

    try:
        pred_img = np.asarray(Image.open(stem.with_suffix(".png")).convert("RGB"))
        pp = prep.process(pred_img, size=pred_img.shape[:2])
        h, w = pred_img.shape[:2]
        m = np.isin(cv2.resize(pp.parse, (w, h), interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
        Image.fromarray((m * 255).astype("uint8")).save(f"{stem}.predsil.png")
    except Exception as e:
        print(f"  predsil skipped ({stem.name}): {e}", file=sys.stderr)


def run_variant(sampler, combos, ctx, control_scale: float, tag: str) -> dict:
    cfg, out_root, angles = ctx["cfg"], ctx["out_root"], ctx["angles"]
    prep = ctx["prep"]
    pred_dir = out_root / f"ckpt_{tag}" / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, (person, garment, bundle_by_angle) in enumerate(combos):
        for angle, bundle in bundle_by_angle.items():
            stem = (pred_dir / cfg["pred_pattern"].format(
                person_id=person.id, garment_id=garment.id, angle=angle)).with_suffix("")
            if not stem.with_suffix(".png").exists():
                pred = sampler.sample(bundle, control_scale=control_scale)
                Image.fromarray(pred).save(stem.with_suffix(".png"))
                tensor_to_pil(bundle.appearance_ref).save(f"{stem}.ref.png")
                Image.fromarray((bundle.inpaint_mask.numpy()[0] * 255).astype("uint8")).save(f"{stem}.mask.png")
                sil = (bundle.control_depth_sil[2].numpy() > 0).astype("uint8") * 255
                Image.fromarray(sil).save(f"{stem}.sil.png")
            if not Path(f"{stem}.predsil.png").exists():  # added to existing preds too (cheap)
                _save_predsil(prep, stem)
        print(f"[{tag} {i+1}/{len(combos)}] {person.id}×{garment.id}")
    summary = harness.evaluate(ctx["manifest"], pred_dir, cfg["pred_pattern"])
    harness.write_report(summary, out_root / f"ckpt_{tag}.json")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=10, help="first N combos")
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=28)
    # Hang pad: the calibrated value for the PHOTO path (builder.PHOTO_HANG_PAD).
    # The builder's +0.06 is for the SYNTHETIC path; on photos it hung the garment ~21 points
    # too high (mesh↔parser IoU 0.295 → 0.480 at -0.12, placement IoU 0.58 → 0.68).
    # To re-measure the old behaviour: --hang-pad 0.06
    ap.add_argument("--hang-pad", type=float, default=PHOTO_HANG_PAD,
                    help=f"garment hang pad [m]; default {PHOTO_HANG_PAD} (photo path calibration)")
    # Garment scale: CLOTH3D meshes are systematically small relative to HMR2 bodies
    # (calibrate_scale.py: mesh↔parser IoU 0.481 at 1.00, 0.555 at 1.25).
    ap.add_argument("--garment-scale", type=float, default=PHOTO_GARMENT_SCALE,
                    help=f"garment scale; default {PHOTO_GARMENT_SCALE} (photo path calibration)")
    # use_texture: False by default per the RULE. Its ONLY legitimate use is measuring the
    # 2026-07-06 textured checkpoint (stage1_july) in its own regime (see the builder.py note).
    ap.add_argument("--use-texture", action="store_true",
                    help="make the appearance ref TEXTURED (only for the July checkpoint)")
    ap.add_argument("--tag-suffix", default="",
                    help="suffix for the output folder/report name (e.g. _hp-012) — so runs do not overwrite each other")
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]

    # Build the conditionings ONCE (both variants use the same bundles — a fair ablation)
    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    combos = []
    for combo in manifest.combos[: args.limit]:
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        try:
            pp = prep.process(manifest.root / person.image, size=size)
            # person_square_bbox is REQUIRED: the default full-frame bbox projects the body to
            # the image centre (the mesh shifts for an off-centre person) — this alignment fix
            # was not being used in eval (2026-08-09).
            params = hmr2(pp.image, bbox=person_square_bbox(pp))
            asset = load_garment_asset(
                garments_root / garment.mesh,
                texture_path=garments_root / garment.texture if garment.texture else None,
                garment_id=garment.id,
                allow_untextured=True,
            )
            hp = {"hang_pad": args.hang_pad, "garment_scale": args.garment_scale,
                  "use_texture": args.use_texture}
            bundles = {
                a: build_conditioning(
                    pp.image, params, asset,
                    PhotoView() if a == 0 else OrbitView(a),
                    size=size, person_prep=pp, **hp,
                )
                for a in args.angles
            }
            combos.append((person, garment, bundles))
        except Exception as e:
            print(f"SKIP {person.id}×{garment.id}: {e}", file=sys.stderr)
    if not combos:
        raise SystemExit("ERROR: no combo could be built")

    sampler = FluxTryOnSampler(base["model"]["flux_fill_repo"], checkpoint=args.checkpoint,
                               prompt=base["model"]["prompt"])
    ctx = {"cfg": cfg, "out_root": out_root, "manifest": manifest, "angles": args.angles,
           "prep": prep}
    # tag-suffix: keep runs with different hang_pad from overwriting each other's PNGs (run_variant
    # skips an existing file — written to the same folder, the old alignment would silently survive)
    s_on = run_variant(sampler, combos, ctx, 1.0, f"control_on{args.tag_suffix}")
    s_off = run_variant(sampler, combos, ctx, 0.0, f"control_off{args.tag_suffix}")

    # GATE: geo_iou = the garment region IN THE OUTPUT (parse) vs the target silhouette — ON >= OFF
    get = lambda s, k: (s["overall"].get(k) or {}).get("mean")
    geo_on, geo_off = get(s_on, "geo_iou"), get(s_off, "geo_iou")
    de_on, de_off = get(s_on, "garment_delta_e"), get(s_off, "garment_delta_e")
    print(f"\n=== ABLATION GATE ===")
    print(f"geo_iou (output silhouette vs target): ON={geo_on} OFF={geo_off} (higher is better)")
    print(f"garment_delta_e                      : ON={de_on} OFF={de_off} (lower is better)")
    if geo_on is not None and geo_off is not None:
        if geo_on < geo_off - 0.01:
            print("GATE FAILED: control ON breaks the geometry — the v1 PHASE C scenario. Do NOT move"
                  " to Stage 2; remedies in order: raise ref_dropout (0.1→0.2) → continue training → FluxControlNet.")
            return 1
        print(f"GATE PASSED (geo_iou difference: {geo_on - geo_off:+.4f}).")
        return 0
    print("WARNING: geo_iou could not be computed (predsil missing) — the gate was judged on ΔE.")
    if de_on is not None and de_off is not None and de_on > de_off + 1.0:
        print("GATE FAILED (ΔE): control ON breaks color fidelity.")
        return 1
    print("GATE PASSED (ΔE based, weak signal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
