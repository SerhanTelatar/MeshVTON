#!/usr/bin/env python3
"""Evaluate the checkpoint under the STANDARD VITON-HD paired protocol (SSIM / LPIPS / FID).

Why this script exists: the golden-set metrics (geo_iou, specificity, placement) are ours and
have no counterpart in the literature, so the thesis cannot be placed next to published numbers.
The paired VITON-HD protocol can be run, but only in ONE configuration:

    garment=None  ->  geometry is BODY-ONLY, the garment silhouette comes from the parse and the
                      appearance reference is the product photo.

That is exactly the real-data branch the model was trained on (preprocess_vitonhd.py), and it is
also exactly what the published protocol assumes: the target garment is the one the person is
already wearing, so the original photograph is the ground truth. The mesh-conditioned
configuration CANNOT be scored this way -- no photograph exists of a VITON-HD person wearing a
CLOTH3D mesh. Report the row accordingly: it measures the model WITHOUT its 3D contribution.

Metrics are computed at 512x384 by default so they line up with the comparison table, which is
reported at that resolution; generation always happens at the model's fixed 1024x768.

FID is NOT computed here. It is unstable below ~2000 samples, so it is only meaningful on the
full test split; this script writes the predictions and prints the clean-fid command to run.

Usage (Colab, A100):
  python v2/scripts/eval_vitonhd_protocol.py --checkpoint <Drive>/stage1/final.pt \\
      --idm-repo /content/IDM-VTON --limit 500
  [--metric-size 512 384] [--steps 28] [--tag vitonhd_paired]
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
from meshvton2.conditioning.builder import PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval import metrics as M  # noqa: E402
from meshvton2.model.flux_tryon import FluxTryOnSampler  # noqa: E402


def resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Both prediction and ground truth are brought to the metric resolution the table uses."""
    return np.asarray(Image.fromarray(img).resize((hw[1], hw[0]), Image.LANCZOS))


def ci95(v: list[float]) -> tuple[float, float]:
    a = np.asarray(v, np.float64)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, default=None,
                    help="VITON-HD test dir holding image/ and cloth/; auto-detected if omitted")
    ap.add_argument("--limit", type=int, default=500, help="0 = the whole test split (2032 pairs)")
    ap.add_argument("--metric-size", type=int, nargs=2, default=[512, 384], metavar=("H", "W"))
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="vitonhd_paired")
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    msize = tuple(args.metric_size)

    # The download unpacks under different names depending on the source (the archive is called
    # zalando-hd-resized; base.yaml declares v2/data/vitonhd). Try the known layouts in order.
    candidates = [args.test_root] if args.test_root else [
        REPO / "data/zalando-hd-resized/test",       # build_golden_set.py uses this one
        REPO / base["paths"]["vitonhd_root"] / "test",
        REPO / "data/vitonhd/test",
        REPO / "data/raw/test",
    ]
    test_root = next((c for c in candidates
                      if c and (c / "image").is_dir() and (c / "cloth").is_dir()), None)
    if test_root is None:
        print("ERROR: no VITON-HD test split with both image/ and cloth/. Tried:", file=sys.stderr)
        for c in candidates:
            if not c:
                continue
            print(f"  {c}  image={'y' if (c/'image').is_dir() else 'n'} "
                  f"cloth={'y' if (c/'cloth').is_dir() else 'n'}", file=sys.stderr)
        print("\nLocate it and pass --test-root, e.g.:\n"
              "  find /content /content/drive -maxdepth 6 -type d -name cloth 2>/dev/null",
              file=sys.stderr)
        return 2
    img_dir, cloth_dir = test_root / "image", test_root / "cloth"
    print(f"VITON-HD test split: {test_root}")

    # PAIRED setting: person and product photo share the file name. This is the reconstruction
    # test the published SSIM/LPIPS numbers are computed on -- do NOT use test_pairs.txt here,
    # that file defines the UNPAIRED (FID-only) setting.
    names = sorted(p.name for p in img_dir.glob("*.jpg") if (cloth_dir / p.name).exists())
    if not names:
        raise SystemExit("ERROR: no person/cloth pair with a matching name")
    if args.limit:
        names = names[: args.limit]

    out_dir = REPO / base["paths"]["eval_results"] / args.tag
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    sampler = FluxTryOnSampler(base["model"]["flux_fill_repo"], checkpoint=args.checkpoint,
                               prompt=base["model"]["prompt"])

    print(f"VITON-HD paired protocol | {len(names)} pairs | generate {size} | "
          f"score {msize[0]}x{msize[1]} | {args.steps} steps\n")

    ssim_v: list[float] = []
    lpips_v: list[float] = []
    rows = []
    failed = 0

    for i, name in enumerate(names):
        dst = out_dir / "preds" / name.replace(".jpg", ".png")
        try:
            pp = prep.process(img_dir / name, size=size)
            params = hmr2(pp.image, bbox=person_square_bbox(pp))
            ref = np.asarray(Image.open(cloth_dir / name).convert("RGB"))

            if dst.exists():  # resume-safe: scoring is cheap, generation is not
                pred = np.asarray(Image.open(dst).convert("RGB"))
            else:
                bundle = build_conditioning(
                    pp.image, params, None, PhotoView(),  # garment=None -> body-only geometry
                    size=size, person_prep=pp, appearance_ref_image=ref,
                )
                pred = sampler.sample(bundle, steps=args.steps, seed=args.seed)
                Image.fromarray(pred).save(dst)

            p, g = resize(pred, msize), resize(pp.image, msize)
            s = M.compute_ssim(p, g)
            l = M.compute_lpips(p, g)
            ssim_v.append(s)
            if l is not None:
                lpips_v.append(l)
            rows.append({"name": name, "ssim": s, "lpips": l})
            if (i + 1) % 25 == 0 or i == 0:
                lp = f" LPIPS={np.mean(lpips_v):.4f}" if lpips_v else ""
                print(f"[{i+1}/{len(names)}] SSIM={np.mean(ssim_v):.4f}{lp}")
        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(names)}] ERROR {name}: {e}", file=sys.stderr)

    if not ssim_v:
        raise SystemExit("ERROR: not a single pair could be scored")

    s_m, s_ci = ci95(ssim_v)
    l_m, l_ci = ci95(lpips_v) if lpips_v else (float("nan"), 0.0)
    summary = {
        "protocol": "VITON-HD paired reconstruction",
        "configuration": "body-only geometry, product-photo reference (garment=None)",
        "n": len(ssim_v), "failed": failed,
        "generate_size": list(size), "metric_size": list(msize), "steps": args.steps,
        "ssim": {"mean": s_m, "ci95": s_ci},
        "lpips": {"mean": l_m, "ci95": l_ci} if lpips_v else None,
        "rows": rows,
    }
    report = REPO / base["paths"]["eval_results"] / f"{args.tag}.json"
    report.write_text(json.dumps(summary, indent=2))

    print(f"\n=== VITON-HD PAIRED PROTOCOL (n={len(ssim_v)}, failed={failed}) ===")
    print(f"SSIM  = {s_m:.4f} +/- {s_ci:.4f}")
    if lpips_v:
        print(f"LPIPS = {l_m:.4f} +/- {l_ci:.4f}")
    print(f"report: {report}")
    print("\nCONFIGURATION NOTE: this row is the model WITHOUT mesh conditioning "
          "(no photograph exists of these people wearing a CLOTH3D mesh). Label it as such.")
    if len(ssim_v) < 2000:
        print(f"\nFID: NOT computed -- {len(ssim_v)} samples is too few for a stable value.")
        print("     Run the full split (--limit 0), then:")
        print(f"       pip install clean-fid && python -c \"from cleanfid import fid; "
              f"print(fid.compute_fid('{out_dir / 'preds'}', '{img_dir}'))\"")
    else:
        print(f"\nFID: sample count is sufficient. Run:")
        print(f"  python -c \"from cleanfid import fid; "
              f"print(fid.compute_fid('{out_dir / 'preds'}', '{img_dir}'))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
