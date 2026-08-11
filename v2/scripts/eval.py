#!/usr/bin/env python3
"""MeshVTON v2 eval CLI.

Usage:
  python v2/scripts/eval.py --pred-dir <dir>                 # against the golden manifest
  python v2/scripts/eval.py --pred-dir <dir> --name phase1_fill
  python v2/scripts/eval.py --self-check <image dir>         # harness sanity: pred=gt → SSIM≈1

Phase 4+: the --checkpoint / --disable-control flags will arrive with the model integration.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import yaml  # noqa: E402

from meshvton2.eval import harness  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402


def self_check(image_dir: Path, out_dir: Path) -> int:
    """Pairs every image with itself as both prediction and GT; SSIM≈1/PSNR=inf is expected.
    Proves the harness works end to end (the Phase 0 exit criterion)."""
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"ERROR: no images inside {image_dir}", file=sys.stderr)
        return 1
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for img in images:
            stem = tmp / img.stem
            pred = stem.with_suffix(".png")
            shutil.copy(img, pred) if img.suffix.lower() == ".png" else _to_png(img, pred)
            shutil.copy(pred, Path(f"{stem}.gt.png"))
            rows.append(harness.EvalRow(img.stem, "self", 0, found=True, values=harness.evaluate_item(pred)))
    summary = harness.summarize(rows)
    harness.write_report(summary, out_dir / "self_check.json")
    ssim = summary["overall"].get("gt_ssim", {}).get("mean")
    print(f"self-check: {len(images)} images, mean gt_ssim = {ssim}")
    if ssim is None or ssim < 0.999:
        print("ERROR: identity SSIM ≈ 1 was expected — the harness is broken", file=sys.stderr)
        return 1
    print(f"OK — report: {out_dir / 'self_check.json'}")
    return 0


def _to_png(src: Path, dst: Path) -> None:
    from PIL import Image

    Image.open(src).convert("RGB").save(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, help="Directory of prediction images")
    ap.add_argument("--self-check", type=Path, help="Image directory for the harness sanity test")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--name", default="eval", help="Report file name stem")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    out_dir = args.out_dir or REPO / base["paths"]["eval_results"]

    if args.self_check:
        return self_check(args.self_check, out_dir)

    if not args.pred_dir:
        ap.error("--pred-dir or --self-check is required")
    manifest_path = args.manifest or REPO / cfg["golden_manifest"]
    if not manifest_path.exists():
        print(f"ERROR: no manifest: {manifest_path}\nRun first: python v2/scripts/build_golden_set.py", file=sys.stderr)
        return 1
    manifest = load_manifest(manifest_path)
    summary = harness.evaluate(manifest, args.pred_dir, cfg["pred_pattern"])
    harness.write_report(summary, out_dir / f"{args.name}.json")
    print(f"{summary['found']}/{summary['total']} predictions evaluated — report: {out_dir / f'{args.name}.json'} (+.md)")
    if summary["missing"]:
        print(f"Missing {len(summary['missing'])} predictions (first 5): {summary['missing'][:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
