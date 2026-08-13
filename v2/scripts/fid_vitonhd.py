#!/usr/bin/env python3
"""FID against the VITON-HD test images, computed so it is comparable with published numbers.

Two details decide whether the number can sit next to a published FID:

1. IMPLEMENTATION. The comparison table follows Seitzer 2020, i.e. `pytorch-fid`. `clean-fid`
   deliberately changes the resizing path and returns values several points apart. We use
   pytorch-fid.
2. RESOLUTION. Published VITON-HD FID is computed at 512x384, while we generate at 1024x768.
   Inception resizes to 299 internally, but it does so from whatever it is handed, so the source
   resolution still moves the number. Both sets are therefore resized to the metric resolution
   into a temporary directory before scoring.

Sample count matters more than either: FID is biased upward on small sets, so a 500-sample value
cannot be compared with a 2032-sample one. The script refuses to print a comparable number below
--min-n unless --force is given.

Usage:
  python v2/scripts/fid_vitonhd.py \\
      --pred-dir v2/eval_results/vitonhd_paired/preds \\
      --real-dir data/zalando-hd-resized/test/image [--size 512 384]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from PIL import Image  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg"}


def resized_copy(src: Path, dst: Path, size: tuple[int, int], names: set[str] | None) -> int:
    """Writes every image of src into dst at `size`. Returns how many were written."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        if names is not None and p.stem not in names:
            continue
        Image.open(p).convert("RGB").resize((size[1], size[0]), Image.LANCZOS).save(
            dst / f"{p.stem}.png")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--real-dir", type=Path, required=True)
    ap.add_argument("--size", type=int, nargs=2, default=[512, 384], metavar=("H", "W"))
    ap.add_argument("--min-n", type=int, default=2000,
                    help="below this the value is not comparable with published FID")
    ap.add_argument("--all-real", action="store_true",
                    help="score against ALL real test images rather than only the matching ones")
    ap.add_argument("--force", action="store_true", help="compute even below --min-n")
    args = ap.parse_args()

    for d in (args.pred_dir, args.real_dir):
        if not d.is_dir():
            raise SystemExit(f"ERROR: no such directory: {d}")

    pred_names = {p.stem for p in args.pred_dir.iterdir() if p.suffix.lower() in EXTS}
    if not pred_names:
        raise SystemExit(f"ERROR: no images in {args.pred_dir}")
    if len(pred_names) < args.min_n and not args.force:
        print(f"ERROR: only {len(pred_names)} predictions. Published FID on VITON-HD uses all "
              f"2032 test pairs, and FID is biased upward on smaller sets, so this value would "
              f"not be comparable.\n"
              f"       Generate the rest with `eval_vitonhd_protocol.py --limit 0` (resume-safe),\n"
              f"       or pass --force to compute an internal-use-only value.", file=sys.stderr)
        return 2

    try:
        import pytorch_fid  # noqa: F401
    except ImportError:
        raise SystemExit("ERROR: pip install pytorch-fid")

    size = tuple(args.size)
    tmp = Path(tempfile.mkdtemp(prefix="fid_"))
    try:
        # The real side is matched to the predictions by file name, so the two sets describe the
        # same subjects; --all-real switches to the full split if that is what is wanted.
        np_ = resized_copy(args.pred_dir, tmp / "pred", size, None)
        nr = resized_copy(args.real_dir, tmp / "real", size, None if args.all_real else pred_names)
        print(f"pred={np_} real={nr} @ {size[0]}x{size[1]} (pytorch-fid)")
        if nr == 0:
            raise SystemExit("ERROR: no real image matched the prediction names")

        cmd = [sys.executable, "-m", "pytorch_fid", str(tmp / "pred"), str(tmp / "real")]
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if len(pred_names) < args.min_n:
        print(f"\nWARNING: computed on {len(pred_names)} samples. NOT comparable with published "
              f"FID; do not put this value in the comparison table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
