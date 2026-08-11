#!/usr/bin/env python3
"""Phase 3 — Synthetic multi-view data generation (Colab, GPU/EGL).

Usage:
  python v2/scripts/generate_synthetic.py --garments data/garments_3d --num 100 --seed 0
  # full generation: several runs can go in parallel/spread out with different --seed
  # (pairs.csv appends; sample_id contains the seed, so there are no collisions)

Requirements: pyrender + smplx + SMPLX_NEUTRAL.npz (the SMPLX_MODEL_DIR env var or
checkpoints/pretrained/smplx). --poses (an (N,63) .npy) is recommended for the pose bank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import yaml  # noqa: E402

from meshvton2.conditioning.builder import assert_real_impl  # noqa: E402
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.synth.generate import generate  # noqa: E402


def discover_assets(garments_root: Path, cache_dir: Path, limit: int | None):
    """Finds the garment folders recursively; ONLY those under the upper_body folder.

    PERMANENT RULE: the appearance ref is ALWAYS converted to flat grey in the builder
    (with or without a texture) — so no textured/untextured distinction is made here;
    any valid mesh can be used. But the garment TYPE matters: the drape/hang logic is
    tuned for the upper body (the v1 t-shirt/top assumption) — folders outside
    upper_body (e.g. the raw CLOTH3D val_t1 split, which may contain Trousers/Skirt/
    Dress/Jumpsuit with no type check) are excluded entirely.
    """
    from dataclasses import replace

    dirs = sorted(
        d for d in {p.parent for p in garments_root.rglob("*.obj")}
        if "upper_body" in d.relative_to(garments_root).parts
    )
    assets, skipped = [], []
    for d in dirs:
        try:
            gid = str(d.relative_to(garments_root)).replace("/", "__")
            a = load_garment_asset(sorted(d.glob("*.obj"))[0], garment_id=gid, allow_untextured=True)
            assets.append(replace(a, lbs_cache=str(cache_dir / f"{gid}.lbs.npz")))
        except (ValueError, IndexError):  # broken mesh: count it, do not spam
            skipped.append(d.name)
        if limit and len(assets) >= limit:
            break
    if skipped:
        print(f"SKIPPED: {len(skipped)} broken garments (first 3: {skipped[:3]})", file=sys.stderr)
    if not assets:
        raise SystemExit(f"ERROR: no usable garment under {garments_root}")
    print(f"USABLE GARMENTS: {len(assets)}")
    return assets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--garments", type=Path, required=True)
    ap.add_argument("--num", type=int, required=True, help="Number of samples (identity×garment) to generate")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poses", type=Path, default=None, help="(N,63) .npy pose bank (HMR2@VITON-HD)")
    ap.add_argument("--limit-garments", type=int, default=None)
    args = ap.parse_args()

    assert_real_impl()  # generating synthetic data with the stub is FORBIDDEN (the v1 placeholder lesson)

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    out = args.out or REPO / base["paths"]["synth_root"]
    assets = discover_assets(args.garments, Path(out) / "_lbs_cache", args.limit_garments)
    print(f"{len(assets)} garments, {args.num} samples × 4 views → {out}")

    stats = generate(
        assets, out, num_samples=args.num, size=size, seed=args.seed,
        poses_file=str(args.poses) if args.poses else None,
    )
    print(f"\nwritten={stats['written']} rejected={stats['rejected']} errors={len(stats['failed'])}")
    for f in stats["failed"][:5]:
        print(f"  {f}", file=sys.stderr)
    # A REJECT (quality rejection) is not an ERROR — the exit code is 1 only on real errors
    # (in small batches it is normal tail behaviour for every attempt to hit a rejection)
    return 1 if (stats["failed"] and not stats["written"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
