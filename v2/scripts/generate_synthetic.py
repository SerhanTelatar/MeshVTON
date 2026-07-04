#!/usr/bin/env python3
"""Faz 3 — Sentetik multi-view veri üretimi (Colab, GPU/EGL).

Kullanım:
  python v2/scripts/generate_synthetic.py --garments data/garments_3d --num 100 --seed 0
  # tam üretim: birden çok koşu farklı --seed ile paralel/aralıklı yapılabilir
  # (pairs.csv append eder; sample_id seed'i içerir, çakışma olmaz)

Gereksinimler: pyrender + smplx + SMPLX_NEUTRAL.npz (SMPLX_MODEL_DIR env veya
checkpoints/pretrained/smplx). Poz bankası için --poses (N,63 .npy) önerilir.
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
    """Texture'lı giysi klasörlerini özyinelemeli bulur; upper_body öncelikli."""
    from dataclasses import replace

    dirs = sorted({p.parent for p in garments_root.rglob("*.obj")})
    dirs = [d for d in dirs if "upper_body" in d.relative_to(garments_root).parts] + [
        d for d in dirs if "upper_body" not in d.relative_to(garments_root).parts
    ]
    assets, skipped = [], []
    for d in dirs:
        try:
            gid = str(d.relative_to(garments_root)).replace("/", "__")
            a = load_garment_asset(sorted(d.glob("*.obj"))[0], garment_id=gid)
            assets.append(replace(a, lbs_cache=str(cache_dir / f"{gid}.lbs.npz")))
        except (ValueError, IndexError):  # texturesiz/bozuk giysi: say, spam'leme
            skipped.append(d.name)
        if limit and len(assets) >= limit:
            break
    if skipped:
        print(f"ATLANDI: {len(skipped)} texturesiz/bozuk giysi (ilk 3: {skipped[:3]}) — "
              "v1 gri-render dersi gereği bilinçli", file=sys.stderr)
    if not assets:
        raise SystemExit(f"HATA: {garments_root} altında kullanılabilir (texture'lı) giysi yok")
    print(f"KULLANILABİLİR GİYSİ: {len(assets)}")
    return assets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--garments", type=Path, required=True)
    ap.add_argument("--num", type=int, required=True, help="Üretilecek örnek (kimlik×giysi) sayısı")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poses", type=Path, default=None, help="(N,63) .npy poz bankası (HMR2@VITON-HD)")
    ap.add_argument("--limit-garments", type=int, default=None)
    args = ap.parse_args()

    assert_real_impl()  # stub'la sentetik veri üretmek YASAK (v1 placeholder dersi)

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    out = args.out or REPO / base["paths"]["synth_root"]
    assets = discover_assets(args.garments, Path(out) / "_lbs_cache", args.limit_garments)
    print(f"{len(assets)} giysi, {args.num} örnek × 4 görüş → {out}")

    stats = generate(
        assets, out, num_samples=args.num, size=size, seed=args.seed,
        poses_file=str(args.poses) if args.poses else None,
    )
    print(f"\nyazıldı={stats['written']} red={stats['rejected']} hata={len(stats['failed'])}")
    for f in stats["failed"][:5]:
        print(f"  {f}", file=sys.stderr)
    return 0 if stats["written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
