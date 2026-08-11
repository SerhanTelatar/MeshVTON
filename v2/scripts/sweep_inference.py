#!/usr/bin/env python3
"""Çıkarım kolları taraması — EĞİTİMSİZ kalite iyileştirmesi arayışı.

Bağlam (2026-08-11): çıktı doğru yerde/boyutta ama düz ve yarı saydam. Elenenler:
eğitim bütçesi (4000 adımlık koşu da solgundu, [[meshvton-v2-plan]]), guidance
(1.0-15.0 arası etkisiz), kompozit (ham çıktı da düz), kamera, hizalama.

Kalan iki HİÇ DENENMEMİŞ kol:

1. control_scale > 1.0 — şimdiye dek yalnız 0.0 (ablation) ve 1.0 (eğitimdeki)
   denendi. Kıvrım bilgisi zaten girdide var (control_normal drape edilmiş giysinin
   normal haritası); model kullanmıyor olabilir, sinyali yükseltmek işe yarayabilir.

2. geometry_mask=False — maske şu an parse ∪ dilate(silüet) ve %48'e çıkıyor; model
   kolları/omuzları/boynu YENİDEN ÜRETMEK zorunda kalıyor (ten sorunu buradan).
   False ile maske yalnız parser giysisi → ten ve kollar FOTOĞRAFTAN korunur.
   Bedeli: yeni giysi eskisinden genişse maske dışına taşamaz, kırpılır.

Difüzyon koşar ama tarama küçüktür (kol sayısı × varyant). Eğitim YOK.

Kullanım:
  python v2/scripts/sweep_inference.py --checkpoint <Drive>/stage1/final.pt \\
      --idm-repo /content/IDM-VTON [--control-scale 1.0 1.5 2.0 3.0]
      [--person 00000_00] [--garment upper_body__00047_Top] [--steps 28]
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
    PHOTO_GARMENT_SCALE, PHOTO_HANG_PAD, PhotoView, assert_real_impl, build_conditioning)
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.model.flux_tryon import FluxTryOnSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--person", default=None)
    ap.add_argument("--garment", default=None)
    ap.add_argument("--control-scale", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_dir = REPO / base["paths"]["eval_results"] / "inference_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    pid = args.person or manifest.combos[0].person_id
    gid = args.garment or manifest.combos[0].garment_id
    person, garment = by_pid[pid], by_gid[gid]

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    pp = prep.process(manifest.root / person.image, size=size)
    params = hmr2(pp.image, bbox=person_square_bbox(pp))
    asset = load_garment_asset(
        garments_root / garment.mesh,
        texture_path=garments_root / garment.texture if garment.texture else None,
        garment_id=garment.id, allow_untextured=True,
    )
    sampler = FluxTryOnSampler(base["model"]["flux_fill_repo"], checkpoint=args.checkpoint,
                               prompt=base["model"]["prompt"])

    # İki maske rejimi için koşullama AYRI kurulur (maske build_conditioning içinde belirlenir)
    bundles = {}
    for gm in (True, False):
        bundles[gm] = build_conditioning(
            pp.image, params, asset, PhotoView(), size=size, person_prep=pp,
            hang_pad=PHOTO_HANG_PAD, garment_scale=PHOTO_GARMENT_SCALE, geometry_mask=gm)

    print(f"kişi={pid} giysi={gid} steps={args.steps} seed={args.seed}")
    for gm, b in bundles.items():
        print(f"  geometry_mask={gm}: maske alanı = %{(b.inpaint_mask.numpy()[0] > 0.5).mean()*100:.1f}")
    print()

    H, W = size
    TH = 210
    hh = round(TH * H / W)
    cols = 1 + len(args.control_scale)          # girdi + her control_scale
    grid = Image.new("RGB", (TH * cols, hh * len(bundles)), "white")
    rows = []
    for r, (gm, b) in enumerate(bundles.items()):
        grid.paste(Image.fromarray(pp.image).resize((TH, hh), Image.LANCZOS), (0, r * hh))
        m = b.inpaint_mask.numpy()[0] > 0.5
        for c, cs in enumerate(args.control_scale):
            img = sampler.sample(b, steps=args.steps, seed=args.seed, control_scale=float(cs))
            Image.fromarray(img).save(out_dir / f"{pid}__{gid}__gm{int(gm)}_cs{cs:g}.png")
            # maske-içi std: doku/kontrast vekili (guidance taramasındakiyle aynı ölçüt)
            std = float(np.asarray(img, np.float32)[m].std())
            rows.append((gm, cs, std))
            grid.paste(Image.fromarray(img).resize((TH, hh), Image.LANCZOS), ((c + 1) * TH, r * hh))
            print(f"  geometry_mask={gm} control_scale={cs:>4.1f} → maske-içi std={std:6.2f}")

    sp = out_dir / f"{pid}__{gid}__sweep.png"
    grid.save(sp)
    print(f"\nIzgara: {sp}")
    print(f"  satırlar: geometry_mask=True (üst), False (alt)")
    print(f"  sütunlar: girdi, " + ", ".join(f"cs={c:g}" for c in args.control_scale))
    best = max(rows, key=lambda r: r[2])
    print(f"\nEN YÜKSEK DOKU: geometry_mask={best[0]} control_scale={best[1]:g} → {best[2]:.2f}")
    print("NOT: std VEKİL bir ölçüt (gürültü de yükseltir). Kararı IZGARAYA BAKARAK verin:")
    print("     kıvrım/kenar keskinleşiyor mu, yoksa artefakt mı?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())