#!/usr/bin/env python3
"""Giysi ÖLÇEĞİNİ fotoğraf yolunda ampirik kalibre et (askı kalibrasyonunun ikizi).

Bulgu (2026-08-10): askı düzeltildikten (hang_pad=-0.12) sonra bile giysi gövdeye
göre FAZLA GENİŞ görünüyor — kollardan taşan panço benzeri siluet (ör. 00737_00).
`_prealign_garment` bilinçli olarak hiç ölçekleme yapmıyor (CLOTH3D giysileri
metre ölçeğinde ve SMPL bedenine göre modelli). Ama fotoğraf yolunda gövde
HMR2'den geliyor; o beden CLOTH3D'nin varsaydığı bedenle aynı olmak zorunda değil.

Yöntem: garment_scale'i tarayıp her değerde mesh giysi silüeti ile parser'ın
bulduğu GERÇEK giysi maskesi arasındaki IoU'yu ölçer. Difüzyon YOK.

Okuma:
  - Tepe 1.0'ın belirgin ALTINDA ise giysi gerçekten fazla büyük; o değeri kullan.
  - Tepe 1.0 civarındaysa ölçek suçlu değil, genişlik farkı kesim/drape'ten geliyor.

Kullanım:
  python v2/scripts/calibrate_scale.py --idm-repo /content/IDM-VTON
  [--hang-pad -0.12] [--range 0.70 1.15 0.05] [--garment upper_body__00047_Top]
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

from meshvton2.conditioning.body import build_hmr2_backend  # noqa: E402
from meshvton2.conditioning.builder import (  # noqa: E402
    PHOTO_HANG_PAD, PhotoView, assert_real_impl, build_conditioning)
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402

ATR_GARMENT = (4, 7)
# validate_camera.py'nin doğruladığı kişiler (hizalaması kanıtlı olanlarla kalibre et)
DEFAULT_PERSONS = ("00000_00", "02935_00", "01455_00", "00737_00", "02199_00")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--persons", nargs="+", default=list(DEFAULT_PERSONS))
    ap.add_argument("--garment", default=None, help="varsayılan: manifest'teki ilk giysi")
    ap.add_argument("--hang-pad", type=float, default=PHOTO_HANG_PAD)
    ap.add_argument("--range", nargs=3, type=float, default=[0.70, 1.15, 0.05],
                    metavar=("BAS", "BIT", "ADIM"))
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]

    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    gid = args.garment or manifest.combos[0].garment_id
    garment = by_gid[gid]
    asset = load_garment_asset(
        garments_root / garment.mesh,
        texture_path=garments_root / garment.texture if garment.texture else None,
        garment_id=garment.id, allow_untextured=True,
    )

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()

    people = []
    for pid in args.persons:
        if pid not in by_pid:
            print(f"ATLA {pid}: manifest'te yok", file=sys.stderr)
            continue
        pp = prep.process(manifest.root / by_pid[pid].image, size=size)
        params = hmr2(pp.image, bbox=person_square_bbox(pp))
        worn = np.isin(cv2.resize(np.asarray(pp.parse), (size[1], size[0]),
                                  interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
        people.append((pid, pp, params, worn))
    if not people:
        raise SystemExit("HATA: hiç kişi hazırlanamadı")

    lo, hi, step = args.range
    scales = np.arange(lo, hi + 1e-9, step)
    # HEDEF alan: parser'ın bulduğu gerçek giysi. Silüet% bunun etrafında tepe yapmalı —
    # çok küçükse örtüşme eksik, çok büyükse birleşim şişer; IoU her iki uçta düşer.
    worn_pct = float(np.mean([w.mean() for *_, w in people])) * 100
    print(f"giysi: {gid} | kişi: {len(people)} | hang_pad={args.hang_pad:+.2f} | "
          f"ölçek taraması: {lo:.2f} → {hi:.2f} adım {step:.2f}")
    print(f"HEDEF: gerçek giysi alanı (parser) = %{worn_pct:.1f} — silüet% buna yaklaşmalı\n")
    print(f"{'ölçek':>7} {'ort.IoU':>8} {'silüet%':>8}   " + "  ".join(f"{p:>9}" for p, *_ in people))

    rows = []
    for sc in scales:
        ious, areas = [], []
        for pid, pp, params, worn in people:
            b = build_conditioning(pp.image, params, asset, PhotoView(),
                                   size=size, person_prep=pp,
                                   hang_pad=args.hang_pad, garment_scale=float(sc))
            sil = b.control_depth_sil[2].numpy() > 0
            union = (sil | worn).sum()
            ious.append(float((sil & worn).sum() / union) if union else 0.0)
            areas.append(float(sil.mean()))
        mean = float(np.mean(ious))
        rows.append((float(sc), mean, ious))
        print(f"{sc:>7.2f} {mean:>8.3f} {np.mean(areas)*100:>7.1f}%   "
              + "  ".join(f"{v:>9.3f}" for v in ious))

    best_sc, best_iou, _ = max(rows, key=lambda r: r[1])
    cur = next((r for r in rows if abs(r[0] - 1.0) < 1e-6), None)
    print(f"\nEN İYİ: ölçek={best_sc:.2f} → ort.IoU={best_iou:.3f}")
    # Tepe aralığın UCUNDAysa optimum dışarıda kalmıştır — sayıyı kullanma, aralığı genişlet
    if abs(best_sc - rows[-1][0]) < 1e-6 or abs(best_sc - rows[0][0]) < 1e-6:
        print(f"UYARI: tepe tarama SINIRINDA ({best_sc:.2f}) — gerçek optimum aralığın DIŞINDA.")
        print(f"  Aralığı genişletip tekrar koşun, ör: --range {best_sc:.2f} {best_sc + 0.5:.2f} 0.05")
    if cur:
        print(f"MEVCUT: ölçek=1.00 → ort.IoU={cur[1]:.3f}  (fark {best_iou - cur[1]:+.3f})")
    if cur and best_iou - cur[1] < 0.02:
        print("\nSONUÇ: ölçek suçlu DEĞİL — 1.00 zaten tepeye yakın. Genişlik farkı")
        print("  kesim/drape'ten geliyor; garment_scale'i değiştirmeyin.")
    else:
        print(f"\nSONUÇ: ölçek anlamlı — foto yolunda garment_scale={best_sc:.2f} kullanın.")
        print("  DİKKAT: SENTETİK yolu ELLEMEYİN (orada gövde de giysi de aynı SMPL-X'ten;")
        print("  yeniden ölçekleme geçmişte hep hataydı — bkz. _prealign_garment).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())