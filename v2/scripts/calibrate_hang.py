#!/usr/bin/env python3
"""Giysi askı yüksekliğini FOTOĞRAF yolunda ampirik kalibre et.

Bulgu (2026-08-10): kamera kapısından geçmiş kişilerde (IoU 0.82) gövde doğru
projekte oluyor ama giysi mesh'i gerçek giysiye göre ~21 puan YUKARIDA duruyor
(yüzün üstüne biniyor). Sentetik yolda aynı sorun yok — iki yol çelişiyor, o
yüzden hang_pad'i tartışmak yerine ÖLÇÜYORUZ.

Yöntem: hang_pad'i tarayıp her değerde mesh giysi silüeti ile parser'ın bulduğu
GERÇEK giysi maskesi arasındaki IoU'yu hesaplar. Difüzyon YOK — yalnız HMR2 +
parser + render.

Okuma:
  - En iyi IoU YÜKSEK (>0.5) ve tepe belirgin ise: sorun basit bir dikey ofset,
    o hang_pad'i kullan.
  - En iyi IoU her değerde DÜŞÜK ise: sorun ofset değil (ölçek/poz/binding) —
    hang_pad'i değiştirmek çözmez, daha derine bakmak gerekir.

Kullanım:
  python v2/scripts/calibrate_hang.py --idm-repo /content/IDM-VTON
  [--persons 00000_00 02935_00] [--garment upper_body__00047_Top]
  [--range -0.30 0.12 0.03]
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
from meshvton2.conditioning.builder import PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402

ATR_GARMENT = (4, 7)  # parser'da üst giyim etiketleri
# validate_camera.py'nin doğruladığı kişiler (hizalaması kanıtlı olanlarla kalibre et)
DEFAULT_PERSONS = ("00000_00", "02935_00", "01455_00", "00737_00", "02199_00")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--persons", nargs="+", default=list(DEFAULT_PERSONS))
    ap.add_argument("--garment", default=None, help="varsayılan: manifest'teki ilk giysi")
    ap.add_argument("--range", nargs=3, type=float, default=[-0.30, 0.12, 0.03],
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

    # Kişi ön-işlemesi + HMR2 BİR kez (tarama yalnız hang_pad'i değiştirir)
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
    pads = np.arange(lo, hi + 1e-9, step)
    print(f"giysi: {gid} | kişi: {len(people)} | hang_pad taraması: "
          f"{lo:+.2f} → {hi:+.2f} adım {step:.2f}\n")
    print(f"{'hang_pad':>9} {'ort.IoU':>8}   " + "  ".join(f"{p:>9}" for p, *_ in people))

    rows = []
    for pad in pads:
        ious = []
        for pid, pp, params, worn in people:
            b = build_conditioning(pp.image, params, asset, PhotoView(),
                                   size=size, person_prep=pp, hang_pad=float(pad))
            sil = b.control_depth_sil[2].numpy() > 0
            union = (sil | worn).sum()
            ious.append(float((sil & worn).sum() / union) if union else 0.0)
        mean = float(np.mean(ious))
        rows.append((float(pad), mean, ious))
        print(f"{pad:>+9.2f} {mean:>8.3f}   " + "  ".join(f"{v:>9.3f}" for v in ious))

    best_pad, best_iou, _ = max(rows, key=lambda r: r[1])
    cur = next((r for r in rows if abs(r[0] - 0.06) < 1e-6), None)
    print(f"\nEN İYİ: hang_pad={best_pad:+.2f} → ort.IoU={best_iou:.3f}")
    if cur:
        print(f"MEVCUT: hang_pad=+0.06 → ort.IoU={cur[1]:.3f}  (fark {best_iou - cur[1]:+.3f})")
    if best_iou < 0.35:
        print("\nSONUÇ: hiçbir ofset yeterli hizalama vermiyor — sorun dikey kayma DEĞİL.")
        print("  Ölçek/poz/binding tarafına bakılmalı; hang_pad'i değiştirmek çözmez.")
    else:
        print("\nSONUÇ: tepe belirgin — sorun dikey ofset. Bu değeri fotoğraf yolunda kullanın.")
        print("  DİKKAT: sentetik yol +0.06 ile QA'dan geçmişti (builder.py:36-43).")
        print("  Değeri GLOBAL değiştirmeden önce sentetik QA'yı tekrar koşun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
