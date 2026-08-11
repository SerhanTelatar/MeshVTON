#!/usr/bin/env python3
"""GEOMETRİK ÜST SINIR figürü — 3B render'ın doğrudan kompoziti (difüzyon YOK).

Amaç: "koşullamanın taşıdığı geometrik bilgi ne kadar?" sorusunu görselleştirmek.
Giysinin drape edilmiş NORMAL haritası (control_normal) basit bir Lambert ışıkla
gölgelenip derinlik-testli silüet (control_depth_sil[2]) içine yapıştırılır. Sonuç:
fotoğrafın üstünde keskin, kıvrımlı, gölgeli gri giysi — modelin ulaşabileceği
geometrik tavan.

ÖNEMLİ, tezde böyle sunulmalı: bu bir ÜRETİM DEĞİL, 3B kompozit. Model çıktısıyla
yan yana konur ve şöyle okunur:
  - kompozit: geometri kusursuz, ama ESKİ GİYSİ kenarlardan görünmeye devam eder
    (kompozit onu kaldıramaz) ve ışık fotoğrafla uyumsuzdur;
  - model çıktısı: eski giysiyi KALDIRIR ve ışığa oturur, ama kıvrım üretemez.
Bu karşıtlık, üretken adımın neden gerekli olduğunun doğrudan kanıtıdır.

Difüzyon koşmaz; model çıktısı varsa mevcut preds klasöründen OKUNUR.

Kullanım:
  python v2/scripts/render_composite.py --idm-repo /content/IDM-VTON
  [--limit 5] [--pred-set ckpt_control_on_hp-012_sc125]
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

# Kamera-uzayı ışık yönü (+z kameraya doğru), sol-üstten. ÖLÇEREK seçildi: giysi
# yüzeyi büyük ölçüde kameraya bakar, kıvrımlar (0,0,1) etrafında ~25° sapmadır;
# okunabilirlik ışığın TEĞET bileşeniyle artar. 25°'lik rejimde ölçülen yoğunluk std:
#   (-0.35,0.45,0.82) teğet 0.57 → std 0.072, gri 190-248 (neredeyse düz beyaz)
#   (-0.60,0.60,0.53) teğet 0.85 → std 0.105, gri 136-222  ← seçilen
#   (-0.65,0.70,0.30) teğet 0.95 → std 0.112 ama giysi belirgin koyulaşıyor
LIGHT = np.array([-0.60, 0.60, 0.53])
LIGHT /= np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE = 0.42, 0.58  # ambient tabanı olmadan gölgeler tıkanıp siyaha düşüyor


def shade(normal_chw: np.ndarray, sil: np.ndarray) -> tuple[np.ndarray, dict]:
    """(3,H,W) [-1,1] normaller → (H,W) [0,1] Lambert yoğunluğu + teşhis.

    z-İŞARETİ OTOMATİK: kamera sözleşmesi (+y aşağı) yüzünden kameraya bakan
    yüzeylerin n_z işareti sabit değil. Yanlış işarette n·L her yerde negatife
    düşer, clip(0) ile her piksel ambient'e oturur ve kompozit DÜZ KOYU çıkar
    (ilk koşuda tam olarak bu oldu). Silüet içindeki ortalama n_z'ye bakıp
    ışığın z bileşenini ona göre çeviriyoruz.
    """
    n = normal_chw.transpose(1, 2, 0).astype(np.float64)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
    nz = float(n[sil][:, 2].mean()) if sil.any() else 0.0
    L = LIGHT.copy()
    if nz < 0:
        L[2] = -L[2]
    inten = AMBIENT + DIFFUSE * np.clip(n @ L, 0.0, 1.0)
    vals = inten[sil] if sil.any() else np.array([0.0])
    diag = {"nz": nz, "z_cevrildi": nz < 0,
            "gri_p5": int(np.percentile(vals, 5) * 255),
            "gri_p95": int(np.percentile(vals, 95) * 255),
            "std": float(vals.std())}
    return inten, diag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5, help="ilk N kombo")
    ap.add_argument("--pred-set", default="ckpt_control_on_hp-012_sc125",
                    help="model çıktılarının okunacağı eval_results alt klasörü")
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]
    pred_dir = out_root / args.pred_set / "preds"
    out_dir = out_root / "render_composite"
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}

    person_cache: dict[str, tuple] = {}
    panels_all = []
    for combo in manifest.combos[: args.limit]:
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        if person.id not in person_cache:
            pp = prep.process(manifest.root / person.image, size=size)
            person_cache[person.id] = (pp, hmr2(pp.image, bbox=person_square_bbox(pp)))
        pp, params = person_cache[person.id]
        asset = load_garment_asset(
            garments_root / garment.mesh,
            texture_path=garments_root / garment.texture if garment.texture else None,
            garment_id=garment.id, allow_untextured=True)
        b = build_conditioning(pp.image, params, asset, PhotoView(), size=size,
                               person_prep=pp, hang_pad=PHOTO_HANG_PAD,
                               garment_scale=PHOTO_GARMENT_SCALE)

        sil = b.control_depth_sil[2].numpy() > 0
        inten, d = shade(b.control_normal.numpy(), sil)
        comp = pp.image.copy()
        comp[sil] = np.clip(inten[sil, None] * 255.0, 0, 255).astype(np.uint8)
        # Silüet DELİK oranı: drape edilmiş giysi gövdeye giriyorsa derinlik testi
        # parçalar kesiyor → kompozitte eski giysi ortadan sızar (gerçek bir bulgu,
        # artefakt değil). Dolu gövde = morfolojik kapama sonrası alan.
        import cv2

        k = np.ones((max(size[1] // 40, 3),) * 2, np.uint8)
        closed = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        delik = 1.0 - sil.sum() / max(closed.sum(), 1)
        Image.fromarray(comp).save(out_dir / f"{person.id}__{garment.id}.composite.png")

        row = [("kişi", pp.image), ("3B RENDER KOMPOZİTİ (üst sınır)", comp)]
        p = pred_dir / cfg["pred_pattern"].format(
            person_id=person.id, garment_id=garment.id, angle=0)
        if p.exists():
            row.append(("model çıktısı", np.asarray(Image.open(p).convert("RGB"))))
        else:
            print(f"  NOT: model çıktısı yok ({p.name}) — o panel atlandı")
        panels_all.append((f"{person.id} × {garment.id}", row))
        print(f"OK {person.id} × {garment.id}  |  n_z={d['nz']:+.2f}"
              f"{' (ışık z ÇEVRİLDİ)' if d['z_cevrildi'] else ''}"
              f"  gri {d['gri_p5']}-{d['gri_p95']} std={d['std']:.3f}"
              f"  silüet delik oranı=%{delik*100:.1f}")

    TH = 230
    hh = round(TH * size[0] / size[1])
    ncol = max(len(r) for _, r in panels_all)
    grid = Image.new("RGB", (TH * ncol, hh * len(panels_all)), "white")
    for r, (_, row) in enumerate(panels_all):
        for c, (_, a) in enumerate(row):
            grid.paste(Image.fromarray(a).convert("RGB").resize((TH, hh), Image.LANCZOS),
                       (c * TH, r * hh))
    sp = out_dir / "upper_bound_grid.png"
    grid.save(sp)
    print(f"\nIzgara: {sp}")
    print("sütunlar: " + " | ".join(l for l, _ in panels_all[0][1]))
    print("\nOKUMA: kompozitte giysi keskin ve kıvrımlı ama ESKİ GİYSİ kenarlardan sızar")
    print("       ve ışık fotoğrafla uyumsuzdur; model çıktısı eskiyi kaldırır ve ışığa")
    print("       oturur ama kıvrım üretemez. Üretken adımın gerekçesi bu karşıtlıktır.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())