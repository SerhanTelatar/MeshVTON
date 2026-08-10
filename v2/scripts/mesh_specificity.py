#!/usr/bin/env python3
"""Mesh AYIRT EDİCİLİĞİ testi — çıktı gerçekten VERİLEN mesh'e mi özel?

geo_iou "çıktı hedef silüete oturdu mu" der ama tek başına yanıltıcıdır: model
her giysi için AYNI makul gömleği üretse de ortalama IoU yüksek çıkabilir.
control_on/control_off farkı da koşullamanın GÜCÜNÜ ölçer, AYIRT EDİCİLİĞİNİ değil.

Bu test her kişi için N×N matris kurar: i. giysinin çıktı-silüeti (predsil) vs
j. giysinin hedef silüeti (sil).
  köşegen  = doğru mesh eşleşmesi
  köşegen dışı = yanlış mesh
Köşegen belirgin yüksekse çıktı mesh'e özeldir (tez tutuyor). İkisi yakınsa model
hangi mesh verilirse verilsin aynı şeyi üretiyordur.

GPU/difüzyon YOK — yalnız mevcut .predsil.png / .sil.png dosyalarını yeniden puanlar.

Kullanım:
  python v2/scripts/mesh_specificity.py            # bulunan tüm pred setleri
  python v2/scripts/mesh_specificity.py --sets ckpt_control_on --angles 0
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from meshvton2.eval import metrics as M  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.eval.harness import _load_mask  # noqa: E402

# Pred setlerinin kök klasörleri: eval_results/<ad>/preds
DEFAULT_SETS = ("phase1_fill_spatial", "ckpt_control_on", "ckpt_control_off")


def analyse(pred_dir: Path, manifest, pattern: str, angles) -> dict | None:
    """Kişi başına giysi×giysi IoU matrisi kurup köşegen/köşegen-dışı ortalamalarını döndürür."""
    by_person: dict[str, list] = {}
    for combo in manifest.combos:
        by_person.setdefault(combo.person_id, []).append(combo.garment_id)

    diag_all: list[float] = []
    off_all: list[float] = []
    per_person: list[tuple[str, float, float, int]] = []

    for person_id, garment_ids in by_person.items():
        # Aynı giysi iki kez sayılmasın (manifest tekrar içerebilir)
        garments = list(dict.fromkeys(garment_ids))
        for angle in angles:
            stem = lambda g: (pred_dir / pattern.format(
                person_id=person_id, garment_id=g, angle=angle)).with_suffix("")
            predsil = {g: Path(f"{stem(g)}.predsil.png") for g in garments}
            sil = {g: Path(f"{stem(g)}.sil.png") for g in garments}
            usable = [g for g in garments if predsil[g].exists() and sil[g].exists()]
            if len(usable) < 2:  # matris için en az 2 giysi lazım
                continue

            d, o = [], []
            for gi, gj in product(usable, usable):
                # predsil ve sil aynı üretimden gelir → boyut eşit; yine de sil'i predsil'e uydur
                a = _load_mask(predsil[gi])
                b = _load_mask(sil[gj], size=a.shape)
                iou = M.silhouette_iou(a, b)
                if iou is None:
                    continue
                (d if gi == gj else o).append(iou)
            if d and o:
                diag_all += d
                off_all += o
                per_person.append((person_id, float(np.mean(d)), float(np.mean(o)), len(usable)))

    if not diag_all or not off_all:
        return None
    return {
        "diag": float(np.mean(diag_all)),
        "off": float(np.mean(off_all)),
        "n_diag": len(diag_all),
        "n_off": len(off_all),
        "per_person": per_person,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    ap.add_argument("--per-person", action="store_true", help="kişi kırılımını da bas")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    out_root = REPO / base["paths"]["eval_results"]

    print("=== MESH AYIRT EDİCİLİĞİ (köşegen = doğru mesh, yüksek iyi) ===\n")
    any_done = False
    for name in args.sets:
        pred_dir = out_root / name / "preds"
        if not pred_dir.exists():
            print(f"{name:22s} pred klasörü yok — atlandı")
            continue
        res = analyse(pred_dir, manifest, cfg["pred_pattern"], args.angles)
        if res is None:
            print(f"{name:22s} yeterli .predsil/.sil yok — atlandı "
                  f"(önce backfill_baseline_geo.py koşun)")
            continue
        any_done = True
        gap = res["diag"] - res["off"]
        rel = gap / res["off"] * 100 if res["off"] else float("nan")
        print(f"{name}")
        print(f"  doğru mesh  (köşegen)      = {res['diag']:.4f}  (n={res['n_diag']})")
        print(f"  yanlış mesh (köşegen dışı) = {res['off']:.4f}  (n={res['n_off']})")
        print(f"  AYIRT EDİCİLİK             = {gap:+.4f}  ({rel:+.1f}%)")
        if args.per_person:
            for pid, d, o, k in sorted(res["per_person"]):
                print(f"    {pid:12s} {k} giysi: doğru={d:.4f} yanlış={o:.4f} fark={d - o:+.4f}")
        print()

    if any_done:
        print("OKUMA: fark ~0 ise model hangi mesh verilirse verilsin aynı şeyi üretiyor —")
        print("       geo_iou'daki iyileşme mesh'e ÖZGÜ değil, dağılıma uyum demektir.")
        print("       Baseline setindeki farkı çıpa olarak kullanın (eğitimsiz model).")
    return 0 if any_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
