#!/usr/bin/env python3
"""Baseline tahminlerine geo_iou aux'larını GERİ-DOLDUR + eğitimli ckpt ile kıyasla.

Neden: zero_shot_baseline.py yalnız .person/.mask/.ref yazıyor; harness.evaluate'in
geo_iou'su ise .predsil + .sil ister (harness.py::evaluate_item). Bu yüzden
"eğitim baseline'ı yendi mi?" sorusu ölçülemiyordu — ablation kapısı yalnız
kontrol AÇIK/KAPALI'yı kıyaslıyor, eğitimli-vs-eğitimsizi değil.

Bu script MEVCUT baseline PNG'lerini yeniden üretmez (difüzyon koşmaz — GPU'da
yalnız HMR2 + parser çalışır). Sadece eksik aux'ları yazıp raporu tazeler.

Kullanım (Colab, A100):
  python v2/scripts/backfill_baseline_geo.py --idm-repo /content/IDM-VTON
  [--variant fill_spatial] [--limit N] [--angles 0]
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
from meshvton2.conditioning.builder import OrbitView, PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval import harness  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402

from eval_checkpoint import _save_predsil  # noqa: E402  (aynı parser yolu — tutarlılık şart)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--variant", default="fill_spatial")
    ap.add_argument("--limit", type=int, default=0, help="0 = tüm kombolar")
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]
    pred_dir = out_root / f"phase1_{args.variant}" / "preds"
    if not pred_dir.exists():
        raise SystemExit(f"HATA: baseline tahminleri yok: {pred_dir}")

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    combos = manifest.combos[: args.limit] if args.limit else manifest.combos

    # Kişi başı prep+HMR2, giysi başı asset — kombo döngüsünde tekrar yok
    person_cache: dict[str, tuple] = {}
    asset_cache: dict[str, object] = {}
    done = skipped = missing = 0

    for i, combo in enumerate(combos):
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        stems = {
            a: (pred_dir / cfg["pred_pattern"].format(
                person_id=person.id, garment_id=garment.id, angle=a)).with_suffix("")
            for a in args.angles
        }
        # Tahmin PNG'si yoksa aux üretmenin anlamı yok
        todo = {a: s for a, s in stems.items() if s.with_suffix(".png").exists()}
        missing += len(stems) - len(todo)
        if not todo:
            continue
        # Her iki aux da varsa bu komboyu atla (resume-safe)
        if all(Path(f"{s}.sil.png").exists() and Path(f"{s}.predsil.png").exists()
               for s in todo.values()):
            skipped += len(todo)
            print(f"[{i+1}/{len(combos)}] atlandı (aux tam) {person.id}×{garment.id}")
            continue

        try:
            if person.id not in person_cache:
                pp = prep.process(manifest.root / person.image, size=size)
                # person_square_bbox ŞART — tam-kare bbox off-center kişide mesh'i kaydırır
                person_cache[person.id] = (pp, hmr2(pp.image, bbox=person_square_bbox(pp)))
            pp, params = person_cache[person.id]
            if garment.id not in asset_cache:
                # HAM asset — builder zaten force_textureless uyguluyor (builder.py:316,341);
                # burada tekrar uygulamak eval_checkpoint yolundan sapma riski yaratır
                asset_cache[garment.id] = load_garment_asset(
                    garments_root / garment.mesh,
                    texture_path=garments_root / garment.texture if garment.texture else None,
                    garment_id=garment.id,
                    allow_untextured=True,
                )
            asset = asset_cache[garment.id]

            for angle, stem in todo.items():
                bundle = build_conditioning(
                    pp.image, params, asset,
                    PhotoView() if angle == 0 else OrbitView(angle),
                    size=size, person_prep=pp,
                )
                # HEDEF silüet — eval_checkpoint.run_variant ile bit-eş aynı kaynak
                sil = (bundle.control_depth_sil[2].numpy() > 0).astype("uint8") * 255
                Image.fromarray(sil).save(f"{stem}.sil.png")
                _save_predsil(prep, stem)  # ÜRETİLEN görüntünün giysi bölgesi (parser)
                done += 1
            print(f"[{i+1}/{len(combos)}] OK {person.id}×{garment.id}")
        except Exception as e:
            print(f"[{i+1}/{len(combos)}] HATA {person.id}×{garment.id}: {e}", file=sys.stderr)

    print(f"\naux yazıldı={done} atlandı={skipped} tahmin-yok={missing}")

    # Raporu tazele + eğitimli checkpoint'le kıyasla
    summary = harness.evaluate(manifest, pred_dir, cfg["pred_pattern"])
    report = out_root / f"phase1_{args.variant}.json"
    harness.write_report(summary, report)
    print(f"rapor güncellendi: {report}")

    get = lambda s, k: (s["overall"].get(k) or {}).get("mean")
    rows = {(r["person"], r["garment"], r["angle"]): r for r in summary["rows"]}
    print("\n=== EĞİTİM vs BASELINE (geo_iou, yüksek iyi) ===")
    base_all = get(summary, "geo_iou")
    print(f"baseline (tüm {summary['found']} kombo): {base_all}")
    for tag in ("control_on", "control_off"):
        ck = out_root / f"ckpt_{tag}.json"
        if not ck.exists():
            print(f"{tag}: rapor yok ({ck})")
            continue
        c = json.loads(ck.read_text())
        crows = {(r["person"], r["garment"], r["angle"]): r for r in c["rows"]}
        shared = sorted(set(rows) & set(crows))
        bv = [rows[k]["geo_iou"] for k in shared if rows[k].get("geo_iou") is not None]
        cv = [crows[k]["geo_iou"] for k in shared if crows[k].get("geo_iou") is not None]
        if len(bv) == len(cv) == len(shared) and shared:
            b_m, c_m = float(np.mean(bv)), float(np.mean(cv))
            print(f"{tag}: ortak {len(shared)} kombo — baseline={b_m:.4f} ckpt={c_m:.4f} "
                  f"Δ={c_m - b_m:+.4f}  {'EĞİTİM İYİ' if c_m > b_m else 'EĞİTİM KÖTÜ'}")
        else:
            print(f"{tag}: ortak kombo yok ya da geo_iou eksik (baseline {len(bv)}/{len(shared)}, "
                  f"ckpt {len(cv)}/{len(shared)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
