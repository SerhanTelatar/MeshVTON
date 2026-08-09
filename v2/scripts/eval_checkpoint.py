#!/usr/bin/env python3
"""Faz 4 çıkış kapısı — eğitilmiş checkpoint'le golden set değerlendirmesi + ablation.

İki koşu yapar ve karşılaştırır:
  1. kontrol AÇIK  (control_scale=1.0)
  2. kontrol KAPALI (control_scale=0.0 — zero-init sayesinde bit-eş stok davranış)

KAPI (v1 FAZ C dersinin otomatik testi): AÇIK, KAPALI'dan kötüyse eğitim geometriyi
öğrenememiş demektir → Aşama 2'ye geçme, teşhis et (ilk çare: ref_dropout artır).

Kullanım (Colab, A100):
  python v2/scripts/eval_checkpoint.py --checkpoint <Drive>/stage1/latest-ckpt.pt \\
      --idm-repo /content/IDM-VTON [--limit 10] [--angles 0]
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
from meshvton2.conditioning.builder import OrbitView, PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval import harness  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.model.flux_tryon import FluxTryOnSampler  # noqa: E402
from meshvton2.utils.image_utils import tensor_to_pil  # noqa: E402


ATR_GARMENT = (4, 7)


def _save_predsil(prep, stem) -> None:
    """ÜRETİLEN görüntünün giysi bölgesini parser'la çıkar → geo_iou'nun girdisi.
    (Koşullama silüetini koşullamayla kıyaslamak ablation'da sabit çıkıyordu — bug'dı.)"""
    import cv2

    try:
        pred_img = np.asarray(Image.open(stem.with_suffix(".png")).convert("RGB"))
        pp = prep.process(pred_img, size=pred_img.shape[:2])
        h, w = pred_img.shape[:2]
        m = np.isin(cv2.resize(pp.parse, (w, h), interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
        Image.fromarray((m * 255).astype("uint8")).save(f"{stem}.predsil.png")
    except Exception as e:
        print(f"  predsil atlandı ({stem.name}): {e}", file=sys.stderr)


def run_variant(sampler, combos, ctx, control_scale: float, tag: str) -> dict:
    cfg, out_root, angles = ctx["cfg"], ctx["out_root"], ctx["angles"]
    prep = ctx["prep"]
    pred_dir = out_root / f"ckpt_{tag}" / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, (person, garment, bundle_by_angle) in enumerate(combos):
        for angle, bundle in bundle_by_angle.items():
            stem = (pred_dir / cfg["pred_pattern"].format(
                person_id=person.id, garment_id=garment.id, angle=angle)).with_suffix("")
            if not stem.with_suffix(".png").exists():
                pred = sampler.sample(bundle, control_scale=control_scale)
                Image.fromarray(pred).save(stem.with_suffix(".png"))
                tensor_to_pil(bundle.appearance_ref).save(f"{stem}.ref.png")
                Image.fromarray((bundle.inpaint_mask.numpy()[0] * 255).astype("uint8")).save(f"{stem}.mask.png")
                sil = (bundle.control_depth_sil[2].numpy() > 0).astype("uint8") * 255
                Image.fromarray(sil).save(f"{stem}.sil.png")
            if not Path(f"{stem}.predsil.png").exists():  # mevcut pred'lere de eklenir (ucuz)
                _save_predsil(prep, stem)
        print(f"[{tag} {i+1}/{len(combos)}] {person.id}×{garment.id}")
    summary = harness.evaluate(ctx["manifest"], pred_dir, cfg["pred_pattern"])
    harness.write_report(summary, out_root / f"ckpt_{tag}.json")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=10, help="ilk N kombo")
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=28)
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]

    # Koşullamaları BİR kez kur (iki varyant aynı bundle'ları kullanır — adil ablation)
    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    combos = []
    for combo in manifest.combos[: args.limit]:
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        try:
            pp = prep.process(manifest.root / person.image, size=size)
            # person_square_bbox ŞART: varsayılan tam-kare bbox gövdeyi görüntü
            # merkezine projekte eder (off-center kişide mesh kayar) — hizalama
            # düzeltmesi eval'de kullanılmıyordu (2026-08-09).
            params = hmr2(pp.image, bbox=person_square_bbox(pp))
            asset = load_garment_asset(
                garments_root / garment.mesh,
                texture_path=garments_root / garment.texture if garment.texture else None,
                garment_id=garment.id,
                allow_untextured=True,
            )
            bundles = {
                a: build_conditioning(
                    pp.image, params, asset,
                    PhotoView() if a == 0 else OrbitView(a),
                    size=size, person_prep=pp,
                )
                for a in args.angles
            }
            combos.append((person, garment, bundles))
        except Exception as e:
            print(f"ATLA {person.id}×{garment.id}: {e}", file=sys.stderr)
    if not combos:
        raise SystemExit("HATA: hiç kombo kurulamadı")

    sampler = FluxTryOnSampler(base["model"]["flux_fill_repo"], checkpoint=args.checkpoint,
                               prompt=base["model"]["prompt"])
    ctx = {"cfg": cfg, "out_root": out_root, "manifest": manifest, "angles": args.angles,
           "prep": prep}
    s_on = run_variant(sampler, combos, ctx, 1.0, "control_on")
    s_off = run_variant(sampler, combos, ctx, 0.0, "control_off")

    # KAPI: geo_iou = ÇIKTIDAKİ giysi bölgesi (parse) vs hedef silüet — AÇIK >= KAPALI
    get = lambda s, k: (s["overall"].get(k) or {}).get("mean")
    geo_on, geo_off = get(s_on, "geo_iou"), get(s_off, "geo_iou")
    de_on, de_off = get(s_on, "garment_delta_e"), get(s_off, "garment_delta_e")
    print(f"\n=== ABLATION KAPISI ===")
    print(f"geo_iou (çıktı-silüeti vs hedef): AÇIK={geo_on} KAPALI={geo_off} (yüksek iyi)")
    print(f"garment_delta_e                 : AÇIK={de_on} KAPALI={de_off} (düşük iyi)")
    if geo_on is not None and geo_off is not None:
        if geo_on < geo_off - 0.01:
            print("KAPI KALDI: kontrol AÇIK geometriyi bozuyor — v1 FAZ C senaryosu. Aşama 2'ye"
                  " GEÇME; çare sırası: ref_dropout artır (0.1→0.2) → devam eğitimi → FluxControlNet.")
            return 1
        print(f"KAPI GEÇTİ (geo_iou farkı: {geo_on - geo_off:+.4f}).")
        return 0
    print("UYARI: geo_iou hesaplanamadı (predsil eksik) — kapı ΔE'ye göre değerlendirildi.")
    if de_on is not None and de_off is not None and de_on > de_off + 1.0:
        print("KAPI KALDI (ΔE): kontrol AÇIK renk sadakatini bozuyor.")
        return 1
    print("KAPI GEÇTİ (ΔE bazlı, zayıf sinyal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
