#!/usr/bin/env python3
"""Faz 1 — Zero-shot taban çizgisi (eğitimsiz FLUX).

Golden set'in her (kişi, giysi) kombosu için (yalnız 0° = foto görüşü; orbit
görüşler Faz 2 kamerasını gerektirir):
  1. Görünüm referansı: giysinin flat-lit dokulu ön render'ı (pytorch3d)
  2. Agnostic + maske: PersonPreprocessor (IDM-VTON preprocess, densepose YOK)
  3. FluxTryOn(variant) ile üretim
  4. Tahmin + aux dosyaları eval sözleşmesiyle yazılır, harness koşulur, grid PNG

Kullanım (Colab, GPU):
  python v2/scripts/zero_shot_baseline.py --variant fill_spatial --idm-repo /content/IDM-VTON
  python v2/scripts/zero_shot_baseline.py --variant kontext      --idm-repo /content/IDM-VTON
  # hızlı duman testi: --limit 4
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

from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor  # noqa: E402
from meshvton2.conditioning.render import force_textureless, render_appearance_ref  # noqa: E402
from meshvton2.eval import harness  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.model.flux_tryon import VARIANTS, FluxTryOn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--idm-repo", type=Path, required=True, help="Clone'lanmış IDM-VTON yolu (preprocess için)")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="İlk N kombo (duman testi)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload", action="store_true", help="CPU offload (düşük VRAM)")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    h, w = base["resolution"]["height"], base["resolution"]["width"]
    manifest = load_manifest(args.manifest or REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]
    pred_dir = out_root / f"phase1_{args.variant}" / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    prep = PersonPreprocessor(args.idm_repo)
    model = FluxTryOn(
        args.variant,
        fill_repo=base["model"]["flux_fill_repo"],
        kontext_repo=base["model"]["flux_kontext_repo"],
        device="cuda_offload" if args.offload else base["device"],
        steps=args.steps,
    )

    # Önbellekler: kişi başı prep, giysi başı appearance ref (kombo döngüsünde tekrar yok)
    prep_cache: dict[str, object] = {}
    ref_cache: dict[str, np.ndarray] = {}
    combos = manifest.combos[: args.limit] if args.limit else manifest.combos
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}

    done, failed = 0, []
    for i, combo in enumerate(combos):
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        try:
            if person.id not in prep_cache:
                prep_cache[person.id] = prep.process(manifest.root / person.image, size=(h, w))
            pp = prep_cache[person.id]
            if garment.id not in ref_cache:
                asset = load_garment_asset(
                    garments_root / garment.mesh,
                    texture_path=garments_root / garment.texture if garment.texture else None,
                    garment_id=garment.id,
                    allow_untextured=True,
                )
                ref_cache[garment.id] = render_appearance_ref(force_textureless(asset), size=(h, w))
            ref = ref_cache[garment.id]

            pred = model.tryon(pp.image, pp.agnostic, pp.mask, ref, seed=args.seed)

            stem = pred_dir / cfg["pred_pattern"].format(
                person_id=person.id, garment_id=garment.id, angle=0
            )
            stem = stem.with_suffix("")  # .png pattern'den gelir
            Image.fromarray(pred).save(stem.with_suffix(".png"))
            Image.fromarray(pp.image).save(f"{stem}.person.png")
            Image.fromarray(pp.mask).save(f"{stem}.mask.png")
            Image.fromarray(ref).save(f"{stem}.ref.png")
            done += 1
            print(f"[{i+1}/{len(combos)}] {person.id} × {garment.id} OK")
        except Exception as e:  # tek kombo hatası koşuyu öldürmesin, ama SAKLANMASIN
            failed.append(f"{person.id}×{garment.id}: {e}")
            print(f"[{i+1}/{len(combos)}] HATA {person.id}×{garment.id}: {e}", file=sys.stderr)
            if len(failed) == 1:  # ilk hatanın tam traceback'i — teşhis için
                import traceback

                traceback.print_exc(file=sys.stderr)

    # Değerlendirme + grid
    summary = harness.evaluate(manifest, pred_dir, cfg["pred_pattern"])
    summary["failed"] = failed
    harness.write_report(summary, out_root / f"phase1_{args.variant}.json")
    _save_grid(pred_dir, out_root / f"phase1_{args.variant}_grid.png")
    print(f"\n{done}/{len(combos)} üretildi, {len(failed)} hata.")
    print(f"Rapor: {out_root / f'phase1_{args.variant}.json'} (+.md), grid: {out_root / f'phase1_{args.variant}_grid.png'}")
    return 0 if done else 1


def _save_grid(pred_dir: Path, out_path: Path, cols: int = 5, thumb_w: int = 256) -> None:
    preds = sorted(p for p in pred_dir.glob("*.png") if len(p.name.split(".")) == 2)  # aux hariç
    if not preds:
        return
    thumbs = []
    for p in preds:
        img = Image.open(p)
        th = int(img.height * thumb_w / img.width)
        thumbs.append(img.resize((thumb_w, th), Image.LANCZOS))
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_w, rows * thumbs[0].height), "white")
    for i, t in enumerate(thumbs):
        grid.paste(t, ((i % cols) * thumb_w, (i // cols) * t.height))
    grid.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
