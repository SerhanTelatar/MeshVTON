#!/usr/bin/env python3
"""GUIDANCE taraması — solgun/yarı saydam çıktının ucuz çaresi var mı?

Bulgu (2026-08-10): kalibrasyon (hang -0.12, ölçek 1.25) sonrası yerleşim düzeldi
ama çıktı hâlâ "kumaş gibi" durmuyor — düz gri kütle, kıvrım/gölge yok. Sebep
eğitim bütçesi: 1000 adımda akış-eşleme kaybı KOŞULLU ORTALAMAYA yakınsıyor,
o da tüm olası kıvrım desenlerinin ortalaması = düz leke. Sentetik GT'de kıvrım
VAR (gölgeli gri render), model henüz üretmiyor.

Eğitmeden denenecek tek kol: FLUX Fill guidance-damıtılmış; yüksek guidance
(3.5-30) sadakat/doygunluğu artırır (bkz. flux_tryon.py FluxTryOnSampler.sample
docstring: "solgunluk çaresi"). Şimdiye kadar hep 1.0 (eğitimdeki değer) kullanıldı.

Notebook'tan BAĞIMSIZ: kendi prep/hmr2/sampler'ını kurar.

Kullanım:
  python v2/scripts/sweep_guidance.py --checkpoint <Drive>/stage1/final.pt \\
      --idm-repo /content/IDM-VTON [--person 00000_00] [--garment upper_body__00047_Top]
      [--guidance 1.0 3.5 7.0 15.0] [--steps 28]
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
    ap.add_argument("--person", default=None, help="varsayılan: manifest'teki ilk kişi")
    ap.add_argument("--garment", default=None, help="varsayılan: manifest'teki ilk giysi")
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.5, 7.0, 15.0])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hang-pad", type=float, default=PHOTO_HANG_PAD)
    ap.add_argument("--garment-scale", type=float, default=PHOTO_GARMENT_SCALE)
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_dir = REPO / base["paths"]["eval_results"] / "guidance_sweep"
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
    # Koşullama BİR kez: tarama boyunca tek değişken guidance olsun
    bundle = build_conditioning(pp.image, params, asset, PhotoView(), size=size,
                                person_prep=pp, hang_pad=args.hang_pad,
                                garment_scale=args.garment_scale)

    sampler = FluxTryOnSampler(base["model"]["flux_fill_repo"], checkpoint=args.checkpoint,
                               prompt=base["model"]["prompt"])
    print(f"kişi={pid} giysi={gid} hang={args.hang_pad:+.2f} ölçek={args.garment_scale:.2f} "
          f"seed={args.seed}\n")

    outs = []
    for g in args.guidance:
        img = sampler.sample(bundle, steps=args.steps, seed=args.seed,
                             control_scale=1.0, guidance=float(g))
        fn = out_dir / f"{pid}__{gid}__g{g:g}.png"
        Image.fromarray(img).save(fn)
        # Kontrast/doygunluk vekili: maske içi std yükseliyorsa doku BELİRİYOR demektir
        m = bundle.inpaint_mask.numpy()[0] > 0.5
        std = float(np.asarray(img, np.float32)[m].std())
        outs.append((g, img, std))
        print(f"  guidance={g:>5.1f} → {fn.name}   maske-içi std={std:6.2f}")

    h = round(240 * size[0] / size[1])
    strip = Image.new("RGB", (240 * len(outs), h), "white")
    for k, (_, img, _) in enumerate(outs):
        strip.paste(Image.fromarray(img).resize((240, h), Image.LANCZOS), (k * 240, 0))
    sp = out_dir / f"{pid}__{gid}__sweep.png"
    strip.save(sp)

    best = max(outs, key=lambda r: r[2])
    print(f"\nŞerit: {sp}  (soldan sağa: {', '.join(f'g={g:g}' for g, *_ in outs)})")
    print(f"EN YÜKSEK DOKU (maske-içi std): guidance={best[0]:g} → {best[2]:.2f}")
    print("NOT: std yalnız VEKİL bir ölçüt — yüksek std bazen gürültü/artefakt da olabilir.")
    print("     Kararı ŞERİDE BAKARAK verin; kumaş kıvrımı mı, bozulma mı ayırt edin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())