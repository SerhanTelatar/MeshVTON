#!/usr/bin/env python3
"""Inference ayar taraması (EĞİTİM YOK) — guidance/adım renk sadakatini düzeltir mi?

final.pt'yi bir kez yükler, birkaç kombo için birkaç guidance değerinde örnekler,
etiketli bir karşılaştırma grid'i yazar. Solgun renk çaresi guidance olabilir
(FLUX Fill guidance-damıtılmış; eğitim 1.0, baseline 30 kullanmıştı).

Kullanım (Colab, A100):
  python v2/scripts/tune_inference.py \\
      --checkpoint /content/drive/MyDrive/MeshVTON/v2_outputs/stage1/final.pt \\
      --idm-repo /content/IDM-VTON --guidances 1 3.5 7.5 15 30 --combos 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from meshvton2.conditioning.body import build_hmr2_backend  # noqa: E402
from meshvton2.conditioning.builder import PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.model.flux_tryon import FluxTryOnSampler  # noqa: E402
from meshvton2.utils.image_utils import tensor_to_pil  # noqa: E402


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 0))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--guidances", type=float, nargs="+", default=[1.0, 3.5, 7.5, 15.0, 30.0])
    ap.add_argument("--combos", type=int, default=3)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thumb", type=int, default=256)
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    out_root = REPO / base["paths"]["eval_results"]

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}

    # Desen çeşitliliği için ilk N FARKLI giysili kombo (aynı kişide farklı giysiler)
    combos, seen_g = [], set()
    for combo in manifest.combos:
        if combo.garment_id in seen_g:
            continue
        person, garment = by_pid[combo.person_id], by_gid[combo.garment_id]
        try:
            pp = prep.process(manifest.root / person.image, size=size)
            params = hmr2(pp.image)
            asset = load_garment_asset(
                garments_root / garment.mesh,
                texture_path=garments_root / garment.texture if garment.texture else None,
                garment_id=garment.id,
            )
            bundle = build_conditioning(pp.image, params, asset, PhotoView(), size=size, person_prep=pp)
            combos.append((person, garment, bundle))
            seen_g.add(combo.garment_id)
        except Exception as e:
            print(f"ATLA {combo.person_id}×{combo.garment_id}: {e}", file=sys.stderr)
        if len(combos) >= args.combos:
            break
    if not combos:
        raise SystemExit("HATA: kombo kurulamadı")

    sampler = FluxTryOnSampler(base["model"]["flux_fill_repo"], checkpoint=args.checkpoint,
                               prompt=base["model"]["prompt"])

    # Grid: her satır bir kombo; sütun 0 = giysi referansı, sonrası her guidance
    tw = args.thumb
    th = int(tw * size[0] / size[1])
    cols = 1 + len(args.guidances)
    grid = Image.new("RGB", (cols * tw, len(combos) * th), (30, 30, 30))
    for r, (person, garment, bundle) in enumerate(combos):
        ref = tensor_to_pil(bundle.appearance_ref).resize((tw, th))
        grid.paste(label(ref, "REF"), (0, r * th))
        for c, g in enumerate(args.guidances):
            pred = sampler.sample(bundle, steps=args.steps, seed=args.seed, guidance=g)
            im = label(Image.fromarray(pred).resize((tw, th)), f"g={g:g}")
            grid.paste(im, ((c + 1) * tw, r * th))
            print(f"[{person.id}×{garment.id}] guidance={g:g} OK")

    out = out_root / "guidance_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"\nGrid: {out}  (satır=kombo, sütun=guidance; solgunluk hangi g'de düzeliyor?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
