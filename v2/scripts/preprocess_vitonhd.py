#!/usr/bin/env python3
"""VITON-HD gerçek verisini eğitim öğelerine dönüştürür (Colab, GPU).

Kişi başına: parser+openpose (agnostic+mask) → HMR2 (pred_cam kamerası) →
build_conditioning(garment=None): gövde-only normal/depth + parse'tan giysi
silüeti + ürün fotoğrafı referansı. Çıktı data/items.py sözleşmesiyle yazılır:

    {out}/{person_id}/gt.png agnostic.png mask.png normal.png depth_sil.png appearance_ref.png

Kullanım:
  python v2/scripts/preprocess_vitonhd.py \\
      --images data/raw/images --cloth data/raw/cloth --idm-repo /content/IDM-VTON [--limit 20]

Not: cloth/ dizini VITON-HD'nin düz ürün fotoğrafları (kişiyle aynı dosya adı).
v1 Drive'ında yoksa VITON-HD zip'inden 'cloth' klasörü de açılmalı.
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
from meshvton2.conditioning.builder import PhotoView, assert_real_impl, build_conditioning  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.utils.image_utils import tensor_to_pil  # noqa: E402


ATR_GARMENT = (4, 7)  # upper_clothes, dress


def garment_ref_from_person(image: np.ndarray, parse: np.ndarray, pad: float = 0.08) -> np.ndarray | None:
    """Kişinin üzerindeki giysiyi parse maskesiyle kesip beyaz zemine koyar —
    ürün fotoğrafı ikamesi. (H,W,3) uint8 döner; giysi bölgesi yoksa None."""
    import cv2

    h, w = image.shape[:2]
    m = np.isin(cv2.resize(parse, (w, h), interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
    if m.mean() < 0.02:  # giysi bölgesi yok/çok küçük
        return None
    ys, xs = np.where(m)
    py, px = int((ys.max() - ys.min()) * pad), int((xs.max() - xs.min()) * pad)
    y0, y1 = max(0, ys.min() - py), min(h, ys.max() + py)
    x0, x1 = max(0, xs.min() - px), min(w, xs.max() + px)
    out = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    crop_m = m[y0:y1, x0:x1]
    out[crop_m] = image[y0:y1, x0:x1][crop_m]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--cloth", type=Path, default=None,
                    help="VITON-HD ürün fotoğrafları dizini (opsiyonel; yoksa referans "
                         "kişinin ÜZERİNDEKİ giysiden parse maskesiyle kesilir)")
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    out = args.out or REPO / "v2/data/vitonhd_items"
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".png"})
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"HATA: {args.images} altında görüntü yok")

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()

    done, failed = 0, []
    for i, img_path in enumerate(images):
        item_dir = out / img_path.stem
        if (item_dir / "depth_sil.png").exists():  # resume: bitmiş öğeyi atla
            done += 1
            continue
        cloth_path = None
        if args.cloth:
            cloth_path = args.cloth / img_path.name
            if not cloth_path.exists():
                cloth_path = cloth_path.with_suffix(".png" if img_path.suffix == ".jpg" else ".jpg")
            if not cloth_path.exists():
                cloth_path = None
        try:
            pp = prep.process(img_path, size=size)
            params = hmr2(pp.image, bbox=person_square_bbox(pp))  # kişi-merkezli kare (hizalama)
            if cloth_path is not None:
                ref = np.asarray(Image.open(cloth_path).convert("RGB"))
            else:
                # Ürün fotoğrafı yoksa: kişinin GİYDİĞİ giysiyi parse maskesiyle kes —
                # süpervizyon yine tutarlı (referans = GT'deki giysinin kendisi)
                ref = garment_ref_from_person(pp.image, pp.parse)
                if ref is None:
                    failed.append(f"{img_path.stem}: parse'ta giysi bölgesi yok")
                    continue
            bundle = build_conditioning(
                pp.image, params, None, PhotoView(), size=size,
                person_prep=pp, appearance_ref_image=ref,
            )
            item_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pp.image).save(item_dir / "gt.png")
            tensor_to_pil(bundle.agnostic_rgb).save(item_dir / "agnostic.png")
            Image.fromarray((bundle.inpaint_mask.numpy()[0] * 255).astype("uint8")).save(item_dir / "mask.png")
            tensor_to_pil(bundle.appearance_ref).save(item_dir / "appearance_ref.png")
            tensor_to_pil(bundle.control_normal).save(item_dir / "normal.png")
            tensor_to_pil(bundle.control_depth_sil).save(item_dir / "depth_sil.png")
            done += 1
        except Exception as e:
            failed.append(f"{img_path.stem}: {e}")
            if len(failed) == 1:
                import traceback

                traceback.print_exc(file=sys.stderr)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(images)}] tamam={done} hata={len(failed)}")

    print(f"\nBitti: {done}/{len(images)} öğe → {out}; hata={len(failed)}")
    for f in failed[:5]:
        print(f"  {f}", file=sys.stderr)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
