"""Sentetik multi-view veri üretici (Faz 3).

Her örnek = 1 kimlik (β, poz, kadraj) × 1 giysi × 4 görüş. TÜM koşullama
build_conditioning ile üretilir — sentetik üretim, parite sözleşmesinin ölçekli
testidir. Dizin sözleşmesi (plan):

    {out}/{sample_id}/
      meta.json                      # betas, body_pose, garment_id, görüş başına CameraSpec
      appearance_ref.png
      view_{000,090,180,270}/{gt,agnostic,mask,normal,depth_sil}.png
    {out}/pairs.csv                  # sample_id,garment_id

Reddetme: clearance_ratio > eşik (varsayılan 0.02) → örnek atlanır (drape bozuk).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from meshvton2.conditioning.builder import ConditioningBundle, GarmentAsset, OrbitView, build_conditioning
from meshvton2.synth.bodies import load_pose_bank, sample_identity

VIEWS = (0, 90, 180, 270)
DEPTH_REJECT = 0.03   # ortalama penetrasyon derinliği [m]: mm=normal temas, 3cm+=yanlış yerleşim
EXTENT_REJECT = 1.5   # giysi/gövde bbox-diyagonal oranı üst sınırı (patlama kapısı)
# NOT: "itilen vertex oranı" (clearance_ratio) artık RED kriteri DEĞİL — oturan
# giyside teması olan vertex çoktur; bozukluk sinyali derinlik + boyut oranıdır.


def _save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _t2u8(t) -> np.ndarray:
    """(C,H,W) [-1,1] -> (H,W,C) uint8; tek kanal -> (H,W)."""
    a = ((t.numpy().transpose(1, 2, 0) + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)
    return a[..., 0] if a.shape[2] == 1 else a


def _mask_u8(t) -> np.ndarray:
    return (t.numpy()[0] * 255).astype(np.uint8)


def write_sample(out_dir: Path, sample_id: str, bundles: dict[int, ConditioningBundle], smplx_params: dict) -> None:
    sd = out_dir / sample_id
    first = bundles[VIEWS[0]]
    _save_png(sd / "appearance_ref.png", _t2u8(first.appearance_ref))
    meta = {
        "garment_id": first.meta["garment_id"],
        "betas": np.asarray(smplx_params["betas"]).tolist(),
        "body_pose": np.asarray(smplx_params["body_pose"]).tolist(),
        "pred_cam": np.asarray(smplx_params["pred_cam"]).tolist(),
        "views": {str(v): b.camera.to_dict() for v, b in bundles.items()},
        "clearance_ratio": first.meta.get("clearance_ratio"),
    }
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "meta.json").write_text(json.dumps(meta))
    for v, b in bundles.items():
        vd = sd / f"view_{v:03d}"
        if "gt_rgb" in b.meta:
            _save_png(vd / "gt.png", _t2u8(b.meta["gt_rgb"]))
        _save_png(vd / "agnostic.png", _t2u8(b.agnostic_rgb))
        _save_png(vd / "mask.png", _mask_u8(b.inpaint_mask))
        _save_png(vd / "normal.png", _t2u8(b.control_normal))
        _save_png(vd / "depth_sil.png", _t2u8(b.control_depth_sil))


def generate(
    garment_assets: list[GarmentAsset],
    out_dir: str | Path,
    *,
    num_samples: int,
    size: tuple[int, int] = (1024, 768),
    seed: int = 0,
    poses_file: str | None = None,
    depth_reject: float = DEPTH_REJECT,
    log=print,
) -> dict:
    """-> {written, rejected, failed}. pairs.csv'ye ekler (append, header'lı ilk yazım)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    pose_bank = load_pose_bank(poses_file)
    pairs_path = out_dir / "pairs.csv"
    new_header = not pairs_path.exists()

    written, rejected, failed = 0, 0, []
    with pairs_path.open("a", newline="") as fh:
        wcsv = csv.writer(fh)
        if new_header:
            wcsv.writerow(["sample_id", "garment_id"])
        for i in range(num_samples):
            # seed ofseti: paralel işçiler aynı giysi sırasını üretmesin
            asset = garment_assets[(i + seed) % len(garment_assets)]
            params = sample_identity(rng, size, pose_bank)
            sample_id = f"s{seed:03d}_{i:06d}"
            try:
                bundles = {
                    v: build_conditioning(None, params, asset, OrbitView(v), size=size)
                    for v in VIEWS
                }
            except Exception as e:
                failed.append(f"{sample_id}: {e}")
                log(f"[{i+1}/{num_samples}] HATA {sample_id}: {e}")
                continue
            meta0 = bundles[VIEWS[0]].meta
            depth = meta0.get("penetration_depth") or 0.0
            er = meta0.get("drape_extent_ratio") or 0.0
            if depth > depth_reject or er > EXTENT_REJECT:
                rejected += 1
                log(f"[{i+1}/{num_samples}] RED {sample_id}: derinlik={depth*100:.1f}cm extent={er:.2f}")
                continue
            write_sample(out_dir, sample_id, bundles, params)
            wcsv.writerow([sample_id, asset.garment_id])
            written += 1
            log(f"[{i+1}/{num_samples}] OK {sample_id} ({asset.garment_id})")
    return {"written": written, "rejected": rejected, "failed": failed}
