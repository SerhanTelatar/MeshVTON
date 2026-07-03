"""Giysi varlığı yükleme (v1 garment_draper.load_garment_mesh'ten uyarlandı).

v2 kuralları:
- trimesh yoksa sahte mesh DÖNMEZ, ImportError yükselir (v1 sessiz-fallback dersi).
- Gerçek texture bulunamazsa ValueError yükselir — texturesiz mesh pipeline'a
  giremez (v1 Probe D: gri render = desen halüsinasyonunun ana nedeni).
  Bilinçli istisna için allow_untextured=True.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from meshvton2.conditioning.builder import GarmentAsset


def find_sibling_texture(obj_path: str | Path) -> str | None:
    """CLOTH3D: .obj yanında .mtl'siz ayrık texture PNG. Sıra: aynı ad →
    isim ipucu (tex/atlas/diffuse/albedo/color/uv) → klasördeki tek görüntü."""
    obj_path = str(obj_path)
    d = os.path.dirname(os.path.abspath(obj_path))
    stem = os.path.splitext(os.path.basename(obj_path))[0]
    exts = ("png", "jpg", "jpeg", "bmp")
    for e in exts:
        cand = os.path.join(d, f"{stem}.{e}")
        if os.path.exists(cand):
            return cand
    imgs: list[str] = []
    for e in exts:
        imgs += glob.glob(os.path.join(d, f"*.{e}"))
    if not imgs:
        return None
    for hint in ("tex", "atlas", "diffuse", "albedo", "color", "uv"):
        for f in imgs:
            if hint in os.path.basename(f).lower():
                return f
    return imgs[0] if len(imgs) == 1 else None


def _is_real_texture(a) -> bool:
    """trimesh .mtl yokken 2x2 placeholder üretir — gerçek texture >= 8px olmalı."""
    return a is not None and getattr(a, "ndim", 0) == 3 and min(a.shape[:2]) >= 8


def load_garment_asset(
    mesh_path: str | Path,
    texture_path: str | Path | None = None,
    *,
    garment_id: str | None = None,
    allow_untextured: bool = False,
) -> GarmentAsset:
    """Mesh + texture yükler. Texture önceliği: açık yol > sibling dosya >
    (yalnız gerçekse) mesh materyali. Hiçbiri yoksa ValueError."""
    import trimesh  # yoksa ImportError yükselsin — sahte mesh fallback YOK
    from PIL import Image

    mesh_path = Path(mesh_path)
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    uv = None
    if getattr(mesh.visual, "uv", None) is not None:
        cand = np.asarray(mesh.visual.uv, dtype=np.float32)
        if cand.shape == (len(verts), 2):
            uv = cand

    texture = None
    if texture_path and os.path.exists(str(texture_path)):
        cand = np.asarray(Image.open(texture_path).convert("RGB"))
        if _is_real_texture(cand):
            texture = cand
    if texture is None:
        sib = find_sibling_texture(mesh_path)
        if sib is not None:
            cand = np.asarray(Image.open(sib).convert("RGB"))
            if _is_real_texture(cand):
                texture = cand
    if texture is None:
        mimg = getattr(getattr(mesh.visual, "material", None), "image", None)
        if mimg is not None:
            cand = np.asarray(mimg)[:, :, :3]
            if _is_real_texture(cand):
                texture = cand

    if texture is None and not allow_untextured:
        raise ValueError(
            f"{mesh_path}: gerçek texture bulunamadı (açık yol / sibling PNG / materyal). "
            "Texturesiz mesh pipeline'a giremez (gri render = halüsinasyon, v1 Probe D). "
            "texture_path verin veya bilinçli olarak allow_untextured=True kullanın."
        )
    if texture is not None and uv is None and not allow_untextured:
        raise ValueError(
            f"{mesh_path}: texture var ama geçerli per-vertex UV yok — desen render edilemez."
        )

    return GarmentAsset(
        garment_id=garment_id or mesh_path.parent.name,
        verts=verts,
        faces=faces,
        uv=uv,
        texture=texture,
    )
