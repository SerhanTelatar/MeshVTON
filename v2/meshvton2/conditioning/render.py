"""PyTorch3D render yardımcıları.

Faz 1: yalnız görünüm referansı (flat-lit dokulu UV render — ürün fotoğrafı gibi).
Faz 2: ekran-uzayı normal/depth/silüet + explicit PerspectiveCameras buraya eklenecek
(v1 src/modules/mesh_renderer.py'den taşınarak).

pytorch3d import'ları tembeldir: CPU-only geliştirme makinesinde modül import
edilebilir kalır, render çağrısı net bir hata verir.
"""

from __future__ import annotations

import numpy as np
import torch

from meshvton2.conditioning.builder import GarmentAsset


def zup_to_yup(verts: np.ndarray) -> np.ndarray:
    """CLOTH3D Z-up -> PyTorch3D/kamera Y-up: (x,y,z) -> (x,z,-y). (v1 _geometric_align kuralı)"""
    return np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=1)


def center_unit(verts: np.ndarray) -> np.ndarray:
    """Merkeze al, en büyük eksen yarıçapını 1'e ölçekle (kamera kadrajı için)."""
    v = verts - verts.mean(axis=0, keepdims=True)
    r = np.abs(v).max()
    return v / max(r, 1e-8)


def render_appearance_ref(
    asset: GarmentAsset,
    *,
    size: tuple[int, int] = (1024, 768),
    azim: float = 0.0,
    dist: float = 2.5,
    convert_zup: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Giysinin flat-lit (gölgesiz, ürün-foto benzeri) dokulu ön render'ı.

    Returns:
        (H,W,3) uint8 RGB, beyaz arka plan.
    """
    if asset.texture is None or asset.uv is None:
        raise ValueError(
            f"{asset.garment_id}: appearance ref için texture+UV şart "
            "(load_garment_asset bunu garanti eder)."
        )
    try:
        from pytorch3d.renderer import (
            AmbientLights,
            BlendParams,
            FoVPerspectiveCameras,
            MeshRasterizer,
            MeshRenderer,
            RasterizationSettings,
            SoftPhongShader,
            TexturesUV,
            look_at_view_transform,
        )
        from pytorch3d.structures import Meshes
    except ImportError as e:
        raise ImportError(
            "pytorch3d gerekli (Colab: pip install "
            '"git+https://github.com/facebookresearch/pytorch3d.git")'
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = size

    verts_np = asset.verts
    if convert_zup:
        verts_np = zup_to_yup(verts_np)
    verts_np = center_unit(verts_np)

    verts = torch.from_numpy(verts_np).float().unsqueeze(0).to(device)
    faces = torch.from_numpy(asset.faces).long().to(device)
    tex = TexturesUV(
        maps=torch.from_numpy(asset.texture.astype(np.float32) / 255.0).unsqueeze(0).to(device),
        faces_uvs=faces.unsqueeze(0),
        verts_uvs=torch.from_numpy(asset.uv).float().unsqueeze(0).to(device),
        padding_mode="border",
        align_corners=True,
    )
    meshes = Meshes(verts=verts, faces=faces.unsqueeze(0), textures=tex)

    R, T = look_at_view_transform(dist=dist, elev=0.0, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R.to(device), T=T.to(device))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(image_size=(h, w), faces_per_pixel=1),
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=AmbientLights(device=device),  # gölgesiz: yalnız ambient (v1 FAZ A1)
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        ),
    )
    with torch.no_grad():
        rgba = renderer(meshes)[0]  # (H,W,4) [0,1]
    return (rgba[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
