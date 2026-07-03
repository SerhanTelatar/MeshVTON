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
    backend: str = "auto",
) -> np.ndarray:
    """Giysinin flat-lit (gölgesiz, ürün-foto benzeri) dokulu ön render'ı.

    backend: "auto" (pytorch3d varsa o, yoksa pyrender) | "pytorch3d" | "pyrender".
    pyrender saniyeler içinde pip'lenir (Colab'da derleme YOK) — Faz 1 bunu kullanır;
    pytorch3d Faz 2'nin ekran-uzayı render'ları için gelecek.

    Returns:
        (H,W,3) uint8 RGB, beyaz arka plan.
    """
    if asset.texture is None or asset.uv is None:
        raise ValueError(
            f"{asset.garment_id}: appearance ref için texture+UV şart "
            "(load_garment_asset bunu garanti eder)."
        )
    if backend == "auto":
        try:
            import pytorch3d  # noqa: F401

            backend = "pytorch3d"
        except ImportError:
            backend = "pyrender"
    if backend == "pyrender":
        return _render_pyrender(asset, size=size, azim=azim, dist=dist, convert_zup=convert_zup)
    return _render_pytorch3d(asset, size=size, azim=azim, dist=dist, convert_zup=convert_zup, device=device)


def _prep_verts(asset: GarmentAsset, convert_zup: bool, azim: float) -> np.ndarray:
    """Ortak hazırlık: eksen dönüşümü, merkez+ölçek, azim için Y-ekseni dönüşü
    (kamerayı döndürmek yerine mesh'i -azim döndürmek eşdeğerdir)."""
    v = asset.verts
    if convert_zup:
        v = zup_to_yup(v)
    v = center_unit(v)
    if azim:
        a = np.deg2rad(-azim)
        rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
        v = v @ rot.T
    return v


def _render_pyrender(asset, *, size, azim, dist, convert_zup) -> np.ndarray:
    import os

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # Colab headless GPU
    import pyrender
    import trimesh
    from PIL import Image

    h, w = size
    verts = _prep_verts(asset, convert_zup, azim)
    tm = trimesh.Trimesh(vertices=verts, faces=asset.faces, process=False)
    tm.visual = trimesh.visual.TextureVisuals(uv=asset.uv, image=Image.fromarray(asset.texture))
    scene = pyrender.Scene(ambient_light=np.ones(3), bg_color=(1.0, 1.0, 1.0, 1.0))
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0), aspectRatio=w / h)
    pose = np.eye(4)
    pose[2, 3] = dist
    scene.add(cam, pose=pose)
    renderer = pyrender.OffscreenRenderer(w, h)
    try:
        # FLAT: ışıklama yok, texture olduğu gibi — ürün fotoğrafı görünümü
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    finally:
        renderer.delete()
    return np.asarray(color[..., :3], dtype=np.uint8)


def _render_pytorch3d(asset, *, size, azim, dist, convert_zup, device) -> np.ndarray:
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
    verts_np = _prep_verts(asset, convert_zup, azim)  # azim mesh'e bakıldı

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

    R, T = look_at_view_transform(dist=dist, elev=0.0, azim=0.0)  # azim _prep_verts'te uygulandı
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
