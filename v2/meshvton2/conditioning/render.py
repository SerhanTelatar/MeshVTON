"""Render yardımcıları (pyrender öncelikli — Colab'da derleme YOK).

- Faz 1: görünüm referansı (flat-lit dokulu UV render — ürün fotoğrafı gibi).
- Faz 2: ekran-uzayı geometri pass'leri (`render_geometry`): kamera-uzayı normal +
  depth + giysi silüeti, CameraSpec'in K/R/T'siyle birebir (IntrinsicsCamera).
  Normal pass'i "normalleri vertex-rengi yap + FLAT render" hilesiyle pyrender'da
  çözülür; pytorch3d şart değil.

Koordinat notu: CameraSpec OpenCV sözleşmesidir (+z ileri, +y aşağı); pyrender
OpenGL ister (-z ileri, +y yukarı) — dönüşüm `_cv_pose_to_gl`.

3D import'ları tembeldir: CPU-only geliştirme makinesinde modül import edilebilir
kalır, render çağrısı net bir hata verir.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from meshvton2.conditioning.builder import CameraSpec, GarmentAsset


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


# --------------------------------------------------------------------------- #
# Faz 2: ekran-uzayı geometri pass'leri (explicit CameraSpec, pyrender)
# --------------------------------------------------------------------------- #


def _cv_pose_to_gl(camera: CameraSpec) -> np.ndarray:
    """OpenCV world2cam [R|T] -> pyrender/OpenGL cam2world pose (4x4).
    X_gl = D·X_cv, D=diag(1,-1,-1)  =>  pose = [[RᵀD, -RᵀT], [0,1]]."""
    R = np.asarray(camera.R, np.float64)
    T = np.asarray(camera.T, np.float64)
    D = np.diag([1.0, -1.0, -1.0])
    pose = np.eye(4)
    pose[:3, :3] = R.T @ D
    pose[:3, 3] = -R.T @ T
    return pose


def _intrinsics_camera(camera: CameraSpec, pyrender):
    K = np.asarray(camera.K, np.float64)
    return pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])


def _camera_space_normals(verts: np.ndarray, faces: np.ndarray, camera: CameraSpec) -> np.ndarray:
    """Vertex normallerini kamera uzayına çevirir; renk kodlaması [0,1] = (n+1)/2.
    Kamera-uzayı normal, görüşten bağımsız aynı yüzeye aynı rengi verir (koşullama
    tutarlılığı) — v1'in dünya-uzayı normal'inin aksine."""
    from meshvton2.conditioning.lbs_drape import vertex_normals

    n_world = vertex_normals(verts, faces)
    R = np.asarray(camera.R, np.float64)
    n_cam = n_world @ R.T
    return ((n_cam + 1.0) / 2.0).clip(0, 1)


def _flat_scene(pyrender, bg=(0.0, 0.0, 0.0)):
    return pyrender.Scene(ambient_light=np.ones(3), bg_color=(*bg, 0.0))


def _add_colored(pyrender, scene, verts, faces, colors01):
    import trimesh

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tm.visual = trimesh.visual.ColorVisuals(
        tm, vertex_colors=(np.clip(colors01, 0, 1) * 255).astype(np.uint8)
    )
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))


def render_geometry(
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    garment_verts: np.ndarray,
    garment_faces: np.ndarray,
    camera: CameraSpec,
    *,
    size: tuple[int, int] = (1024, 768),
) -> dict:
    """Kontrol kanalları için ekran-uzayı pass'leri (hepsi CameraSpec kamerasıyla):

    Returns dict:
        normal   (H,W,3) float32 [0,1] — kamera-uzayı sahne normalleri (gövde+giysi)
        depth    (H,W)   float32 [0,1] — sahne maskesi içinde min-max normalize; arka plan 0
        garment_sil (H,W) bool         — yalnız giysinin görünür silüeti (derinlik testli)
        scene_mask  (H,W) bool         — gövde∪giysi maskesi
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    h, w = size
    pose = _cv_pose_to_gl(camera)
    cam = _intrinsics_camera(camera, pyrender)

    def _render(scene):
        scene.add(cam, pose=pose)
        r = pyrender.OffscreenRenderer(w, h)
        try:
            color, depth = r.render(scene, flags=pyrender.RenderFlags.FLAT)
        finally:
            r.delete()
        return color, depth

    # Pass 1: birleşik sahne — normal renkleri + depth buffer
    scene = _flat_scene(pyrender)
    _add_colored(pyrender, scene, body_verts, body_faces,
                 _camera_space_normals(body_verts, body_faces, camera))
    _add_colored(pyrender, scene, garment_verts, garment_faces,
                 _camera_space_normals(garment_verts, garment_faces, camera))
    normal_rgb, depth_raw = _render(scene)
    scene_mask = depth_raw > 0

    # Pass 2: giysi kimlik rengi (derinlik testi gövde önünü/arkasını doğru ayırır)
    scene2 = _flat_scene(pyrender)
    _add_colored(pyrender, scene2, body_verts, body_faces, np.zeros((len(body_verts), 3)))
    _add_colored(pyrender, scene2, garment_verts, garment_faces, np.ones((len(garment_verts), 3)))
    idmap, _ = _render(scene2)
    garment_sil = idmap[..., 0] > 127

    depth = np.zeros((h, w), np.float32)
    if scene_mask.any():
        d = depth_raw[scene_mask]
        lo, hi = float(d.min()), float(d.max())
        span = max(hi - lo, 1e-6)
        # yakın=1, uzak→0'a: koşullamada yakınlık sinyali pozitif olsun
        depth[scene_mask] = 1.0 - (depth_raw[scene_mask] - lo) / span

    return {
        "normal": normal_rgb.astype(np.float32) / 255.0,
        "depth": depth,
        "garment_sil": garment_sil,
        "scene_mask": scene_mask,
    }


def render_textured_scene(
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    body_colors01: np.ndarray,
    garment_verts: np.ndarray,
    garment_asset: GarmentAsset,
    camera: CameraSpec,
    *,
    size: tuple[int, int] = (1024, 768),
    bg=(0.82, 0.82, 0.84),  # VITON-HD stüdyo grisi
) -> np.ndarray:
    """Sentetik GT: dokulu giysi + renkli gövde, flat ışık. (H,W,3) uint8."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    from PIL import Image

    h, w = size
    scene = pyrender.Scene(ambient_light=np.ones(3), bg_color=(*bg, 1.0))
    _add_colored(pyrender, scene, body_verts, body_faces, body_colors01)
    tm = trimesh.Trimesh(vertices=garment_verts, faces=garment_asset.faces, process=False)
    tm.visual = trimesh.visual.TextureVisuals(
        uv=garment_asset.uv, image=Image.fromarray(garment_asset.texture)
    )
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
    scene.add(_intrinsics_camera(camera, pyrender), pose=_cv_pose_to_gl(camera))
    r = pyrender.OffscreenRenderer(w, h)
    try:
        color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
    finally:
        r.delete()
    return np.asarray(color[..., :3], dtype=np.uint8)


def render_body_mask(
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    camera: CameraSpec,
    *,
    size: tuple[int, int] = (1024, 768),
) -> np.ndarray:
    """Yalnız gövde silüeti (H,W) bool — kamera doğrulama (reprojection IoU) için."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    h, w = size
    scene = _flat_scene(pyrender)
    _add_colored(pyrender, scene, body_verts, body_faces, np.ones((len(body_verts), 3)))
    scene.add(_intrinsics_camera(camera, pyrender), pose=_cv_pose_to_gl(camera))
    r = pyrender.OffscreenRenderer(w, h)
    try:
        _, depth = r.render(scene, flags=pyrender.RenderFlags.FLAT)
    finally:
        r.delete()
    return depth > 0
