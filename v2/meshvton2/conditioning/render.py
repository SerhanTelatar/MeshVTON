"""Render helpers (pyrender first — NO compilation on Colab).

- Phase 1: appearance reference (`render_appearance_ref`): a SHADED grey cloth render.
  It used to be FLAT (unshaded); flat grey texture + FLAT = a formless silhouette blob, no
  information reached the model (2026-08-09 diagnosis) — now `add_studio_lights` + a matte PBR material.
- Phase 2: screen-space geometry passes (`render_geometry`): camera-space normal +
  depth + garment silhouette, exactly matching CameraSpec's K/R/T (IntrinsicsCamera).
  The normal pass is solved in pyrender with the "make normals vertex colors + FLAT render"
  trick; pytorch3d is not required.

Coordinate note: CameraSpec follows the OpenCV convention (+z forward, +y down); pyrender
wants OpenGL (-z forward, +y up) — the conversion is `_cv_pose_to_gl`.

3D imports are lazy: the module stays importable on a CPU-only dev machine and a render
call raises a clear error.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from meshvton2.conditioning.builder import CameraSpec, GarmentAsset


def zup_to_yup(verts: np.ndarray) -> np.ndarray:
    """CLOTH3D Z-up -> PyTorch3D/camera Y-up: (x,y,z) -> (x,z,-y). (the v1 _geometric_align rule)"""
    return np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=1)


def center_unit(verts: np.ndarray) -> np.ndarray:
    """Centre, then scale the largest axis radius to 1 (for camera framing)."""
    v = verts - verts.mean(axis=0, keepdims=True)
    r = np.abs(v).max()
    return v / max(r, 1e-8)


def force_textureless(asset: GarmentAsset) -> GarmentAsset:
    """PERMANENT RULE: a real texture never reaches the appearance ref — only the garment's
    shape (a flat grey cloth render), never color/pattern. Applied whether or not
    `asset.texture` exists (a single shared path across training/eval/inference)."""
    from dataclasses import replace

    return replace(
        asset,
        texture=np.full((8, 8, 3), 200, np.uint8),
        uv=asset.uv if asset.uv is not None else np.zeros((len(asset.verts), 2), np.float32),
    )


def _dir_light_pose(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """A pyrender DirectionalLight illuminates along its own -z axis; aim it with yaw/pitch."""
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    pose = np.eye(4)
    pose[:3, :3] = ry @ rx
    return pose


def _matte(mesh):
    """pyrender's default PBR material has metallicFactor=1.0 — under directional light
    the diffuse component vanishes and cloth looks METALLIC/pitch black. A matte surface is
    mandatory for cloth and skin (noticed while switching to lighting, 2026-08-09)."""
    for prim in mesh.primitives:
        mat = prim.material
        if hasattr(mat, "metallicFactor"):
            mat.metallicFactor = 0.0
            mat.roughnessFactor = 0.9
    return mesh


def add_studio_lights(pyrender, scene, cam_pose: np.ndarray) -> None:
    """Camera-attached 3-point lighting — it produces SHADING, so the grey cloth's FORM is visible.

    WHY (2026-08-09 diagnosis): the appearance ref and the synthetic GT were rendered with
    `RenderFlags.FLAT` (no lighting); flat grey texture + FLAT = a completely flat silhouette
    blob. The garment's folds/sleeve depth/shoulder roundness reached the model NOT AT ALL
    (the silhouette is already in control_depth_sil) → the model made the garment up and the
    output came out ghostly and semi-transparent. The lights are attached to the CAMERA: at the
    0/90/180/270 views the same surface is shaded the same way (multi-view consistency).
    """
    for yaw, pitch, intensity in ((30.0, -20.0, 3.0), (-45.0, 10.0, 1.4), (160.0, -30.0, 0.9)):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        scene.add(light, pose=cam_pose @ _dir_light_pose(yaw, pitch))


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
    """SHADED (form visible) grey cloth front render of the garment.

    backend: "auto" (pytorch3d if available, otherwise pyrender) | "pytorch3d" | "pyrender".
    pyrender pip-installs in seconds (NO compilation on Colab) — Phase 1 uses it;
    pytorch3d is coming for Phase 2's screen-space renders.

    Returns:
        (H,W,3) uint8 RGB, white background.
    """
    if asset.texture is None or asset.uv is None:
        raise ValueError(
            f"{asset.garment_id}: texture+UV are required for the appearance ref "
            "(load_garment_asset guarantees this)."
        )
    # pyrender is FIXED: the pytorch3d path is still ambient-only (unshaded) — if the two
    # backends produced different images, training/inference would be inconsistent. pyrender is already installed on Colab.
    if backend == "auto":
        backend = "pyrender"
    if backend == "pyrender":
        return _render_pyrender(asset, size=size, azim=azim, dist=dist, convert_zup=convert_zup)
    return _render_pytorch3d(asset, size=size, azim=azim, dist=dist, convert_zup=convert_zup, device=device)


def _prep_verts(asset: GarmentAsset, convert_zup: bool, azim: float) -> np.ndarray:
    """Shared preparation: axis conversion, centre+scale, Y-axis rotation for azim
    (rotating the mesh by -azim is equivalent to rotating the camera)."""
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
    scene = pyrender.Scene(ambient_light=np.full(3, 0.45), bg_color=(1.0, 1.0, 1.0, 1.0))
    scene.add(_matte(pyrender.Mesh.from_trimesh(tm, smooth=True)))  # smooth: keep folds from becoming hard facets
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0), aspectRatio=w / h)
    pose = np.eye(4)
    pose[2, 3] = dist
    scene.add(cam, pose=pose)
    add_studio_lights(pyrender, scene, pose)  # NOT FLAT: shading = the garment's form
    renderer = pyrender.OffscreenRenderer(w, h)
    try:
        color, _ = renderer.render(scene)
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
            "pytorch3d is required (Colab: pip install "
            '"git+https://github.com/facebookresearch/pytorch3d.git")'
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = size
    verts_np = _prep_verts(asset, convert_zup, azim)  # azim applied to the mesh

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

    R, T = look_at_view_transform(dist=dist, elev=0.0, azim=0.0)  # azim was applied in _prep_verts
    cameras = FoVPerspectiveCameras(device=device, R=R.to(device), T=T.to(device))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(image_size=(h, w), faces_per_pixel=1),
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=AmbientLights(device=device),  # unshaded: ambient only (v1 PHASE A1)
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        ),
    )
    with torch.no_grad():
        rgba = renderer(meshes)[0]  # (H,W,4) [0,1]
    return (rgba[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Phase 2: screen-space geometry passes (explicit CameraSpec, pyrender)
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
    """Converts vertex normals into camera space; color encoding [0,1] = (n+1)/2.
    Camera-space normals give the same surface the same color regardless of view
    (conditioning consistency) — unlike v1's world-space normals."""
    from meshvton2.conditioning.lbs_drape import vertex_normals

    n_world = vertex_normals(verts, faces)
    R = np.asarray(camera.R, np.float64)
    n_cam = n_world @ R.T
    return ((n_cam + 1.0) / 2.0).clip(0, 1)


def _flat_scene(pyrender, bg=(0.0, 0.0, 0.0)):
    return pyrender.Scene(ambient_light=np.ones(3), bg_color=(*bg, 0.0))


def _add_colored(pyrender, scene, verts, faces, colors01, *, smooth: bool = False):
    """smooth=False (default): geometry passes — vertex color carries DATA (normal/id),
    interpolation must not corrupt it. smooth=True only in the visual (GT) render."""
    import trimesh

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tm.visual = trimesh.visual.ColorVisuals(
        tm, vertex_colors=(np.clip(colors01, 0, 1) * 255).astype(np.uint8)
    )
    scene.add(_matte(pyrender.Mesh.from_trimesh(tm, smooth=smooth)))


def render_geometry(
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    garment_verts: np.ndarray | None,
    garment_faces: np.ndarray | None,
    camera: CameraSpec,
    *,
    size: tuple[int, int] = (1024, 768),
) -> dict:
    """Screen-space passes for the control channels (all with the CameraSpec camera).

    garment_verts=None → body-only mode (real-data training: no mesh; the silhouette is
    supplied by the caller from the parse) — garment_sil is returned as zeros.

    Returns dict:
        normal   (H,W,3) float32 [0,1] — camera-space scene normals (body+garment)
        depth    (H,W)   float32 [0,1] — min-max normalized within the scene mask; background 0
        garment_sil (H,W) bool         — only the garment's visible silhouette (depth tested)
        scene_mask  (H,W) bool         — body∪garment mask
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

    # Pass 1: combined scene — normal colors + depth buffer
    scene = _flat_scene(pyrender)
    _add_colored(pyrender, scene, body_verts, body_faces,
                 _camera_space_normals(body_verts, body_faces, camera))
    if garment_verts is not None:
        _add_colored(pyrender, scene, garment_verts, garment_faces,
                     _camera_space_normals(garment_verts, garment_faces, camera))
    normal_rgb, depth_raw = _render(scene)
    scene_mask = depth_raw > 0

    # Pass 2: garment id color (the depth test correctly separates the body's front from its back)
    if garment_verts is not None:
        scene2 = _flat_scene(pyrender)
        _add_colored(pyrender, scene2, body_verts, body_faces, np.zeros((len(body_verts), 3)))
        _add_colored(pyrender, scene2, garment_verts, garment_faces, np.ones((len(garment_verts), 3)))
        idmap, _ = _render(scene2)
        garment_sil = idmap[..., 0] > 127
    else:
        garment_sil = np.zeros((h, w), dtype=bool)

    depth = np.zeros((h, w), np.float32)
    if scene_mask.any():
        d = depth_raw[scene_mask]
        lo, hi = float(d.min()), float(d.max())
        span = max(hi - lo, 1e-6)
        # near=1, far→0: make the proximity signal positive in the conditioning
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
    bg=(0.82, 0.82, 0.84),  # VITON-HD studio grey
) -> np.ndarray:
    """Synthetic GT: textured garment + colored body, flat light. (H,W,3) uint8."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    from PIL import Image

    h, w = size
    scene = pyrender.Scene(ambient_light=np.full(3, 0.45), bg_color=(*bg, 1.0))
    _add_colored(pyrender, scene, body_verts, body_faces, body_colors01, smooth=True)
    tm = trimesh.Trimesh(vertices=garment_verts, faces=garment_asset.faces, process=False)
    tm.visual = trimesh.visual.TextureVisuals(
        uv=garment_asset.uv, image=Image.fromarray(garment_asset.texture)
    )
    scene.add(_matte(pyrender.Mesh.from_trimesh(tm, smooth=True)))
    cam_pose = _cv_pose_to_gl(camera)
    scene.add(_intrinsics_camera(camera, pyrender), pose=cam_pose)
    # NOT FLAT: if the training TARGET is unshaded flat color, the model learns to produce
    # unshaded flat (= semi-transparent looking) garments too — 2026-08-09 diagnosis.
    add_studio_lights(pyrender, scene, cam_pose)
    r = pyrender.OffscreenRenderer(w, h)
    try:
        color, _ = r.render(scene)
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
    """Body silhouette only (H,W) bool — for camera validation (reprojection IoU)."""
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
