"""
Shared 3D conditioning builder — single source of truth for `conditioning_3d`.

Both training preprocessing (render_garment.py), scripts/inference.py, and the
Colab inference notebook MUST build `conditioning_3d` through this function so the
ControlNet3D sees the *same* distribution at train and inference time.

Pipeline (mirrors the validated training render path):
    person image → SMPL-X estimate (real pose + global_orient)
        → load garment mesh → geometric drape onto body (size match)
        → camera azimuth derived from the body's global_orient (front=0, back=180)
        → render RGB(textured) + normal + depth at that azimuth, 512×384
        → conditioning_3d = cat(rgb, normal, depth)  (1,9,H,W) in [-1,1]

The previous notebook path (manual `_rot`/`_normalize`, fixed angle, 768×1024,
gray, person-independent) is exactly what made the garment ignore the person —
do not reintroduce it.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.modules.garment_draper import load_garment_mesh


def _to_bgr(person_image) -> np.ndarray:
    """Accept a PIL image, RGB ndarray, or BGR ndarray → return BGR ndarray."""
    if isinstance(person_image, Image.Image):
        rgb = np.array(person_image.convert("RGB"))
        return rgb[:, :, ::-1].copy()
    arr = np.asarray(person_image)
    return arr  # assume already BGR if ndarray was passed by a cv2-based caller


def azim_from_global_orient(global_orient: np.ndarray) -> float:
    """Map an SMPL-X global_orient (axis-angle, 3) to a camera azimuth in degrees.

    Only the garment is rendered (centered at origin), so to reveal the side of
    the garment that matches the person we rotate the *camera* by the body's yaw:
    a front-facing person (yaw≈0) → azim 0; a back-facing person (yaw≈π) → azim 180.

    Sign convention may need flipping for a given regressor; callers can override
    via `view_angle`. Uses Rodrigues to get the rotation matrix, then extracts yaw.
    """
    try:
        import cv2
        R, _ = cv2.Rodrigues(np.asarray(global_orient, dtype=np.float64).reshape(3))
        # Yaw about the vertical (y) axis from the rotation matrix.
        yaw = np.arctan2(R[0, 2], R[2, 2])
        return float(np.degrees(yaw)) % 360.0
    except Exception:
        return 0.0


@torch.no_grad()
def build_conditioning_3d(
    person_image,
    mesh_path: str,
    estimator,
    renderer,
    draper,
    height: int = 512,
    width: int = 384,
    view_angle: Optional[float] = None,
    smplx_params: Optional[dict] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Build the 9-channel ControlNet3D conditioning for one (person, garment).

    Args:
        person_image: PIL/RGB/BGR person image (only needed if smplx_params is None).
        mesh_path: path to the 3D garment mesh (.obj/.glb/.ply).
        estimator: SMPLXEstimator-like (estimate()/get_body_mesh()).
        renderer: a set-up MeshRenderer.
        draper: a GarmentDraper.
        height, width: canonical output size (must match training: 512×384).
        view_angle: optional azimuth override (deg). If None, derived from pose.
        smplx_params: precomputed SMPL-X params (training reuses cached .npz).
        device, dtype: output device/dtype.

    Returns dict:
        conditioning_3d: (1, 9, H, W) in [-1,1]  (rgb + normal + depth)
        render_rgb: PIL.Image of the textured garment render (GarmentNet/IP-Adapter input)
        azim: the azimuth used (deg)
        global_orient: the SMPL-X global_orient used
    """
    # 1) Pose: real SMPL-X params (estimate from image or reuse cached params).
    if smplx_params is None:
        params = estimator.estimate(_to_bgr(person_image))
    else:
        params = dict(smplx_params)
    body_mesh = estimator.get_body_mesh(params)

    # 2) Camera azimuth: explicit override, else derived from the body's yaw.
    azim = float(view_angle) if view_angle is not None else azim_from_global_orient(
        params["global_orient"]
    )
    cam = {"dist": 2.7, "elev": 0, "azim": azim}

    # 3) Garment: load, drape (geometric align to body size), keep real colors.
    garment = load_garment_mesh(mesh_path)
    g_verts = torch.tensor(garment["vertices"], dtype=torch.float32, device=device).unsqueeze(0)
    g_faces = torch.tensor(garment["faces"], dtype=torch.long, device=device)
    g_colors = torch.tensor(garment["vertex_colors"], dtype=torch.float32, device=device).unsqueeze(0)
    b_verts = torch.tensor(body_mesh["vertices"], dtype=torch.float32, device=device).unsqueeze(0)
    b_faces = torch.tensor(body_mesh["faces"], dtype=torch.long, device=device)

    drape = draper(g_verts, g_faces, b_verts, b_faces)
    draped = drape["draped_verts"]

    # 4) Render RGB (textured) + normal + depth, all at the SAME azimuth.
    rgb = renderer.render(draped, g_faces, g_colors, cam)            # (1,4,H,W)
    normal = renderer.render_normal_map(draped, g_faces, cam)        # (1,4,H,W)
    depth = renderer.render_depth_map(draped, g_faces, cam)          # (1,4,H,W)

    def _prep(t):  # (1,4,h,w)[0,1] → (1,3,H,W)[-1,1] at canonical size
        t = t[:, :3].to(torch.float32)
        t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
        return (t * 2.0 - 1.0)

    rgb_n, normal_n, depth_n = _prep(rgb), _prep(normal), _prep(depth)
    conditioning_3d = torch.cat([rgb_n, normal_n, depth_n], dim=1).to(device, dtype)  # (1,9,H,W)

    # RGB render as a PIL image for GarmentNet / IP-Adapter (real garment appearance).
    rgb_np = (rgb[0, :3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    render_rgb = Image.fromarray(rgb_np).resize((width, height), Image.LANCZOS)

    return {
        "conditioning_3d": conditioning_3d,
        "render_rgb": render_rgb,
        "azim": azim,
        "global_orient": params["global_orient"],
    }
