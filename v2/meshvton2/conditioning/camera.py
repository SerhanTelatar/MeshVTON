"""Camera model: HMR2 weak-perspective -> full-frame perspective + orbit views.

v2's camera philosophy (the antidote to v1's front/back bug):
- The photo's camera is DERIVED from HMR2's `pred_cam` (s,tx,ty) + bbox;
  there is no azimuth/orientation ESTIMATION.
- The other angles (90/180/270) are obtained not by rotating the garment/body but by
  orbiting the CAMERA around the pelvis's vertical axis (`orbit_camera`).

Coordinate convention (same as HMR2/SPIN):
- "World" = the camera-aligned SMPL frame: +x right, +y DOWN, +z away from the camera.
  (HMR2 predicts global_orient in this frame; that is why R=I for the photo camera.)
- Pixel projection: u = fx·X/Z + cx, v = fy·Y/Z + cy (OpenCV pinhole).
- The body's "up" is -y in this frame; that is the orbit axis.

Conversion (identical to 4D-Humans `cam_crop_to_full`):
    b  = max(bbox_w, bbox_h)          # HMR2 square crop side
    f  = 5000/256 · max(W,H)          # scaling the crop focal to the full frame
    tz = 2f / (b·s)
    tx = 2(cx_box − W/2)/(b·s) + pred_cam.tx
    ty = 2(cy_box − H/2)/(b·s) + pred_cam.ty
"""

from __future__ import annotations

import numpy as np

from meshvton2.conditioning.builder import CameraSpec
from meshvton2.conditioning.body import HMR2_CROP, HMR2_FOCAL

UP_AXIS = np.array([0.0, -1.0, 0.0])  # up in the camera-aligned SMPL frame


def _tt(a: np.ndarray) -> tuple:
    """np array -> CameraSpec's nested tuple format."""
    a = np.asarray(a, dtype=float)
    return tuple(a.tolist()) if a.ndim == 1 else tuple(tuple(r) for r in a.tolist())


def spec_arrays(spec: CameraSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(spec.K, dtype=np.float64),
        np.asarray(spec.R, dtype=np.float64),
        np.asarray(spec.T, dtype=np.float64),
    )


def weak_persp_to_full(
    pred_cam: np.ndarray, bbox: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """(s,tx,ty) + bbox -> full-frame camera translation (tx,ty,tz) and focal length."""
    h, w = image_size
    s, tx_c, ty_c = (float(v) for v in pred_cam)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    b = max(x1 - x0, y1 - y0)
    if s <= 0 or b <= 0:
        raise ValueError(f"Invalid pred_cam/bbox: s={s}, b={b}")
    focal = HMR2_FOCAL / HMR2_CROP * max(w, h)
    bs = b * s
    tz = 2.0 * focal / bs
    cx_box, cy_box = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    tx = 2.0 * (cx_box - w / 2.0) / bs + tx_c
    ty = 2.0 * (cy_box - h / 2.0) / bs + ty_c
    return np.array([tx, ty, tz], dtype=np.float64), float(focal)


def photo_camera(smplx_params: dict, image_size: tuple[int, int]) -> CameraSpec:
    """The photo's real perspective camera. R = I (since HMR2's global_orient is already in
    the camera frame), T = the translation derived from weak-persp."""
    h, w = image_size
    transl, focal = weak_persp_to_full(smplx_params["pred_cam"], smplx_params["bbox"], image_size)
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]])
    return CameraSpec(K=_tt(K), R=_tt(np.eye(3)), T=_tt(transl), source="photo")


def _axis_rot(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues: rotation matrix of deg degrees around an axis."""
    a = np.deg2rad(deg)
    u = axis / np.linalg.norm(axis)
    Kx = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * (Kx @ Kx)


def orbit_camera(base: CameraSpec, pivot: np.ndarray, azim_deg: float) -> CameraSpec:
    """Rotates the camera by azim_deg around the vertical axis of the pivot (pelvis).

    A world2cam composition equivalent to applying M(x) = Rot·(x − p) + p to world points:
    R' = R·Rot, T' = R·(p − Rot·p) + T. The intrinsics do not change.
    azim=0 -> identical to base; 180 -> the view from behind.
    """
    K, R, T = spec_arrays(base)
    p = np.asarray(pivot, dtype=np.float64).reshape(3)
    rot = _axis_rot(UP_AXIS, azim_deg)
    R2 = R @ rot
    T2 = R @ (p - rot @ p) + T
    src = base.source.replace("photo", f"orbit:{int(azim_deg) % 360}")
    if "orbit" not in src:
        src = f"orbit:{int(azim_deg) % 360}"
    return CameraSpec(K=base.K, R=_tt(R2), T=_tt(T2), source=src)


def camera_center(spec: CameraSpec) -> np.ndarray:
    """Camera centre in world coordinates: C = -Rᵀ·T."""
    _, R, T = spec_arrays(spec)
    return -R.T @ T


def project(spec: CameraSpec, points: np.ndarray) -> np.ndarray:
    """(N,3) world point -> (N,2) pixel. Points with Z<=0 return NaN (behind the camera)."""
    K, R, T = spec_arrays(spec)
    pc = points @ R.T + T
    z = pc[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc[:, :2] / np.where(z > 1e-8, z, np.nan)) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
    return uv
