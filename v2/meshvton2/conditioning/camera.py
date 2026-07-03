"""Kamera modeli: HMR2 weak-perspective -> tam-kare perspektif + orbit görüşleri.

v2'nin kamera felsefesi (v1 ön/arka bug'ının panzehiri):
- Fotoğrafın kamerası HMR2'nin `pred_cam` (s,tx,ty) + bbox'ından TÜRETİLİR;
  azimuth/yön TAHMİNİ yoktur.
- Diğer açılar (90/180/270) giysiyi/gövdeyi döndürerek değil, KAMERAYI pelvis'in
  dikey ekseni etrafında yörüngeleyerek elde edilir (`orbit_camera`).

Koordinat sözleşmesi (HMR2/SPIN ile aynı):
- "Dünya" = kameraya hizalı SMPL çerçevesi: +x sağ, +y AŞAĞI, +z kameradan içeri.
  (HMR2 global_orient'i bu çerçevede tahmin eder; bu yüzden foto kamerasında R=I.)
- Piksel projeksiyonu: u = fx·X/Z + cx, v = fy·Y/Z + cy (OpenCV pinhole).
- Gövdenin "yukarısı" bu çerçevede -y'dir; orbit ekseni budur.

Dönüşüm (4D-Humans `cam_crop_to_full` birebir):
    b  = max(bbox_w, bbox_h)          # HMR2 kare crop kenarı
    f  = 5000/256 · max(W,H)          # crop odağının tam kareye ölçeklenmesi
    tz = 2f / (b·s)
    tx = 2(cx_box − W/2)/(b·s) + pred_cam.tx
    ty = 2(cy_box − H/2)/(b·s) + pred_cam.ty
"""

from __future__ import annotations

import numpy as np

from meshvton2.conditioning.builder import CameraSpec
from meshvton2.conditioning.body import HMR2_CROP, HMR2_FOCAL

UP_AXIS = np.array([0.0, -1.0, 0.0])  # kameraya hizalı SMPL çerçevesinde yukarı


def _tt(a: np.ndarray) -> tuple:
    """np array -> CameraSpec'in iç içe tuple formatı."""
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
    """(s,tx,ty) + bbox -> tam-kare kamera çevirisi (tx,ty,tz) ve odak uzaklığı."""
    h, w = image_size
    s, tx_c, ty_c = (float(v) for v in pred_cam)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    b = max(x1 - x0, y1 - y0)
    if s <= 0 or b <= 0:
        raise ValueError(f"Geçersiz pred_cam/bbox: s={s}, b={b}")
    focal = HMR2_FOCAL / HMR2_CROP * max(w, h)
    bs = b * s
    tz = 2.0 * focal / bs
    cx_box, cy_box = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    tx = 2.0 * (cx_box - w / 2.0) / bs + tx_c
    ty = 2.0 * (cy_box - h / 2.0) / bs + ty_c
    return np.array([tx, ty, tz], dtype=np.float64), float(focal)


def photo_camera(smplx_params: dict, image_size: tuple[int, int]) -> CameraSpec:
    """Fotoğrafın gerçek perspektif kamerası. R = I (HMR2 global_orient kamera
    çerçevesinde olduğundan), T = weak-persp'ten türetilen çeviri."""
    h, w = image_size
    transl, focal = weak_persp_to_full(smplx_params["pred_cam"], smplx_params["bbox"], image_size)
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]])
    return CameraSpec(K=_tt(K), R=_tt(np.eye(3)), T=_tt(transl), source="photo")


def _axis_rot(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues: eksen etrafında deg derecelik rotasyon matrisi."""
    a = np.deg2rad(deg)
    u = axis / np.linalg.norm(axis)
    Kx = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * (Kx @ Kx)


def orbit_camera(base: CameraSpec, pivot: np.ndarray, azim_deg: float) -> CameraSpec:
    """Kamerayı pivot (pelvis) noktasının dikey ekseni etrafında azim_deg döndürür.

    Dünya noktalarına M(x) = Rot·(x − p) + p uygulamakla eşdeğer world2cam
    kompozisyonu: R' = R·Rot, T' = R·(p − Rot·p) + T. Intrinsics değişmez.
    azim=0 -> base'in aynısı; 180 -> arkadan bakış.
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
    """Kamera merkezi dünya koordinatında: C = -Rᵀ·T."""
    _, R, T = spec_arrays(spec)
    return -R.T @ T


def project(spec: CameraSpec, points: np.ndarray) -> np.ndarray:
    """(N,3) dünya noktası -> (N,2) piksel. Z<=0 noktalar NaN döner (kamera arkası)."""
    K, R, T = spec_arrays(spec)
    pc = points @ R.T + T
    z = pc[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc[:, :2] / np.where(z > 1e-8, z, np.nan)) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
    return uv
