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


def _coco18_keypoints(pose_keypoints):
    """OpenPose çıktısını (N,3) [x,y,conf] COCO-18 dizisine normalize et.

    `pose_keypoints` bir dict ('pose_keypoints_2d') veya düz/array olabilir;
    confidence sütunu yoksa hepsi görünür kabul edilir. Başarısızsa None döner.
    """
    if pose_keypoints is None:
        return None
    try:
        kp = pose_keypoints
        if isinstance(kp, dict):
            kp = kp.get("pose_keypoints_2d", kp)
        arr = np.asarray(kp, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3) if arr.size % 3 == 0 else arr.reshape(-1, 2)
        if arr.shape[-1] == 2:  # confidence yok → tümü görünür
            arr = np.concatenate([arr, np.ones((arr.shape[0], 1), np.float32)], axis=1)
        return arr
    except Exception:
        return None


def detect_facing(parse=None, pose_keypoints=None, conf_th: float = 0.1):
    """Kişinin ön mü arka mı baktığını 2D ipuçlarından tahmin et.

    HMR2'nin monoküler global_orient'i ön/arka'yı güvenilmez ayırdığı için
    ([azim_from_global_orient] gürültülü) bu fonksiyon parsing + openpose'tan
    sağlam bir oy verir. Döner: ('front' | 'back' | None, açıklama).

    Sinyaller:
      1) Human parsing: head/face(=11) vs hair(=2) piksel oranı (IDM-VTON label_map).
      2) OpenPose: omuz x-sırası (Lsho - Rsho işareti) + yüz noktası görünürlüğü.
    """
    votes = []
    # 1) Parsing — yüz(head) vs saç
    if parse is not None:
        a = np.asarray(parse)
        if a.ndim == 3:
            a = a[..., 0]
        n = max(a.size, 1)
        face = float((a == 11).sum()) / n
        hair = float((a == 2).sum()) / n
        if face > 0.002:
            votes.append("front")
        elif hair > 0.002 and face < 0.0005:
            votes.append("back")

    # 2) OpenPose — omuz sırası + yüz noktaları (COCO-18:
    #    0 nose,2 Rsho,5 Lsho,14 Reye,15 Leye,16 Rear,17 Lear)
    kp = _coco18_keypoints(pose_keypoints)
    if kp is not None and len(kp) >= 6:
        def ok(i):
            return i < len(kp) and kp[i, 2] > conf_th
        if ok(2) and ok(5):
            dx = kp[5, 0] - kp[2, 0]  # ön: sol omuz daha büyük x'te
            span = float(kp[:, 0].max() - kp[:, 0].min()) + 1e-3
            if abs(dx) > 0.06 * span:
                votes.append("front" if dx > 0 else "back")
        face_pts = sum(ok(i) for i in (0, 14, 15))
        if face_pts >= 2:
            votes.append("front")
        elif face_pts == 0 and (ok(16) or ok(17)):
            votes.append("back")

    if not votes:
        return None, "belirsiz (yan profil olabilir)"
    nf, nb = votes.count("front"), votes.count("back")
    if nf == nb:
        return None, f"çelişkili oylar={votes}"
    return ("front" if nf > nb else "back"), f"oylar={votes}"


def azim_from_facing(facing):
    """Ön/arka etiketini kamera azimuth'una çevir. front→0°, back→180°, diğer→None."""
    return {"front": 0.0, "back": 180.0}.get(facing)


def azim_from_2d(parse=None, pose_keypoints=None, conf_th: float = 0.1):
    """2D ipuçlarından TÜM-AÇI azimuth tahmini (ön/arka/yan). Döner: (azim|None, açıklama).

    Mantık:
      - Omuzlar x'te birbirine YAKINSA (üst üste) → yan profil (90° veya 270°).
        Sol/sağ: burun, boyun/omuz ortasının hangi tarafında → 90/270.
      - Omuzlar AÇIK ise → ön/arka (detect_facing: parse yüz/saç + omuz sırası) → 0/180.
      - Hiçbiri net değilse → None (çağıran HMR2 azim'ine veya VIEW_ANGLE'a düşer).

    Yan profil işareti (90 vs 270) regressor/kameraya göre ters olabilir → gerekirse
    notebook'ta VIEW_ANGLE ile elle ezilir.
    """
    kp = _coco18_keypoints(pose_keypoints)
    if kp is not None and len(kp) >= 6:
        def ok(i):
            return i < len(kp) and kp[i, 2] > conf_th
        if ok(2) and ok(5):
            dx = abs(kp[5, 0] - kp[2, 0])
            span = float(kp[:, 0].max() - kp[:, 0].min()) + 1e-3
            if dx < 0.18 * span:  # omuzlar üst üste → yan profil
                ref_x = kp[1, 0] if ok(1) else (kp[2, 0] + kp[5, 0]) / 2.0
                nose_x = kp[0, 0] if ok(0) else ref_x
                # burun referansın solundaysa sol profil → 270, sağındaysa sağ → 90
                side = "left" if nose_x < ref_x else "right"
                return (270.0 if side == "left" else 90.0), f"profile/{side}"
    facing, why = detect_facing(parse, pose_keypoints, conf_th)
    return azim_from_facing(facing), why


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
    texture_path: Optional[str] = None,
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
    garment = load_garment_mesh(mesh_path, texture_path=texture_path)
    g_verts = torch.tensor(garment["vertices"], dtype=torch.float32, device=device).unsqueeze(0)
    g_faces = torch.tensor(garment["faces"], dtype=torch.long, device=device)
    g_colors = torch.tensor(garment["vertex_colors"], dtype=torch.float32, device=device).unsqueeze(0)
    b_verts = torch.tensor(body_mesh["vertices"], dtype=torch.float32, device=device).unsqueeze(0)
    b_faces = torch.tensor(body_mesh["faces"], dtype=torch.long, device=device)

    drape = draper(g_verts, g_faces, b_verts, b_faces)
    draped = drape["draped_verts"]

    # 4) Render RGB + normal + depth, all at the SAME azimuth.
    # Prefer a REAL UV texture (preserves the garment pattern) when the mesh carries
    # one (.obj + .mtl + texture image); else fall back to baked vertex colors (gray).
    uv = np.asarray(garment.get("uv", np.zeros((0, 2))), dtype=np.float32)
    tex = np.asarray(garment.get("texture", np.zeros((0, 0, 0))))
    n_verts = garment["vertices"].shape[0]
    has_uv_tex = (
        tex.ndim == 3 and min(tex.shape[:2]) >= 8 and float(tex.std()) > 2.0
        and uv.shape == (n_verts, 2) and float(np.abs(uv).sum()) > 0.0
    )
    if has_uv_tex:
        verts_uvs = torch.tensor(uv, dtype=torch.float32, device=device).unsqueeze(0)
        faces_uvs = g_faces.unsqueeze(0)
        tmap = torch.tensor(tex[..., :3].astype(np.float32) / 255.0,
                            dtype=torch.float32, device=device).unsqueeze(0)  # (1,H,W,3)
        rgb = renderer.render_uv(draped, g_faces, verts_uvs, faces_uvs, tmap, cam, flat=True)
    else:
        rgb = renderer.render(draped, g_faces, g_colors, cam)        # (1,4,H,W) gri fallback
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

    # Garment silhouette (mask) from the render: alpha if present, else non-white pixels.
    # Used (optionally) to make the inpaint mask follow the garment's real shape.
    if rgb.shape[1] >= 4:
        sil = (rgb[0, 3] > 0.5).float()
    else:
        sil = (rgb[0, :3].mean(0) < 0.98).float()  # white bg → foreground = garment
    sil = F.interpolate(sil[None, None], size=(height, width), mode="nearest")[0, 0]
    silhouette = Image.fromarray((sil.cpu().numpy() * 255).astype(np.uint8))

    return {
        "conditioning_3d": conditioning_3d,
        "render_rgb": render_rgb,
        "silhouette": silhouette,
        "azim": azim,
        "global_orient": params["global_orient"],
    }
