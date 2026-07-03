"""HMR2.0 (4D-Humans) -> SMPL-X parametre backend'i.

v1 src/modules/hmr2_adapter.py'den taşındı; KRİTİK FARK: `pred_cam` (s,tx,ty) ve
`bbox` artık ATILMIYOR, döndürülüyor — fotoğrafın gerçek perspektif kamerası
bunlardan türetilir (camera.py). v2'de kamera azimuth tahmini YOKTUR; v1'in
ön/arka bug'ının kökü olan `azim_from_global_orient` zinciri bilinçli olarak
taşınmadı.

Sözleşme (builder.build_conditioning'in beklediği smplx_params):
    betas (10,), body_pose (63,), global_orient (3,), transl (3,),
    pred_cam (3,) [s, tx, ty — crop-normalize weak-perspective], bbox (4,) [x0,y0,x1,y1]

Kurulum (Colab): pip install git+https://github.com/shubham-goel/4D-Humans.git
"""

from __future__ import annotations

import numpy as np

HMR2_CROP = 256          # HMR2 girdi çözünürlüğü
HMR2_FOCAL = 5000.0      # HMR2'nin crop-uzayı sabit odak uzaklığı (256px'e göre)


def _patch_torch_load_weights_only(torch):
    """PyTorch 2.6+ torch.load(weights_only=True) varsayılanı HMR2'nin Lightning
    checkpoint'ini reddeder. Güvenilir checkpoint → weights_only=False'a zorla."""
    if getattr(torch.load, "_wo_patched", False):
        return
    _orig = torch.load

    def _load(*a, **k):
        k["weights_only"] = False
        return _orig(*a, **k)

    _load._wo_patched = True
    torch.load = _load
    try:
        import torch.serialization

        torch.serialization.add_safe_globals([dict, list, set, tuple, bytes])
    except Exception:
        pass


def detect_person_bbox(image_rgb: np.ndarray) -> np.ndarray:
    """Basit kişi bbox'ı: tam kare (VITON-HD tarzı tek kişi, tam boy çekimlerde
    yeterli). Gerekirse Faz 3'te detektörle değiştirilir — imza sabit."""
    h, w = image_rgb.shape[:2]
    return np.array([0, 0, w, h], dtype=np.float32)


def build_hmr2_backend(device: str = "cuda"):
    """HMR2.0 yükler; `regress(image_rgb, bbox=None) -> smplx_params dict` döndürür."""
    import cv2
    import torch

    _patch_torch_load_weights_only(torch)
    from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2

    try:
        from hmr2.configs import CACHE_DIR_4DHUMANS

        download_models(CACHE_DIR_4DHUMANS)
    except Exception:
        download_models()

    model, _cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def _aa(R):  # (3,3) rotasyon -> (3,) axis-angle
        a, _ = cv2.Rodrigues(np.asarray(R, np.float64))
        return a.reshape(3).astype(np.float32)

    @torch.no_grad()
    def regress(image_rgb: np.ndarray, bbox: np.ndarray | None = None) -> dict:
        if bbox is None:
            bbox = detect_person_bbox(image_rgb)
        x0, y0, x1, y1 = (int(v) for v in bbox)
        crop = image_rgb[y0:y1, x0:x1]
        img = cv2.resize(crop, (HMR2_CROP, HMR2_CROP)).astype(np.float32) / 255.0
        img = (img - mean) / std
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model({"img": t})
        sp = out["pred_smpl_params"]
        go = sp["global_orient"][0].cpu().numpy().reshape(3, 3)
        bp = sp["body_pose"][0].cpu().numpy().reshape(-1, 3, 3)  # (23,3,3)
        betas = sp["betas"][0].cpu().numpy().reshape(-1)[:10].astype(np.float32)
        body_pose = np.concatenate([_aa(bp[i]) for i in range(21)]).astype(np.float32)
        # v1'de atılan alan — v2 kamerasının kilit taşı:
        pred_cam = out["pred_cam"][0].cpu().numpy().reshape(3).astype(np.float32)
        return {
            "betas": betas,
            "body_pose": body_pose,
            "global_orient": _aa(go),
            "transl": np.zeros(3, np.float32),
            "pred_cam": pred_cam,
            "bbox": np.asarray(bbox, dtype=np.float32),
        }

    return regress


# --------------------------------------------------------------------------- #
# SMPL-X gövde modeli (builder'ın vertex kaynağı)
# --------------------------------------------------------------------------- #

_BODY_MODEL = None


def get_body_model(model_dir: str | None = None, device: str = "cpu") -> "SMPLXBody":
    """Süreç-başına tek SMPLXBody (model yüklemesi pahalı). Yol önceliği:
    açık argüman > SMPLX_MODEL_DIR env > v1 konumu checkpoints/pretrained/smplx."""
    global _BODY_MODEL
    if _BODY_MODEL is None:
        import os

        path = model_dir or os.environ.get("SMPLX_MODEL_DIR", "checkpoints/pretrained/smplx")
        _BODY_MODEL = SMPLXBody(path, device=device)
    return _BODY_MODEL


class SMPLXBody:
    """smplx paketinin ince sarmalayıcısı — builder sözleşmesindeki parametrelerden
    vertex/pelvis üretir. Kamera-hizalı çerçevede çalışır (global_orient HMR2'den)."""

    def __init__(self, model_dir: str, gender: str = "neutral", device: str = "cpu"):
        import smplx
        import torch

        self.torch = torch
        self.device = device
        self.model = smplx.create(
            model_dir, model_type="smplx", gender=gender, use_pca=False, batch_size=1
        ).to(device)
        self.faces = np.asarray(self.model.faces, dtype=np.int64)

    def _t(self, a, n):
        t = self.torch.from_numpy(np.asarray(a, np.float32).reshape(1, n)).to(self.device)
        return t

    def __call__(self, smplx_params: dict) -> dict:
        """-> {verts (V,3) float64, faces (F,3), pelvis (3,)} — kamera çerçevesinde."""
        with self.torch.no_grad():
            out = self.model(
                betas=self._t(smplx_params["betas"], 10),
                body_pose=self._t(smplx_params["body_pose"], 63),
                global_orient=self._t(smplx_params["global_orient"], 3),
                transl=self._t(smplx_params.get("transl", np.zeros(3)), 3),
                return_verts=True,
            )
        verts = out.vertices[0].cpu().numpy().astype(np.float64)
        pelvis = out.joints[0, 0].cpu().numpy().astype(np.float64)
        return {"verts": verts, "faces": self.faces, "pelvis": pelvis}

    def rest(self) -> dict:
        """Varsayılan rest gövde (β=0, θ=0) — giysi bağlama (binding) referansı.
        Cache per-garment evrensel kalsın diye kişiye özgü β KULLANILMAZ."""
        zeros = {
            "betas": np.zeros(10), "body_pose": np.zeros(63),
            "global_orient": np.zeros(3), "transl": np.zeros(3),
        }
        return self(zeros)
