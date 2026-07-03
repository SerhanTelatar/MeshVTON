"""Sentetik gövde/kamera örnekleme.

Sentetik modda HMR2 yoktur; onun rolünü örneklenmiş parametreler oynar:
`fabricate_camera_params` sahte pred_cam+bbox üretir ve builder FOTOĞRAFLA
BİREBİR AYNI weak-persp→perspektif yolundan geçer (parite tasarım gereği).

Poz kaynağı önceliği: poses_file (HMR2'nin VITON-HD üzerindeki tahminlerinden
(N,63) .npy — hedef dağılımla birebir) > A-pose'a hafif gürültü (fallback;
kontrat/duman testleri için yeterli, üretim verisi için poses_file önerilir).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BETA_STD = 1.2
BETA_CLIP = 2.5


def sample_betas(rng: np.random.RandomState) -> np.ndarray:
    return np.clip(rng.randn(10) * BETA_STD, -BETA_CLIP, BETA_CLIP).astype(np.float32)


def _a_pose(rng: np.random.RandomState) -> np.ndarray:
    """Kollar ~55° indirilmiş doğal duruş + hafif gürültü (SMPL-X body_pose 63).
    Omuz eklemleri: sol=16, sağ=17 (0-indeksli body joint; axis-angle z-bileşeni)."""
    pose = np.zeros(63, np.float32)
    pose[16 * 3 + 2] = -0.95  # sol kol aşağı
    pose[17 * 3 + 2] = 0.95   # sağ kol aşağı
    pose += rng.randn(63).astype(np.float32) * 0.03
    return pose


def load_pose_bank(poses_file: str | Path | None) -> np.ndarray | None:
    if poses_file is None:
        return None
    arr = np.load(poses_file)
    if arr.ndim != 2 or arr.shape[1] != 63:
        raise ValueError(f"poses_file (N,63) olmalı, gelen {arr.shape}")
    return arr.astype(np.float32)


def sample_pose(rng: np.random.RandomState, pose_bank: np.ndarray | None) -> np.ndarray:
    if pose_bank is not None:
        return pose_bank[rng.randint(len(pose_bank))].copy()
    return _a_pose(rng)


def fabricate_camera_params(rng: np.random.RandomState, size: tuple[int, int]) -> dict:
    """Sahte pred_cam+bbox: VITON-HD benzeri tam boy kadraj + hafif jitter.
    global_orient=0 → kişi kameraya dönük (arka görünüm ORBIT ile alınır, gövde
    döndürülerek değil — v2 kuralı)."""
    h, w = size
    return {
        "pred_cam": np.array([rng.uniform(0.85, 1.05), 0.0, rng.uniform(-0.02, 0.08)], np.float32),
        "bbox": np.array([0, 0, w, h], np.float32),
        "global_orient": np.zeros(3, np.float32),
        "transl": np.zeros(3, np.float32),
    }


def sample_identity(rng: np.random.RandomState, size: tuple[int, int], pose_bank=None) -> dict:
    """builder sözleşmesine hazır tam smplx_params seti."""
    p = fabricate_camera_params(rng, size)
    p["betas"] = sample_betas(rng)
    p["body_pose"] = sample_pose(rng, pose_bank)
    return p
