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

# Fallback beden çeşitliliği. DAR tutulur: hedef alan VITON-HD (manken fotoğrafları),
# uç bedenler orada YOK. Eskiden 1.2/2.5 idi ve QA'da aşırı iri gövdeler üretti —
# CLOTH3D giysisi (ortalama SMPL bedeni için modellenmiş) onları örtemeyip yırtıldı,
# omuz/göğüste gövde kumaşın içinden fırladı (2026-08-09 QA görselleri).
BETA_STD = 0.6
BETA_CLIP = 1.6

# SMPL body_pose (63 = 21 eklem × 3) PELVİSİ İÇERMEZ: body_pose[i] ↔ tam iskelet
# eklemi i+1. Dolayısıyla L_shoulder(16)→15, R_shoulder(17)→16. Eskiden 16/17
# yazılıydı; bu R_shoulder + L_ELBOW demekti (sol omuz T-pozda kalıyordu).
_L_SHOULDER, _R_SHOULDER = 15, 16


def sample_betas(rng: np.random.RandomState) -> np.ndarray:
    return np.clip(rng.randn(10) * BETA_STD, -BETA_CLIP, BETA_CLIP).astype(np.float32)


def _a_pose(rng: np.random.RandomState) -> np.ndarray:
    """Kollar ~55° indirilmiş doğal duruş + hafif gürültü (SMPL-X body_pose 63)."""
    pose = np.zeros(63, np.float32)
    pose[_L_SHOULDER * 3 + 2] = -0.95
    pose[_R_SHOULDER * 3 + 2] = 0.95
    pose += rng.randn(63).astype(np.float32) * 0.03
    return pose


def load_identity_bank(poses_file: str | Path | None) -> dict | None:
    """Gerçek kişilerden KİMLİK bankası: body_pose (N,63) + varsa betas (N,10).

    Poz ve beden AYNI kişiden EŞLEŞMİŞ çift olarak alınır — sentetik gövde dağılımı
    böylece çıkarım dağılımıyla (VITON-HD) birebir örtüşür. Eskiden yalnız poz
    alınıp beden rastgele örnekleniyordu; sonuç, gerçek hayatta hiç görülmeyen
    uç bedenlerde yırtılan giysilerdi.

    Kaynak: (N,63) .npy VEYA body_pose içeren .npz klasörü (v1 extract_smplx çıktısı).
    """
    if poses_file is None:
        return None
    p = Path(poses_file)
    if p.is_dir():
        poses, betas = [], []
        for f in sorted(p.glob("*.npz")):
            d = np.load(f)
            if "body_pose" not in d or d["body_pose"].reshape(-1).shape[0] != 63:
                continue
            poses.append(d["body_pose"].reshape(63))
            b = d["betas"].reshape(-1) if "betas" in d else None
            betas.append(b[:10] if b is not None and b.shape[0] >= 10 else None)
        if not poses:
            raise ValueError(f"{p} içinde body_pose'lu .npz yok")
        bank = {"body_pose": np.stack(poses).astype(np.float32), "betas": None}
        if all(b is not None for b in betas):
            bank["betas"] = np.stack(betas).astype(np.float32)
        return bank
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != 63:
        raise ValueError(f"poses_file (N,63) olmalı, gelen {arr.shape}")
    return {"body_pose": arr.astype(np.float32), "betas": None}


def fabricate_camera_params(rng: np.random.RandomState, size: tuple[int, int]) -> dict:
    """Sahte pred_cam+bbox: VITON-HD benzeri tam boy kadraj + hafif jitter.

    global_orient = [π,0,0]: SMPL-X model uzayı y-YUKARI, kamera sözleşmemiz
    y-AŞAĞI — gerçek fotoğraflarda bu dönüşü HMR2'nin global_orient'i taşır,
    sentetikte biz taşırız (0 bırakınca gövde görüntüde BAŞ AŞAĞI çıkıyordu,
    QA'da yakalandı). Kişi kameraya dönüktür; arka görünüm ORBIT ile alınır."""
    h, w = size
    return {
        "pred_cam": np.array([rng.uniform(0.85, 1.05), 0.0, rng.uniform(-0.02, 0.08)], np.float32),
        "bbox": np.array([0, 0, w, h], np.float32),
        "global_orient": np.array([np.pi, 0.0, 0.0], np.float32),
        "transl": np.zeros(3, np.float32),
    }


def sample_identity(rng: np.random.RandomState, size: tuple[int, int], bank: dict | None = None) -> dict:
    """builder sözleşmesine hazır tam smplx_params seti.

    Banka betas içeriyorsa poz+beden AYNI indeksten (aynı gerçek kişiden) alınır —
    çiftin tutarlılığı önemlidir: zayıf bir gövdeye şişman bir poz takmak, ya da
    tersi, gerçek dağılımda olmayan bir kimlik üretir.
    """
    p = fabricate_camera_params(rng, size)
    if bank is None:
        p["body_pose"], p["betas"] = _a_pose(rng), sample_betas(rng)
        return p
    i = rng.randint(len(bank["body_pose"]))
    p["body_pose"] = bank["body_pose"][i].copy()
    p["betas"] = bank["betas"][i].copy() if bank["betas"] is not None else sample_betas(rng)
    return p
