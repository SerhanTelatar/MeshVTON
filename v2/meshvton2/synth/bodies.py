"""Synthetic body/camera sampling.

In synthetic mode there is no HMR2; sampled parameters play its role:
`fabricate_camera_params` produces a fake pred_cam+bbox and the builder goes through the
EXACT SAME weak-persp→perspective path as a PHOTO (parity by design).

Pose source priority: poses_file (an (N,63) .npy from HMR2's predictions over VITON-HD —
matching the target distribution exactly) > A-pose with slight noise (fallback; enough for
contract/smoke tests, poses_file is recommended for production data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fallback body diversity. Kept NARROW: the target domain is VITON-HD (model photos),
# extreme bodies do NOT exist there. It used to be 1.2/2.5 and produced oversized bodies in
# QA — the CLOTH3D garment (modelled for an average SMPL body) could not cover them and tore,
# the body poked through the cloth at the shoulder/chest (2026-08-09 QA images).
BETA_STD = 0.6
BETA_CLIP = 1.6

# SMPL body_pose (63 = 21 joints × 3) DOES NOT INCLUDE THE PELVIS: body_pose[i] ↔ full
# skeleton joint i+1. So L_shoulder(16)→15, R_shoulder(17)→16. It used to say 16/17;
# that meant R_shoulder + L_ELBOW (the left shoulder stayed in the T-pose).
_L_SHOULDER, _R_SHOULDER = 15, 16


def sample_betas(rng: np.random.RandomState) -> np.ndarray:
    return np.clip(rng.randn(10) * BETA_STD, -BETA_CLIP, BETA_CLIP).astype(np.float32)


def _a_pose(rng: np.random.RandomState) -> np.ndarray:
    """Natural stance with the arms lowered ~55° + slight noise (SMPL-X body_pose 63)."""
    pose = np.zeros(63, np.float32)
    pose[_L_SHOULDER * 3 + 2] = -0.95
    pose[_R_SHOULDER * 3 + 2] = 0.95
    pose += rng.randn(63).astype(np.float32) * 0.03
    return pose


def load_identity_bank(poses_file: str | Path | None) -> dict | None:
    """IDENTITY bank from real people: body_pose (N,63) + betas (N,10) when available.

    Pose and body are taken as a MATCHED pair from the SAME person — this makes the synthetic
    body distribution overlap exactly with the inference distribution (VITON-HD). It used to
    take only the pose and sample the body at random; the result was garments tearing on
    extreme bodies that never occur in real life.

    Source: an (N,63) .npy OR a folder of .npz files containing body_pose (v1 extract_smplx output).
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
            raise ValueError(f"no .npz with body_pose inside {p}")
        bank = {"body_pose": np.stack(poses).astype(np.float32), "betas": None}
        if all(b is not None for b in betas):
            bank["betas"] = np.stack(betas).astype(np.float32)
        return bank
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != 63:
        raise ValueError(f"poses_file must be (N,63), got {arr.shape}")
    return {"body_pose": arr.astype(np.float32), "betas": None}


def fabricate_camera_params(rng: np.random.RandomState, size: tuple[int, int]) -> dict:
    """Fake pred_cam+bbox: VITON-HD-like full-body framing + slight jitter.

    global_orient = [π,0,0]: the SMPL-X model space is y-UP while our camera convention is
    y-DOWN — on real photos HMR2's global_orient carries this rotation, in synthetic mode we
    carry it (leaving it at 0 made the body come out UPSIDE DOWN in the image, caught in QA).
    The person faces the camera; the back view is obtained with an ORBIT."""
    h, w = size
    return {
        "pred_cam": np.array([rng.uniform(0.85, 1.05), 0.0, rng.uniform(-0.02, 0.08)], np.float32),
        "bbox": np.array([0, 0, w, h], np.float32),
        "global_orient": np.array([np.pi, 0.0, 0.0], np.float32),
        "transl": np.zeros(3, np.float32),
    }


def sample_identity(rng: np.random.RandomState, size: tuple[int, int], bank: dict | None = None) -> dict:
    """A full smplx_params set, ready for the builder contract.

    If the bank contains betas, pose and body are taken from the SAME index (the same real
    person) — the pair's consistency matters: putting a heavy pose on a thin body, or the
    reverse, produces an identity that does not exist in the real distribution.
    """
    p = fabricate_camera_params(rng, size)
    if bank is None:
        p["body_pose"], p["betas"] = _a_pose(rng), sample_betas(rng)
        return p
    i = rng.randint(len(bank["body_pose"]))
    p["body_pose"] = bank["body_pose"][i].copy()
    p["betas"] = bank["betas"][i].copy() if bank["betas"] is not None else sample_betas(rng)
    return p
