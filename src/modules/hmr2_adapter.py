"""
HMR2.0 (4D-Humans) → SMPL-X parameter adapter.

Provides a `regress_fn(image_rgb, bbox) -> dict` callable conforming to the
contract expected by `RealSMPLXEstimator`:
    betas (10,), body_pose (63,), global_orient (3,), transl (3,)   [numpy]

This is the *real* image→pose backend so global_orient reflects the person's
true facing direction (front/back/side). HMR2 predicts SMPL (not SMPL-X), but
the body pose / global orientation transfer directly (SMPL-X body uses the same
first 21 body joints), which is all the conditioning needs.

Install on Colab: `pip install git+https://github.com/shubham-goel/4D-Humans.git`
Used by both the inference notebook and `extract_smplx.py` so training data and
inference share the same pose source.
"""

import numpy as np


def build_hmr2_regressor(device: str = "cuda", model_path: str = None):
    """Load HMR2.0 and return a `regress_fn(image_rgb, bbox) -> param dict`."""
    import cv2
    import torch
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

    model, _cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def _aa(R):  # rotation matrix (3,3) -> axis-angle (3,)
        a, _ = cv2.Rodrigues(np.asarray(R, np.float64))
        return a.reshape(3).astype(np.float32)

    @torch.no_grad()
    def regress(image_rgb: np.ndarray, bbox) -> dict:
        x0, y0, x1, y1 = bbox
        crop = image_rgb[y0:y1, x0:x1]
        img = cv2.resize(crop, (256, 256)).astype(np.float32) / 255.0
        img = (img - mean) / std
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model({"img": t})
        sp = out["pred_smpl_params"]
        go = sp["global_orient"][0].cpu().numpy().reshape(3, 3)
        bp = sp["body_pose"][0].cpu().numpy().reshape(-1, 3, 3)  # (23,3,3)
        betas = sp["betas"][0].cpu().numpy().reshape(-1)[:10].astype(np.float32)
        body_pose = np.concatenate([_aa(bp[i]) for i in range(21)]).astype(np.float32)  # 63
        return {
            "betas": betas,
            "body_pose": body_pose,
            "global_orient": _aa(go),
            "transl": np.zeros(3, np.float32),
        }

    return regress
