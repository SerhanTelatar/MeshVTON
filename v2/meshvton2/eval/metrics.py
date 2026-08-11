"""MeshVTON v2 evaluation metrics.

Design rules (v1 lessons):
- If a metric cannot be computed (missing dependency / missing input) it returns None, not a
  LIE like 0.0; the harness reports None as "n/a". (In v1 compute_lpips returned 0.0 on an
  ImportError — a silent illusion of success.)
- All inputs are (H,W,3) uint8 RGB, masks are (H,W) bool; the conversions live in utils.image_utils.
"""

from __future__ import annotations

import numpy as np

_LPIPS_MODEL = None
_LPIPS_UNAVAILABLE = False


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return m > (127 if m.dtype == np.uint8 else 0.5)


def compute_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """SSIM; if a mask is given, the mean of the SSIM map inside the mask (region SSIM)."""
    from skimage.metrics import structural_similarity

    if mask is None:
        return float(structural_similarity(pred, target, channel_axis=-1, data_range=255))
    _, smap = structural_similarity(pred, target, channel_axis=-1, data_range=255, full=True)
    m = _as_bool_mask(mask)
    if not m.any():
        return None
    return float(smap[m].mean())


def compute_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    if mask is not None:
        m = _as_bool_mask(mask)
        if not m.any():
            return None
        diff = diff[m]
    mse = np.mean(diff**2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def compute_lpips(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """LPIPS (AlexNet). If a mask is given, everything OUTSIDE it is pinned to neutral grey in
    both images → the metric only picks up signal from the region. None if lpips is not installed."""
    global _LPIPS_MODEL, _LPIPS_UNAVAILABLE
    if _LPIPS_UNAVAILABLE:
        return None
    try:
        import lpips
        import torch
    except ImportError:
        _LPIPS_UNAVAILABLE = True
        return None
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).eval()

    def prep(img: np.ndarray) -> "torch.Tensor":
        a = img.astype(np.float32) / 255.0
        if mask is not None:
            m = _as_bool_mask(mask)
            a = a.copy()
            a[~m] = 0.5
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
        return t * 2.0 - 1.0

    import torch

    with torch.no_grad():
        return float(_LPIPS_MODEL(prep(pred), prep(target)).item())


def garment_delta_e(pred: np.ndarray, pred_mask: np.ndarray, ref: np.ndarray, ref_mask: np.ndarray | None = None) -> float | None:
    """Garment color fidelity: CIEDE2000 between pred's mean in-mask Lab color and the
    reference's mean Lab color (the appearance ref, optionally foreground-masked)."""
    from skimage.color import deltaE_ciede2000, rgb2lab

    pm = _as_bool_mask(pred_mask)
    if not pm.any():
        return None
    if ref_mask is not None:
        rm = _as_bool_mask(ref_mask)
    else:
        # Roughly isolate the foreground in the reference: drop the near-pure-white background.
        rm = ~np.all(ref > 245, axis=-1)
    if not rm.any():
        return None
    pred_lab = rgb2lab(pred.astype(np.float64) / 255.0)
    ref_lab = rgb2lab(ref.astype(np.float64) / 255.0)
    mean_pred = pred_lab[pm].mean(axis=0)
    mean_ref = ref_lab[rm].mean(axis=0)
    return float(deltaE_ciede2000(mean_pred[None, None, :], mean_ref[None, None, :])[0, 0])


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    a, b = _as_bool_mask(mask_a), _as_bool_mask(mask_b)
    union = (a | b).sum()
    if union == 0:
        return None
    return float((a & b).sum() / union)


def outside_mask_preservation(pred: np.ndarray, person: np.ndarray, inpaint_mask: np.ndarray) -> dict:
    """Identity preservation: OUTSIDE the inpaint mask, pred must stay identical to the original person."""
    outside = ~_as_bool_mask(inpaint_mask)
    return {
        "outside_mask_ssim": compute_ssim(pred, person, mask=outside),
        "outside_mask_psnr": compute_psnr(pred, person, mask=outside),
        "outside_mask_lpips": compute_lpips(pred, person, mask=outside),
    }
