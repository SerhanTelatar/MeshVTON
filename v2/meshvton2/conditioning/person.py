"""Person preprocessing: parsing + pose -> agnostic image + inpaint mask.

The Phase 1 backend = the IDM-VTON repo's preprocessing modules (humanparsing ONNX +
openpose + get_mask_location). Rationale:
- Known-good mask logic, validated for months on v1 Colab.
- The heavy parts v2 DOES NOT NEED (densepose/detectron2 source build) are never
  installed — only onnxruntime + lightweight torch modules.
- v1's own AgnosticMaskGenerator relied on an UNTRAINED placeholder segmentation;
  it cannot be trusted (plan: "a real backend is mandatory").

This is NOT a diffusion-backbone dependency (it is about the person, not the model).
It can be fully vendored in Phase 2+ if desired.

Setup (Colab):
  git clone https://github.com/yisol/IDM-VTON /content/IDM-VTON
  + the humanparsing onnx / openpose ckpt files (the notebook setup cell downloads them)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# IDM-VTON preprocessing runs at 384x512; the outputs are scaled to the target size.
_PREP_SIZE = (384, 512)  # (W,H)
GRAY = 128


@dataclass
class PersonPrep:
    image: np.ndarray      # (H,W,3) uint8 RGB — the person at the target size
    agnostic: np.ndarray   # (H,W,3) uint8 RGB — grey inside the mask
    mask: np.ndarray       # (H,W) uint8 {0,255} — 255 = the inpaint (garment) region
    parse: np.ndarray      # (Hp,Wp) uint8 — raw parsing labels (384x512)
    keypoints: dict        # openpose output ("pose_keypoints_2d")


def apply_agnostic(image: np.ndarray, mask: np.ndarray, fill: int = GRAY) -> np.ndarray:
    """Fill the inside of the mask with neutral grey — pure numpy, testable."""
    out = image.copy()
    out[mask > 127] = (fill, fill, fill)
    return out


def person_square_bbox(prep: PersonPrep, pad: float = 0.10) -> np.ndarray:
    """Person-centred SQUARE bbox from the parse (corners at the target resolution: x0,y0,x1,y1).

    detect_person_bbox's full-frame default assumes 'the person is centred and fills the
    frame'; for an off-centre/small person it zeroes weak_persp_to_full's offset terms and
    shifts the body to the image centre. This bbox is derived from the real person region;
    being square, it preserves HMR2's crop contract (overflow is resolved with zero padding
    in regress)."""
    ys, xs = np.nonzero(prep.parse > 0)
    if len(xs) == 0:
        raise ValueError("No person region in the parse — the bbox could not be derived")
    h, w = prep.image.shape[:2]
    ph, pw = prep.parse.shape[:2]
    sx, sy = w / pw, h / ph
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    side = max(x1 - x0, y1 - y0) * (1.0 + pad)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return np.array(
        [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32
    )


class PersonPreprocessor:
    """IDM-VTON preprocess wrapper. The heavy models are loaded on the first process() call."""

    def __init__(self, idm_repo: str | Path, device_index: int = 0, category: str = "upper_body"):
        self.idm_repo = Path(idm_repo)
        if not self.idm_repo.exists():
            raise FileNotFoundError(
                f"no IDM-VTON repo: {self.idm_repo} — the notebook setup cell must clone it."
            )
        self.device_index = device_index
        self.category = category
        self._parsing = None
        self._openpose = None
        self._get_mask_location = None

    def _load(self):
        if self._parsing is not None:
            return
        repo = str(self.idm_repo)
        for p in (repo, str(self.idm_repo / "gradio_demo")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose

        try:
            from utils_mask import get_mask_location
        except ImportError:
            from gradio_demo.utils_mask import get_mask_location

        self._parsing = Parsing(self.device_index)
        self._openpose = OpenPose(self.device_index)
        self._get_mask_location = get_mask_location

    def process(self, image: np.ndarray | str | Path, size: tuple[int, int] = (1024, 768)) -> PersonPrep:
        """
        Args:
            image: (H,W,3) uint8 RGB or a file path.
            size: target (height, width).
        """
        self._load()
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = Image.fromarray(image).convert("RGB")

        h, w = size
        pil = pil.resize((w, h), Image.LANCZOS)
        small = pil.resize(_PREP_SIZE, Image.LANCZOS)

        keypoints = self._openpose(small)
        model_parse, _ = self._parsing(small)
        mask_pil, _ = self._get_mask_location("hd", self.category, model_parse, keypoints)
        mask = np.asarray(mask_pil.resize((w, h), Image.NEAREST))
        mask = ((mask > 127) * 255).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.sum() == 0:
            raise RuntimeError("Empty inpaint mask — parsing/pose failed (person not detected?)")

        img = np.asarray(pil)
        return PersonPrep(
            image=img,
            agnostic=apply_agnostic(img, mask),
            mask=mask,
            parse=np.asarray(model_parse, dtype=np.uint8),
            keypoints=keypoints,
        )
