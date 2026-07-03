"""Kişi ön-işleme: parsing + pose -> agnostic görüntü + inpaint maskesi.

Faz 1 backend'i = IDM-VTON reposunun ön-işleme modülleri (humanparsing ONNX +
openpose + get_mask_location). Gerekçe:
- v1 Colab'da aylarca doğrulanmış, bilinen-iyi maske mantığı.
- v2'nin İHTİYAÇ DUYMADIĞI ağır parçalar (densepose/detectron2 source build) hiç
  kurulmaz — yalnız onnxruntime + hafif torch modülleri.
- v1'in kendi AgnosticMaskGenerator'ı EĞİTİLMEMİŞ placeholder segmentasyona
  dayanıyordu; ona güvenilmez (plan: "gerçek backend şart").

Bu bir diffusion-backbone bağımlılığı DEĞİLDİR (kişi hakkında, model hakkında değil).
Faz 2+'da istenirse tamamen vendor edilebilir.

Kurulum (Colab):
  git clone https://github.com/yisol/IDM-VTON /content/IDM-VTON
  + humanparsing onnx / openpose ckpt dosyaları (notebook setup hücresi indirir)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# IDM-VTON ön-işlemesi 384x512'de çalışır; çıktılar hedef boyuta ölçeklenir.
_PREP_SIZE = (384, 512)  # (W,H)
GRAY = 128


@dataclass
class PersonPrep:
    image: np.ndarray      # (H,W,3) uint8 RGB — hedef boyutta kişi
    agnostic: np.ndarray   # (H,W,3) uint8 RGB — maske içi gri
    mask: np.ndarray       # (H,W) uint8 {0,255} — 255 = inpaint (giysi) bölgesi
    parse: np.ndarray      # (Hp,Wp) uint8 — ham parsing etiketleri (384x512)
    keypoints: dict        # openpose çıktısı ("pose_keypoints_2d")


def apply_agnostic(image: np.ndarray, mask: np.ndarray, fill: int = GRAY) -> np.ndarray:
    """Maske içini nötr griyle doldur — saf numpy, testlenebilir."""
    out = image.copy()
    out[mask > 127] = (fill, fill, fill)
    return out


class PersonPreprocessor:
    """IDM-VTON preprocess sarmalayıcısı. Ağır modeller ilk process() çağrısında yüklenir."""

    def __init__(self, idm_repo: str | Path, device_index: int = 0, category: str = "upper_body"):
        self.idm_repo = Path(idm_repo)
        if not self.idm_repo.exists():
            raise FileNotFoundError(
                f"IDM-VTON repo yok: {self.idm_repo} — notebook setup hücresi clone etmeli."
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
            image: (H,W,3) uint8 RGB veya dosya yolu.
            size: hedef (height, width).
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
            raise RuntimeError("Boş inpaint maskesi — parsing/pose başarısız (kişi tespit edilemedi?)")

        img = np.asarray(pil)
        return PersonPrep(
            image=img,
            agnostic=apply_agnostic(img, mask),
            mask=mask,
            parse=np.asarray(model_parse, dtype=np.uint8),
            keypoints=keypoints,
        )
