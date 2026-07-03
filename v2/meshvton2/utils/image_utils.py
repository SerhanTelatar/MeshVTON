"""Görüntü I/O ve dönüşüm yardımcıları (v1 src/utils/image_utils.py'den uyarlandı).

v2 sözleşmesi: tensörler (C,H,W) float32; RGB görüntüler [-1,1], maskeler {0,1}.
Boyut her zaman (height, width) sırasıyla verilir ve LANCZOS ile yeniden örneklenir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, size: tuple[int, int] | None = None) -> Image.Image:
    """size = (height, width)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        h, w = size
        img = img.resize((w, h), Image.LANCZOS)
    return img


def save_image(image: Image.Image | torch.Tensor | np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)


def pil_to_tensor(image: Image.Image, size: tuple[int, int] | None = None) -> torch.Tensor:
    """PIL RGB -> (3,H,W) float32 [-1,1]."""
    if size is not None:
        h, w = size
        image = image.resize((w, h), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(C,H,W) veya (1,C,H,W), [-1,1] veya [0,1] -> PIL."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    t = tensor.detach().float().cpu()
    if t.min() < -0.01:  # [-1,1] varsay
        t = (t.clamp(-1, 1) + 1) / 2
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Gri maske dosyası -> (1,H,W) float32 {0,1} (eşik 127, NEAREST resize)."""
    img = Image.open(path).convert("L")
    if size is not None:
        h, w = size
        img = img.resize((w, h), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    return torch.from_numpy((arr > 127).astype(np.float32)).unsqueeze(0)


def to_uint8_rgb(image: torch.Tensor | np.ndarray | Image.Image) -> np.ndarray:
    """Her formatı (H,W,3) uint8 RGB'ye indirger — metrik girişleri bunu kullanır."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, torch.Tensor):
        return np.asarray(tensor_to_pil(image).convert("RGB"))
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = (arr * 255).round().astype(np.uint8)
        elif arr.min() < 0:  # [-1,1]
            arr = ((arr.clip(-1, 1) + 1) / 2 * 255).round().astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr
