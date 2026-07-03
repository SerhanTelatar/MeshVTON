"""FLUX try-on sarmalayıcısı — TÜM diffusers temas noktaları bu dosyada.

Faz 1 (zero-shot, eğitimsiz) varyantları:
- "fill_spatial": FluxFillPipeline; canvas = [görünüm referansı | agnostic kişi],
  maske yalnız kişi yarısında → model referansı görerek inpaint eder (CatVTON hilesi).
- "kontext": FluxKontextPipeline; giriş = [kişi | referans] dikişli tek görüntü +
  edit talimatı, çıktı sol yarıdan kırpılır.

Plan notu: üçüncü varyant (Fill + eğitimsiz Kontext-tarzı ref-token sequence-concat)
Faz 4'e ertelendi — eğitimsiz halinin kanıt değeri düşük (Fill ref-token görmeden
eğitildi, OOD) ve özel örnekleme döngüsü zaten Faz 4'te reference_tokens.py ile
geliyor. Karar Faz 1 raporunda (a) vs (c) kanıtıyla verilir.

Maske disiplini: fill_spatial çıktısında maske DIŞI pikseller orijinal kişiyle
kompozit edilir — zero-shot modelin kişi/arka planı bozması metriklere sızmaz.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

VARIANTS = ("fill_spatial", "kontext")

FILL_PROMPT = (
    "the person is wearing the garment shown on the left, "
    "photorealistic fashion photo, natural fit, correct drape"
)
KONTEXT_PROMPT = (
    "Make the person on the left wear the garment shown on the right. "
    "Keep the person's identity, pose, body shape and background unchanged."
)


# ------------------------- saf yardımcılar (lokal testlenebilir) ------------------------- #


def make_side_canvas(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """İki (H,W,3) görüntüyü yatay dikişle (H,2W,3) yapar; boyutlar eşit olmalı."""
    if left.shape != right.shape:
        raise ValueError(f"Boyut uyuşmazlığı: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=1)


def make_side_mask(mask_right: np.ndarray) -> np.ndarray:
    """(H,W) maskeyi (H,2W) canvas maskesine koyar: sol yarı (referans) daima 0."""
    h, w = mask_right.shape[:2]
    out = np.zeros((h, 2 * w), dtype=np.uint8)
    out[:, w:] = mask_right
    return out


def crop_half(canvas: np.ndarray, side: str) -> np.ndarray:
    h, w2 = canvas.shape[:2]
    w = w2 // 2
    return canvas[:, :w] if side == "left" else canvas[:, w:]


def composite_outside_mask(pred: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maske dışını orijinalden geri koy (kenarda 3px yumuşatma)."""
    import cv2

    m = ((mask > 127) * 255).astype(np.uint8)
    m = cv2.GaussianBlur(m, (7, 7), 0).astype(np.float32) / 255.0
    m = m[..., None]
    out = pred.astype(np.float32) * m + original.astype(np.float32) * (1 - m)
    return out.round().astype(np.uint8)


# ------------------------------- pipeline sarmalayıcı ------------------------------- #


class FluxTryOn:
    def __init__(
        self,
        variant: str,
        *,
        fill_repo: str = "black-forest-labs/FLUX.1-Fill-dev",
        kontext_repo: str = "black-forest-labs/FLUX.1-Kontext-dev",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        steps: int = 28,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant {VARIANTS} içinden olmalı, gelen: {variant}")
        self.variant = variant
        self.device, self.dtype, self.steps = device, dtype, steps
        self.repo = fill_repo if variant == "fill_spatial" else kontext_repo
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        if self.variant == "fill_spatial":
            from diffusers import FluxFillPipeline

            self._pipe = FluxFillPipeline.from_pretrained(self.repo, torch_dtype=self.dtype)
        else:
            from diffusers import FluxKontextPipeline

            self._pipe = FluxKontextPipeline.from_pretrained(self.repo, torch_dtype=self.dtype)
        self._pipe.enable_model_cpu_offload() if self.device == "cuda_offload" else self._pipe.to(self.device)

    @torch.no_grad()
    def tryon(
        self,
        person: np.ndarray,       # (H,W,3) uint8 RGB
        agnostic: np.ndarray,     # (H,W,3) uint8 RGB
        mask: np.ndarray,         # (H,W) uint8 {0,255}
        appearance_ref: np.ndarray,  # (H,W,3) uint8 RGB
        *,
        seed: int = 0,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Tek try-on üretimi; (H,W,3) uint8 RGB döner."""
        self._load()
        h, w = person.shape[:2]
        gen = torch.Generator(device="cpu").manual_seed(seed)

        if self.variant == "fill_spatial":
            canvas = make_side_canvas(appearance_ref, agnostic)
            cmask = make_side_mask(mask)
            out = self._pipe(
                prompt=prompt or FILL_PROMPT,
                image=Image.fromarray(canvas),
                mask_image=Image.fromarray(cmask),
                height=h,
                width=2 * w,
                num_inference_steps=self.steps,
                guidance_scale=30.0,  # Fill-dev önerilen yüksek guidance
                generator=gen,
            ).images[0]
            pred = crop_half(np.asarray(out.convert("RGB")), "right")
        else:
            stitched = make_side_canvas(person, appearance_ref)
            out = self._pipe(
                prompt=prompt or KONTEXT_PROMPT,
                image=Image.fromarray(stitched),
                height=h,
                width=2 * w,
                num_inference_steps=self.steps,
                guidance_scale=2.5,
                generator=gen,
            ).images[0]
            arr = np.asarray(out.convert("RGB"))
            if arr.shape[:2] != (h, 2 * w):  # Kontext tercih ettiği çözünürlüğe kayabilir
                arr = np.asarray(Image.fromarray(arr).resize((2 * w, h), Image.LANCZOS))
            pred = crop_half(arr, "left")

        return composite_outside_mask(pred, person, mask)
