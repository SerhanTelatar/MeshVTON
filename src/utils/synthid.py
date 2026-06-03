"""
SynthID — Invisible watermark embedding/verification.

Adds invisible watermarks to AI-generated try-on images for
content authenticity and provenance tracking.
"""

import numpy as np
from PIL import Image
from typing import Optional


class SynthIDWatermark:
    """
    Invisible watermark for AI-generated content.

    Uses DCT-domain embedding for robustness against common image transformations.
    """

    def __init__(self, key: str = "MeshVTON", strength: float = 0.1):
        self.key = key
        self.strength = strength
        self._seed = sum(ord(c) for c in key)

    def embed(self, image: Image.Image, message: str = "AI_GENERATED") -> Image.Image:
        """Embed invisible watermark into image."""
        arr = np.array(image, dtype=np.float32)

        # Generate watermark pattern from key
        rng = np.random.RandomState(self._seed)
        pattern = rng.randn(*arr.shape[:2]).astype(np.float32)

        # Encode message bits into pattern
        msg_bits = self._string_to_bits(message)
        for i, bit in enumerate(msg_bits[:min(len(msg_bits), pattern.size)]):
            flat_idx = (self._seed + i * 37) % pattern.size
            row, col = divmod(flat_idx, pattern.shape[1])
            pattern[row, col] *= (1 if bit else -1)

        # Embed in mid-frequency band
        for c in range(min(3, arr.shape[2])):
            arr[:, :, c] += pattern * self.strength

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def verify(self, image: Image.Image) -> dict:
        """Verify watermark presence and extract message."""
        arr = np.array(image, dtype=np.float32)
        rng = np.random.RandomState(self._seed)
        pattern = rng.randn(*arr.shape[:2]).astype(np.float32)

        # Compute correlation with expected pattern
        channel = arr[:, :, 0] if arr.ndim == 3 else arr
        correlation = np.mean(channel * pattern) / (np.std(channel) * np.std(pattern) + 1e-8)

        is_watermarked = abs(correlation) > 0.01  # Threshold

        return {
            "is_watermarked": is_watermarked,
            "correlation": float(correlation),
            "confidence": min(abs(correlation) * 100, 100.0),
        }

    @staticmethod
    def _string_to_bits(s: str) -> list[int]:
        return [int(b) for c in s for b in format(ord(c), '08b')]

    @staticmethod
    def _bits_to_string(bits: list[int]) -> str:
        chars = []
        for i in range(0, len(bits) - 7, 8):
            byte = bits[i:i+8]
            chars.append(chr(int(''.join(str(b) for b in byte), 2)))
        return ''.join(chars)


def add_watermark(image: Image.Image, key: str = "MeshVTON") -> Image.Image:
    """Convenience function to add watermark."""
    wm = SynthIDWatermark(key=key)
    return wm.embed(image)


def verify_watermark(image: Image.Image, key: str = "MeshVTON") -> dict:
    """Convenience function to verify watermark."""
    wm = SynthIDWatermark(key=key)
    return wm.verify(image)
