"""build_conditioning() — koşullama üretiminin TEK kaynağı (parite sözleşmesi).

v1'in en pahalı dersi: eğitim ön-işlemesi ile inference koşullaması farklı kod
yollarından üretilince ControlNet dağıtım-dışı girdi aldı ve sonuçları bozdu.
v2 kuralı: eğitim ön-işleme (scripts/preprocess_vitonhd.py), sentetik üretici
(synth/generate.py, person_image=None modu) ve inference (inference/run_tryon.py)
İSTİSNASIZ bu modüldeki build_conditioning()'i çağırır. tests/test_parity.py
bunu iki yoldan aynı girdiyle çağırıp tensör eşitliğini zorlar.

Bu imza Faz 0'da donmuştur; alan eklemek serbest, mevcut alanı değiştirmek yasak.

FAZ 0 DURUMU: implementasyon deterministik STUB'tır (_IS_STUB=True) — doğru
şekil/aralıkta, girdilere deterministik bağlı sahte tensörler üretir ki parite
testi gerçek implementasyon gelmeden önce de anlamlı çalışsın. Faz 2'de stub
gövdesi gerçek hatla (HMR2 kamera + LBS drape + PyTorch3D render) değiştirilir;
imza ve testler değişmez.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_IS_STUB = True  # Faz 2'de False yapılır; test_parity bunu umursamaz, run_tryon uyarır.

CANONICAL_SIZE = (1024, 768)  # (height, width) — configs/base.yaml ile aynı


# --------------------------------------------------------------------------- #
# Veri tipleri
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraSpec:
    """Serileştirilebilir kamera: tam-kare intrinsics + dünya->kamera dönüşümü."""

    K: tuple  # 3x3, iç içe tuple
    R: tuple  # 3x3
    T: tuple  # 3
    source: str  # "photo" | "orbit:90" | "synth" ...

    def to_dict(self) -> dict:
        return {"K": self.K, "R": self.R, "T": self.T, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSpec":
        as_t = lambda x: tuple(tuple(r) if isinstance(r, (list, tuple)) else r for r in x)
        return cls(K=as_t(d["K"]), R=as_t(d["R"]), T=tuple(d["T"]), source=d["source"])


@dataclass(frozen=True)
class PhotoView:
    """Fotoğrafın kendi kamerası (HMR2 pred_cam'den türetilir)."""


@dataclass(frozen=True)
class OrbitView:
    """Foto kamerasının pelvis dikey ekseni etrafında döndürülmüş hali."""

    azim_deg: int  # 0/90/180/270 (0 = foto kamerasıyla aynı yön)


ViewSpec = PhotoView | OrbitView


@dataclass(frozen=True)
class GarmentAsset:
    """Yüklenmiş giysi varlığı. Faz 2'de garment.py::load_garment_asset doldurur."""

    garment_id: str
    verts: np.ndarray          # (V,3) float32
    faces: np.ndarray          # (F,3) int64
    uv: np.ndarray | None      # (V,2) float32
    texture: np.ndarray | None  # (Ht,Wt,3) uint8
    lbs_cache: str | None = None  # garment_id.lbs.npz yolu (Faz 2)


@dataclass(frozen=True)
class ConditioningBundle:
    """build_conditioning çıktısı. Tüm tensörler CPU float32, (C,H,W)."""

    agnostic_rgb: torch.Tensor      # (3,H,W) [-1,1]
    inpaint_mask: torch.Tensor      # (1,H,W) {0,1}
    control_normal: torch.Tensor    # (3,H,W) [-1,1] kamera-uzayı sahne normalleri (body+giysi)
    control_depth_sil: torch.Tensor  # (3,H,W) [-1,1]: [depth, depth, giysi silüeti]
    appearance_ref: torch.Tensor    # (3,H,W) [-1,1] flat-lit dokulu UV render
    camera: CameraSpec
    meta: dict = field(default_factory=dict)  # synth modunda meta["gt_rgb"] içerir

    def __post_init__(self):
        h, w = self.inpaint_mask.shape[-2:]
        for name in ("agnostic_rgb", "control_normal", "control_depth_sil", "appearance_ref"):
            t = getattr(self, name)
            if t.shape != (3, h, w):
                raise ValueError(f"{name}: beklenen (3,{h},{w}), gelen {tuple(t.shape)}")
            if t.dtype != torch.float32:
                raise ValueError(f"{name}: float32 olmalı, gelen {t.dtype}")
        if self.inpaint_mask.shape != (1, h, w):
            raise ValueError(f"inpaint_mask: beklenen (1,{h},{w}), gelen {tuple(self.inpaint_mask.shape)}")
        uniq = torch.unique(self.inpaint_mask)
        if not torch.all((uniq == 0) | (uniq == 1)):
            raise ValueError("inpaint_mask ikili {0,1} olmalı")


# --------------------------------------------------------------------------- #
# Tek kaynak fonksiyon
# --------------------------------------------------------------------------- #


def build_conditioning(
    person_image: np.ndarray | None,
    smplx_params: dict[str, Any],
    garment: GarmentAsset,
    view: ViewSpec,
    *,
    size: tuple[int, int] = CANONICAL_SIZE,
    device: str = "cpu",
) -> ConditioningBundle:
    """Koşullama demetini üretir.

    Args:
        person_image: (H,W,3) uint8 RGB fotoğraf; None => sentetik mod
            (gerçek foto yok, GT render meta["gt_rgb"] olarak döner).
        smplx_params: betas(10), body_pose(63), global_orient(3), transl(3),
            pred_cam(3: s,tx,ty), bbox(4: x,y,w,h) — HMR2 adapter sözleşmesi.
        garment: yüklenmiş giysi varlığı (LBS cache'i ile).
        view: PhotoView() = fotoğrafın kamerası; OrbitView(azim) = döndürülmüş.
        size: (height, width); tek geçerli değer CANONICAL_SIZE, testler küçük
            boyut kullanabilir.

    Returns:
        ConditioningBundle — eğitim, sentetik üretim ve inference için birebir aynı.
    """
    if person_image is not None:
        person_image = np.ascontiguousarray(person_image)
        if person_image.ndim != 3 or person_image.shape[2] != 3 or person_image.dtype != np.uint8:
            raise ValueError("person_image (H,W,3) uint8 RGB olmalı")
    required = {"betas", "body_pose", "global_orient", "pred_cam", "bbox"}
    missing = required - set(smplx_params)
    if missing:
        raise ValueError(f"smplx_params eksik alanlar: {sorted(missing)} (hmr2_adapter pred_cam+bbox döndürmeli)")
    if not isinstance(view, (PhotoView, OrbitView)):
        raise TypeError(f"view PhotoView|OrbitView olmalı, gelen {type(view)}")

    return _build_impl(person_image, smplx_params, garment, view, size=size, device=device)


# --------------------------------------------------------------------------- #
# Faz 0 stub implementasyonu — Faz 2'de gerçek hatla değiştirilir
# --------------------------------------------------------------------------- #


def _stable_seed(person_image, smplx_params, garment: GarmentAsset, view: ViewSpec, size) -> int:
    """Girdilerin tamamından deterministik tohum — parite testinin temeli."""
    h = hashlib.sha256()
    h.update(b"none" if person_image is None else person_image.tobytes())
    for key in sorted(k for k in smplx_params if k in ("betas", "body_pose", "global_orient", "transl", "pred_cam", "bbox")):
        h.update(key.encode())
        h.update(np.asarray(smplx_params[key], dtype=np.float64).tobytes())
    h.update(garment.garment_id.encode())
    h.update(garment.verts.astype(np.float64).tobytes())
    view_tag = "photo" if isinstance(view, PhotoView) else f"orbit:{view.azim_deg}"
    h.update(view_tag.encode())
    h.update(json.dumps(size).encode())
    return int.from_bytes(h.digest()[:8], "little")


def _build_impl(person_image, smplx_params, garment, view, *, size, device) -> ConditioningBundle:
    hgt, wdt = size
    gen = torch.Generator().manual_seed(_stable_seed(person_image, smplx_params, garment, view, size))

    def rand_img() -> torch.Tensor:
        return (torch.rand(3, hgt, wdt, generator=gen) * 2 - 1).float()

    mask = (torch.rand(1, hgt, wdt, generator=gen) > 0.7).float()
    view_tag = "photo" if isinstance(view, PhotoView) else f"orbit:{view.azim_deg}"
    camera = CameraSpec(
        K=tuple(map(tuple, np.eye(3).tolist())),
        R=tuple(map(tuple, np.eye(3).tolist())),
        T=(0.0, 0.0, 2.7),
        source=view_tag if person_image is not None else f"synth:{view_tag}",
    )
    meta: dict[str, Any] = {"stub": True, "garment_id": garment.garment_id}
    if person_image is None:
        meta["gt_rgb"] = rand_img()
    return ConditioningBundle(
        agnostic_rgb=rand_img(),
        inpaint_mask=mask,
        control_normal=rand_img(),
        control_depth_sil=rand_img(),
        appearance_ref=rand_img(),
        camera=camera,
        meta=meta,
    )
