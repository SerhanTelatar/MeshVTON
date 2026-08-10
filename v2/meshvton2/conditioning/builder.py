"""build_conditioning() — koşullama üretiminin TEK kaynağı (parite sözleşmesi).

v1'in en pahalı dersi: eğitim ön-işlemesi ile inference koşullaması farklı kod
yollarından üretilince ControlNet dağıtım-dışı girdi aldı ve sonuçları bozdu.
v2 kuralı: eğitim ön-işleme (scripts/preprocess_vitonhd.py), sentetik üretici
(synth/generate.py, person_image=None modu) ve inference (inference/run_tryon.py)
İSTİSNASIZ bu modüldeki build_conditioning()'i çağırır. tests/test_parity.py
bunu iki yoldan aynı girdiyle çağırıp tensör eşitliğini zorlar.

Bu imza Faz 0'da donmuştur; alan eklemek serbest, mevcut alanı değiştirmek yasak.

İMPLEMENTASYON SEÇİMİ: gerçek hat (SMPL-X + LBS drape + pyrender) varsayılandır.
`MESHVTON2_STUB=1` ortam değişkeni deterministik stub'ı seçer — YALNIZ 3D
bağımlılıkları olmayan geliştirme makinesindeki testler için (tests/conftest.py
bunu otomatik ayarlar). Üretim scriptleri `assert_real_impl()` çağırır: v1'in
"sessiz placeholder" felaketi yapısal olarak engellenir.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

CANONICAL_SIZE = (1024, 768)  # (height, width) — configs/base.yaml ile aynı

# Askı payı [m]: giysi üst kenarının omuz (üst giyim) / pelvis (alt giyim) referansına
# göre dikey ofseti.
#
# +0.06 DOĞRU DEĞERDİR, DEĞİŞTİRME. Gerekçe: SMPL j[16]/j[17] omuz EKLEMİ merkezidir,
# omzun üstü değil (~6cm altı) — tişört yaka/omuz dikişi ancak +6cm ile omza oturur.
# 2026-07-04 sentetik drape QA'sı bu değerle geçti.
#
# 2026-08-09'da kısa süre -0.06 denendi ([[meshvton-v2-inference-alignment]]'teki foto
# kalibrasyonuna dayanarak) ve QA görselinde giysi KOLTUKALTINA kaydı (tişört straplez
# büstiyer gibi göründü) → geri alındı. Oradaki -0.06, düzeltilmemiş person_square_bbox
# hatasını telafi eden bir yamaydı; bbox artık her yolda doğru olduğu için gereksiz.
DEFAULT_HANG_PAD = 0.06

# FOTOĞRAF yolu için AYRI değer. 2026-08-10 ampirik kalibrasyonu
# (v2/scripts/calibrate_hang.py, kamera kapısından geçmiş 5 kişi): +0.06 giysiyi
# gerçek giysiye göre ~21 puan YUKARI asıyor (yüzün üstüne biniyor);
# mesh<->parser IoU +0.06'da 0.295, -0.12'de 0.480. Uçtan uca doğrulandı:
# yerleşim IoU (placement_iou.py) 0.5815 -> 0.6829, mesh ayırt ediciliği değişmedi
# (+0.0236 -> +0.0219, aynı 10 kombo). geo_iou DÜŞTÜ ama o metrik kendi verdiğimiz
# silüeti hedef aldığı için hedefin bozukluğunu göremiyor — bkz. placement_iou.py.
#
# SENTETİK yol +0.06'da KALIR: orada gövde de giysi de aynı SMPL-X'ten geliyor ve
# 2026-07-04 drape QA'sı o değerle geçti; -0.06 denendiğinde giysi koltukaltına
# kaymıştı. İki yolun farklı değer istemesi HMR2 (foto) ile SMPL-X (sentetik)
# gövdelerinin farklı olmasından; tek global değer ikisine birden uymuyor.
PHOTO_HANG_PAD = -0.12

# FOTOĞRAF yolu giysi ölçeği. 2026-08-10 kalibrasyonu (calibrate_scale.py, aynı 5
# doğrulanmış kişi, hang_pad=-0.12): mesh<->parser IoU 1.00'da 0.481, tepe 1.25'te
# 0.555. Tepe SINIRDA değil (1.10-1.70 tarandı, 1.30'dan sonra düşüyor) ve bağımsız
# bir tutarlılık kontrolü geçiyor: 1.25'te silüet alanı %19.2, parser'ın ölçtüğü
# gerçek giysi alanı %18.8 — iki farklı yoldan aynı nokta.
#
# Yorum: CLOTH3D giysileri HMR2'nin kestirdiği bedenlere göre sistematik olarak
# KÜÇÜK kalıyor (sentetik/gerçek beden dağılımı uyumsuzluğu). SENTETİK yol 1.0'da
# KALIR — orada gövde de giysi de aynı SMPL-X'ten gelir ve yeniden ölçekleme
# geçmişte iki kez hata çıktı (bkz. _prealign_garment ders zinciri).
PHOTO_GARMENT_SCALE = 1.25

# Sentetik gövde teni için geri-uyumlu varsayılan (skin_rgb verilmezse).
SYNTH_DEFAULT_SKIN = (0.80, 0.62, 0.52)


def implementation() -> str:
    """"real" | "stub" — her çağrıda env'den okunur (test izolasyonu için)."""
    return "stub" if os.environ.get("MESHVTON2_STUB") == "1" else "real"


def assert_real_impl() -> None:
    """Üretim giriş noktaları (synth üretici, ön-işleme, run_tryon) bunu çağırır."""
    if implementation() != "real":
        raise RuntimeError(
            "build_conditioning STUB modda (MESHVTON2_STUB=1) — üretimde yasak. "
            "Bu, v1'in 'eğitilmemiş placeholder sessizce çalıştı' hatasının panzehiridir."
        )


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
    appearance_ref: torch.Tensor    # (3,H,W) [-1,1] GÖLGELİ gri kumaş render'ı (texture ASLA)
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
    garment: GarmentAsset | None,
    view: ViewSpec,
    *,
    size: tuple[int, int] = CANONICAL_SIZE,
    device: str = "cpu",
    person_prep: Any | None = None,  # PersonPrep — foto modunda ZORUNLU (agnostic+mask kaynağı)
    appearance_ref_image: np.ndarray | None = None,  # garment=None modunda ZORUNLU (ürün fotoğrafı)
    geometry_mask: bool = True,  # foto+mesh: maske = parse ∪ dilate(giysi silüeti); False = yalnız parse
    hang_pad: float = DEFAULT_HANG_PAD,  # giysi üst kenarı askı payı [m] (bkz. DEFAULT_HANG_PAD)
    garment_scale: float = 1.0,  # 1.0 = CLOTH3D metrik ölçeği (bkz. _prealign_garment)
    skin_rgb: tuple | None = None,  # sentetik modda gövde teni; None => SYNTH_DEFAULT_SKIN
) -> ConditioningBundle:
    """Koşullama demetini üretir.

    Args:
        person_image: (H,W,3) uint8 RGB fotoğraf; None => sentetik mod
            (gerçek foto yok, GT render meta["gt_rgb"] olarak döner).
        smplx_params: betas(10), body_pose(63), global_orient(3), transl(3),
            pred_cam(3: s,tx,ty), bbox(4: x,y,w,h) — HMR2 adapter sözleşmesi.
        garment: yüklenmiş giysi varlığı (LBS cache'i ile) VEYA None = "gerçek-veri
            modu": kişinin GİYDİĞİ giysinin mesh'i yok (VITON-HD eğitimi) →
            geometri gövde-only, giysi silüeti parse'tan, görünüm referansı
            appearance_ref_image'dan (ürün fotoğrafı). Yalnız foto modunda geçerli;
            süpervizyon tutarlılığının şartı (GT'deki giysi ≠ rastgele mesh olamaz).
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
    if garment is None:
        if person_image is None:
            raise ValueError("Sentetik mod (person_image=None) giysi mesh'i olmadan üretilemez")
        if appearance_ref_image is None:
            raise ValueError("garment=None modunda appearance_ref_image (ürün fotoğrafı) zorunlu")

    if implementation() == "stub":
        return _build_impl_stub(
            person_image, smplx_params, garment, view,
            size=size, device=device, appearance_ref_image=appearance_ref_image,
        )
    return _build_impl_real(
        person_image, smplx_params, garment, view, size=size, device=device,
        person_prep=person_prep, appearance_ref_image=appearance_ref_image,
        geometry_mask=geometry_mask, hang_pad=hang_pad, garment_scale=garment_scale,
        skin_rgb=skin_rgb,
    )


# --------------------------------------------------------------------------- #
# Gerçek implementasyon (Faz 2): SMPL-X + LBS drape + ekran-uzayı render
# --------------------------------------------------------------------------- #


def _to_tensor01(img01: np.ndarray) -> torch.Tensor:
    """(H,W,3) [0,1] float -> (3,H,W) float32 [-1,1]."""
    return torch.from_numpy(np.ascontiguousarray(img01.transpose(2, 0, 1))).float() * 2.0 - 1.0


def _prealign_garment(gverts: np.ndarray, rest: dict, garment_id: str = "",
                      hang_pad: float = DEFAULT_HANG_PAD, scale: float = 1.0) -> np.ndarray:
    """Bağlama öncesi hizalama: ÖLÇEK YOK, sadece eksen çevir + askı hizala.

    Ders zinciri (QA'da yakalandı): (1) gövde-genişliği ölçeği T-poz kulaç
    açıklığını ölçüp giysiyi 3x şişirdi; (2) omuz-heuristik ölçeği ise zaten
    DOĞRU ölçekli CLOTH3D giysisini küçültüp gövdenin içine soktu (clearance
    ~0.9). Gerçek: CLOTH3D giysileri metre ölçeğinde ve SMPL bedenine göre
    modelli — yeniden ölçekleme her durumda hataydı. Şimdi: Z-up→Y-up çevir,
    x/z'yi pelvise ortala, dikeyde 'askı' hizala: üst/elbise omuzdan asılır
    (üst kenar ≈ omuz hizası + pay), alt giyim belden (üst kenar ≈ pelvis + pay)."""
    from meshvton2.conditioning.render import zup_to_yup

    j = rest["joints"]
    pelvis = j[0]
    mid_sh = (j[16] + j[17]) / 2.0

    g = zup_to_yup(np.asarray(gverts, np.float64))  # metrik ölçek KORUNUR
    # scale=1.0 varsayılanı yukarıdaki "yeniden ölçekleme HER ZAMAN hataydı" dersini
    # korur. 1.0'dan farklı değer YALNIZCA ampirik kalibrasyon içindir
    # (v2/scripts/calibrate_scale.py): foto yolunda gövde HMR2'den geldiği için
    # CLOTH3D'nin SMPL-referanslı ölçeği tam oturmayabiliyor. Ölçek giysinin KENDİ
    # merkezine göre uygulanır; ardından gelen ortalama+askı hizalaması bozulmaz.
    if scale != 1.0:
        c = (g.max(axis=0) + g.min(axis=0)) / 2.0
        g = (g - c) * scale + c
    g[:, 0] += pelvis[0] - (g[:, 0].max() + g[:, 0].min()) / 2.0
    g[:, 2] += pelvis[2] - (g[:, 2].max() + g[:, 2].min()) / 2.0
    # Hizalama giysinin ÜST KENARINDAN (max y) yapılır — tişörtte yaka rimi, straplez
    # üstte göğüs bandı. Tek global ofset her giysi tipi için ideal olamaz; varsayılan
    # tişört/top ailesine göre seçilidir (bkz. DEFAULT_HANG_PAD).
    hang_y = (pelvis[1] + hang_pad) if "lower_body" in garment_id else (mid_sh[1] + hang_pad)
    g[:, 1] += hang_y - g[:, 1].max()  # rest gövde y-YUKARI: üst kenar = max(y)
    return g


def _get_binding(garment: GarmentAsset, body_model, hang_pad: float = DEFAULT_HANG_PAD,
                 scale: float = 1.0) -> "GarmentBinding":  # noqa: F821
    """Giysi bağlamasını cache'ten yükler ya da rest gövdeye kurar ve cache'ler.
    Cache anahtarı hang_pad VE scale'i İÇERİR — farklı askı/ölçek farklı bağlamadır
    (eskiden cache yalnız +0.06'da açıktı; varsayılan değişince sessizce kapanırdı;
    scale eklenmeseydi kalibrasyon taraması ilk ölçeğin cache'ini okurdu)."""
    from meshvton2.conditioning.lbs_drape import GarmentBinding, bind_garment

    cache_path = None
    if garment.lbs_cache:
        p = Path(garment.lbs_cache)
        key = f".hp{int(round(hang_pad * 1000)):+04d}"
        if scale != 1.0:
            key += f".sc{int(round(scale * 1000)):04d}"
        cache_path = p.with_name(f"{p.stem}{key}{p.suffix}")
        if cache_path.exists():
            try:
                return GarmentBinding.load(cache_path)
            except Exception:  # paralel işçi yarışında yarım yazılmış cache — yeniden kur
                pass
    rest = body_model.rest()
    aligned = _prealign_garment(garment.verts, rest, garment.garment_id,
                                hang_pad=hang_pad, scale=scale)
    binding = bind_garment(aligned, rest["verts"], rest["faces"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        binding.save(cache_path)
    return binding


ATR_GARMENT_LABELS = (4, 7)  # upper_clothes, dress — parse'tan giysi silüeti (garment=None modu)


def _build_impl_real(
    person_image, smplx_params, garment, view, *, size, device,
    person_prep=None, appearance_ref_image=None, geometry_mask=True, hang_pad=DEFAULT_HANG_PAD,
    garment_scale=1.0, skin_rgb=None,
):
    import cv2

    from meshvton2.conditioning import camera as cam_mod
    from meshvton2.conditioning.body import get_body_model
    from meshvton2.conditioning.lbs_drape import apply_binding, push_clearance
    from meshvton2.conditioning.render import (
        force_textureless,
        render_appearance_ref,
        render_geometry,
        render_textured_scene,
    )

    if person_image is not None and person_prep is None:
        raise ValueError(
            "Foto modunda person_prep zorunlu (PersonPreprocessor.process çıktısı) — "
            "agnostic+mask oradan gelir; builder parser yüklemez."
        )
    hgt, wdt = size

    # 1) Gövde + kamera (azimuth tahmini YOK: foto kamerası pred_cam'den, diğerleri orbit)
    body_model = get_body_model()
    body = body_model(smplx_params)
    cam0 = cam_mod.photo_camera(smplx_params, size)
    cam = cam0 if isinstance(view, PhotoView) else cam_mod.orbit_camera(cam0, body["pelvis"], view.azim_deg)

    meta: dict[str, Any] = {"view": "photo" if isinstance(view, PhotoView) else f"orbit:{view.azim_deg}"}

    if garment is not None:
        # 2) Drape: rest-binding (cache'li) -> pozlu gövdeye uygula -> clearance
        binding = _get_binding(garment, body_model, hang_pad=hang_pad, scale=garment_scale)
        gverts = apply_binding(binding, body["verts"])
        gverts, clearance_ratio, pen_depth = push_clearance(gverts, body["verts"], body["faces"])
        # Patlama dedektörü: drape edilmiş giysi gövdeden ne kadar büyük?
        # (clearance bunu YAKALAYAMAZ — patlayan giysi gövdenin dışındadır;
        # QA'da 3-4x boyutlu parçalanmış giysi görüldü, bu metrik onun kapısı)
        diag = lambda v: float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
        extent_ratio = diag(gverts) / max(diag(body["verts"]), 1e-8)

        # 3) Ekran-uzayı geometri + görünüm referansı (mesh render'ı)
        geo = render_geometry(body["verts"], body["faces"], gverts, garment.faces, cam, size=size)
        garment_sil = geo["garment_sil"].astype(np.float32)
        # KALICI KURAL: sistem TEXTURESUZ çalışır — appearance ref HER ZAMAN düz
        # gri kumaşlı ŞEKİLLİ render'dır (düz gri kart değil: giysinin formu modele
        # geçer, renk/desen bilgisi asla geçmez), garment.texture var olsun ya da
        # olmasın (bkz. proje kuralı: appearance sadakati hedef değil).
        appearance01 = render_appearance_ref(force_textureless(garment), size=size).astype(np.float32) / 255.0
        meta.update(garment_id=garment.garment_id, clearance_ratio=clearance_ratio,
                    penetration_depth=pen_depth, drape_extent_ratio=extent_ratio)
    else:
        # Gerçek-veri modu: gövde-only geometri; giysi silüeti PARSE'tan,
        # görünüm referansı ürün fotoğrafından (süpervizyon tutarlılığı).
        geo = render_geometry(body["verts"], body["faces"], None, None, cam, size=size)
        parse = np.asarray(person_prep.parse)
        parse = cv2.resize(parse, (wdt, hgt), interpolation=cv2.INTER_NEAREST)
        garment_sil = np.isin(parse, ATR_GARMENT_LABELS).astype(np.float32)
        ref = cv2.resize(appearance_ref_image, (wdt, hgt), interpolation=cv2.INTER_AREA)
        appearance01 = ref.astype(np.float32) / 255.0
        meta.update(garment_id="real_worn_garment", clearance_ratio=None)

    depth_sil01 = np.stack([geo["depth"], geo["depth"], garment_sil], axis=2)

    # 4) Agnostic + maske
    if person_image is None:  # sentetik mod: GT render'dan türet
        # TEN RENGİ: eskiden HER örnekte tek sabit bej (0.80,0.62,0.52) idi — model
        # ten çeşitliliğini hiç görmedi, teni o tondan uzak kişilerde sistematik
        # yanlılık üretti (2026-08-10 teşhisi). Artık örnek başına verilebiliyor;
        # üretici her örnek için rastgele bir ton geçer (synth/generate.py).
        skin01 = np.array(skin_rgb, np.float64) / 255.0 if skin_rgb else SYNTH_DEFAULT_SKIN
        skin = np.full((len(body["verts"]), 3), skin01)
        # GT giysi HER ZAMAN gri (appearance ref ile AYNI görünüm).
        # 2026-08-09 teşhisi: GT texture'lıyken ref gri kalıyordu → aynı girdiye
        # bazen desenli bazen gri hedef; desen girdiden ÖNGÖRÜLEMEZ olduğu için
        # akış-eşleme kaybının optimumu koşullu ORTALAMA = solgun/yarı saydam leke.
        # Görev ancak hedef de gri olunca öğrenilebilir hale gelir.
        gt = render_textured_scene(body["verts"], body["faces"], skin, gverts,
                                   force_textureless(garment), cam, size=size)
        kernel = np.ones((wdt // 30, wdt // 30), np.uint8)
        mask_u8 = cv2.dilate((geo["garment_sil"] * 255).astype(np.uint8), kernel)
        agnostic = gt.copy()
        # PARİTE: dolgu rejimi foto yoluyla AYNI anahtardan okunur. Kapalıyken düz
        # gri (mevcut checkpoint'in eğitildiği rejim), açıkken gövdeyle aynı ten.
        from meshvton2.conditioning.person import AGNOSTIC_SKIN_FILL

        agnostic[mask_u8 > 127] = (
            tuple(int(round(v * 255)) for v in skin01) if AGNOSTIC_SKIN_FILL else (128, 128, 128)
        )
        meta["gt_rgb"] = _to_tensor01(gt.astype(np.float32) / 255.0)
    else:
        agnostic = person_prep.agnostic
        mask_u8 = person_prep.mask
        if agnostic.shape[:2] != (hgt, wdt):
            raise ValueError(f"person_prep boyutu {agnostic.shape[:2]} != hedef {(hgt, wdt)}")
        if garment is not None and geometry_mask:
            # HİZALAMA DÜZELTMESİ: parse maskesi kişinin ESKİ giysisini işaretler;
            # mesh silüeti ondan geniş/kaymış olabilir → model maskeye hapsolur,
            # geometri dışarıda "hayalet" bırakır. Maske = parse ∪ dilate(sil)
            # (sentetik eğitim maskesi de dilate(sil) — aynı rejim, bkz. yukarısı).
            from meshvton2.conditioning.person import apply_agnostic

            kernel = np.ones((wdt // 30, wdt // 30), np.uint8)
            sil_u8 = cv2.dilate((geo["garment_sil"] * 255).astype(np.uint8), kernel)
            mask_u8 = np.maximum(mask_u8, sil_u8)
            agnostic = apply_agnostic(person_prep.image, mask_u8)
            meta["mask_source"] = "parse+garment_sil"

    return ConditioningBundle(
        agnostic_rgb=_to_tensor01(agnostic.astype(np.float32) / 255.0),
        inpaint_mask=torch.from_numpy((mask_u8 > 127).astype(np.float32)).unsqueeze(0),
        control_normal=_to_tensor01(geo["normal"]),
        control_depth_sil=_to_tensor01(depth_sil01),
        appearance_ref=_to_tensor01(appearance01),
        camera=cam,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Stub implementasyonu (yalnız 3D bağımlılıksız test ortamı; bkz. assert_real_impl)
# --------------------------------------------------------------------------- #


def _stable_seed(person_image, smplx_params, garment: GarmentAsset | None, view: ViewSpec, size, appearance_ref_image=None) -> int:
    """Girdilerin tamamından deterministik tohum — parite testinin temeli."""
    h = hashlib.sha256()
    h.update(b"none" if person_image is None else person_image.tobytes())
    for key in sorted(k for k in smplx_params if k in ("betas", "body_pose", "global_orient", "transl", "pred_cam", "bbox")):
        h.update(key.encode())
        h.update(np.asarray(smplx_params[key], dtype=np.float64).tobytes())
    if garment is not None:
        h.update(garment.garment_id.encode())
        h.update(garment.verts.astype(np.float64).tobytes())
    else:
        h.update(b"nogarment")
        h.update(np.ascontiguousarray(appearance_ref_image).tobytes())
    view_tag = "photo" if isinstance(view, PhotoView) else f"orbit:{view.azim_deg}"
    h.update(view_tag.encode())
    h.update(json.dumps(size).encode())
    return int.from_bytes(h.digest()[:8], "little")


def _build_impl_stub(person_image, smplx_params, garment, view, *, size, device, appearance_ref_image=None) -> ConditioningBundle:
    hgt, wdt = size
    gen = torch.Generator().manual_seed(
        _stable_seed(person_image, smplx_params, garment, view, size, appearance_ref_image)
    )

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
    meta: dict[str, Any] = {"stub": True, "garment_id": garment.garment_id if garment else "real_worn_garment"}
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
