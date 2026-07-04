"""Parite sözleşmesi: eğitim yolu ve inference yolu AYNI build_conditioning'i
çağırır ve aynı girdiyle bit-eş çıktı üretir. (v1'in en pahalı dersi.)

Bu test Faz 0 stub'ına karşı da, Faz 2 gerçek implementasyonuna karşı da aynen
çalışmalıdır — imza donmuştur.
"""

import numpy as np
import pytest
import torch

from meshvton2.conditioning import builder
from meshvton2.conditioning.builder import (
    ConditioningBundle,
    GarmentAsset,
    OrbitView,
    PhotoView,
    build_conditioning,
)

SIZE = (64, 48)  # test hızı için küçük; sözleşme boyuttan bağımsız


def _inputs():
    rng = np.random.RandomState(0)
    person = rng.randint(0, 255, (128, 96, 3), dtype=np.uint8)
    smplx_params = {
        "betas": rng.randn(10).astype(np.float32),
        "body_pose": rng.randn(63).astype(np.float32),
        "global_orient": rng.randn(3).astype(np.float32),
        "transl": np.zeros(3, dtype=np.float32),
        "pred_cam": np.array([0.9, 0.01, 0.2], dtype=np.float32),
        "bbox": np.array([10, 5, 80, 120], dtype=np.float32),
    }
    garment = GarmentAsset(
        garment_id="test_tshirt",
        verts=rng.randn(50, 3).astype(np.float32),
        faces=rng.randint(0, 50, (80, 3)).astype(np.int64),
        uv=rng.rand(50, 2).astype(np.float32),
        texture=rng.randint(0, 255, (16, 16, 3), dtype=np.uint8),
    )
    return person, smplx_params, garment


def train_path(person, smplx_params, garment, view):
    """Eğitim ön-işlemesinin çağırdığı yolun temsili (dataset tarafı)."""
    return build_conditioning(person, smplx_params, garment, view, size=SIZE)


def infer_path(person, smplx_params, garment, view):
    """run_tryon'un çağırdığı yolun temsili (inference tarafı)."""
    return build_conditioning(person, smplx_params, garment, view, size=SIZE)


def _assert_bundles_equal(a: ConditioningBundle, b: ConditioningBundle):
    for name in ("agnostic_rgb", "inpaint_mask", "control_normal", "control_depth_sil", "appearance_ref"):
        assert torch.equal(getattr(a, name), getattr(b, name)), f"{name} parite ihlali"
    assert a.camera == b.camera


@pytest.mark.parametrize("view", [PhotoView(), OrbitView(90), OrbitView(180)])
def test_train_and_infer_paths_are_identical(view):
    person, smplx_params, garment = _inputs()
    _assert_bundles_equal(
        train_path(person, smplx_params, garment, view),
        infer_path(person, smplx_params, garment, view),
    )


def test_single_source_of_truth():
    """Gelecekte biri fonksiyonu çatallarsa yakala: modülde tek giriş noktası var."""
    import meshvton2.conditioning.builder as m

    assert m.build_conditioning is build_conditioning
    public = [n for n in dir(m) if n.startswith("build_") and callable(getattr(m, n))]
    assert public == ["build_conditioning"], f"Beklenmeyen ikinci builder: {public}"


def test_different_views_differ():
    """Görüş değişince koşullama da değişmeli (deterministik sabit çıktı hilesine karşı)."""
    person, smplx_params, garment = _inputs()
    front = build_conditioning(person, smplx_params, garment, OrbitView(0), size=SIZE)
    back = build_conditioning(person, smplx_params, garment, OrbitView(180), size=SIZE)
    assert not torch.equal(front.control_normal, back.control_normal)


def test_synthetic_mode_returns_gt():
    _, smplx_params, garment = _inputs()
    bundle = build_conditioning(None, smplx_params, garment, OrbitView(90), size=SIZE)
    assert "gt_rgb" in bundle.meta, "person_image=None (sentetik mod) meta['gt_rgb'] döndürmeli"
    assert bundle.camera.source.startswith("synth")


def test_contract_validation():
    person, smplx_params, garment = _inputs()
    bad = {k: v for k, v in smplx_params.items() if k != "pred_cam"}
    with pytest.raises(ValueError, match="pred_cam"):
        build_conditioning(person, bad, garment, PhotoView(), size=SIZE)
    with pytest.raises(TypeError):
        build_conditioning(person, smplx_params, garment, "front", size=SIZE)


def test_bundle_shape_validation():
    with pytest.raises(ValueError):
        ConditioningBundle(
            agnostic_rgb=torch.zeros(3, 64, 48),
            inpaint_mask=torch.zeros(1, 64, 48),
            control_normal=torch.zeros(3, 32, 48),  # yanlış boyut
            control_depth_sil=torch.zeros(3, 64, 48),
            appearance_ref=torch.zeros(3, 64, 48),
            camera=builder.CameraSpec(
                K=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                R=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                T=(0, 0, 0),
                source="photo",
            ),
        )


def test_camera_spec_roundtrip():
    cam = builder.CameraSpec(
        K=((500.0, 0, 384), (0, 500.0, 512), (0, 0, 1)),
        R=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        T=(0.0, 0.0, 2.7),
        source="orbit:90",
    )
    assert builder.CameraSpec.from_dict(cam.to_dict()) == cam


def test_garment_none_real_data_mode():
    """garment=None (gerçek-veri modu): foto modunda ürün fotoğrafıyla çalışır,
    sentetik modda ve referanssız reddedilir."""
    person, smplx_params, _ = _inputs()
    ref = np.random.RandomState(2).randint(0, 255, (32, 24, 3), dtype=np.uint8)

    a = build_conditioning(person, smplx_params, None, PhotoView(), size=SIZE, appearance_ref_image=ref)
    b = build_conditioning(person, smplx_params, None, PhotoView(), size=SIZE, appearance_ref_image=ref)
    _assert_bundles_equal(a, b)  # parite bu modda da geçerli
    assert a.meta["garment_id"] == "real_worn_garment"

    with pytest.raises(ValueError, match="appearance_ref_image"):
        build_conditioning(person, smplx_params, None, PhotoView(), size=SIZE)
    with pytest.raises(ValueError, match="Sentetik"):
        build_conditioning(None, smplx_params, None, OrbitView(0), size=SIZE, appearance_ref_image=ref)
