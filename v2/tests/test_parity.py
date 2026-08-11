"""Parity contract: the training path and the inference path call the SAME
build_conditioning and produce bit-identical output for the same input. (v1's most expensive lesson.)

This test must work unchanged against both the Phase 0 stub and the Phase 2 real
implementation — the signature is frozen.
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

SIZE = (64, 48)  # small for test speed; the contract is size-independent


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
    """Stands in for the path training preprocessing calls (dataset side)."""
    return build_conditioning(person, smplx_params, garment, view, size=SIZE)


def infer_path(person, smplx_params, garment, view):
    """Stands in for the path run_tryon calls (inference side)."""
    return build_conditioning(person, smplx_params, garment, view, size=SIZE)


def _assert_bundles_equal(a: ConditioningBundle, b: ConditioningBundle):
    for name in ("agnostic_rgb", "inpaint_mask", "control_normal", "control_depth_sil", "appearance_ref"):
        assert torch.equal(getattr(a, name), getattr(b, name)), f"{name} parity violation"
    assert a.camera == b.camera


@pytest.mark.parametrize("view", [PhotoView(), OrbitView(90), OrbitView(180)])
def test_train_and_infer_paths_are_identical(view):
    person, smplx_params, garment = _inputs()
    _assert_bundles_equal(
        train_path(person, smplx_params, garment, view),
        infer_path(person, smplx_params, garment, view),
    )


def test_single_source_of_truth():
    """Catch anyone forking the function later: the module has a single entry point."""
    import meshvton2.conditioning.builder as m

    assert m.build_conditioning is build_conditioning
    public = [n for n in dir(m) if n.startswith("build_") and callable(getattr(m, n))]
    assert public == ["build_conditioning"], f"Unexpected second builder: {public}"


def test_different_views_differ():
    """The conditioning must change when the view changes (guards against a constant-output trick)."""
    person, smplx_params, garment = _inputs()
    front = build_conditioning(person, smplx_params, garment, OrbitView(0), size=SIZE)
    back = build_conditioning(person, smplx_params, garment, OrbitView(180), size=SIZE)
    assert not torch.equal(front.control_normal, back.control_normal)


def test_synthetic_mode_returns_gt():
    _, smplx_params, garment = _inputs()
    bundle = build_conditioning(None, smplx_params, garment, OrbitView(90), size=SIZE)
    assert "gt_rgb" in bundle.meta, "person_image=None (synthetic mode) must return meta['gt_rgb']"
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
            control_normal=torch.zeros(3, 32, 48),  # wrong size
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
    """garment=None (real-data mode): works with a product photo in photo mode,
    rejected in synthetic mode and without a reference."""
    person, smplx_params, _ = _inputs()
    ref = np.random.RandomState(2).randint(0, 255, (32, 24, 3), dtype=np.uint8)

    a = build_conditioning(person, smplx_params, None, PhotoView(), size=SIZE, appearance_ref_image=ref)
    b = build_conditioning(person, smplx_params, None, PhotoView(), size=SIZE, appearance_ref_image=ref)
    _assert_bundles_equal(a, b)  # parity holds in this mode too
    assert a.meta["garment_id"] == "real_worn_garment"

    with pytest.raises(ValueError, match="appearance_ref_image"):
        build_conditioning(person, smplx_params, None, PhotoView(), size=SIZE)
    with pytest.raises(ValueError, match="Synthetic"):
        build_conditioning(None, smplx_params, None, OrbitView(0), size=SIZE, appearance_ref_image=ref)


def test_prealign_garment_no_rescale_hang_anchor():
    """The CLOTH3D metric scale is PRESERVED (both scaling heuristics blew up in QA);
    upper body hangs from the shoulder, lower body from the waist; x/z is centred on the pelvis."""
    from meshvton2.conditioning.builder import DEFAULT_HANG_PAD, _prealign_garment

    j = np.zeros((22, 3))
    j[0] = [0.1, 0, 0.05]                              # pelvis (deliberately offset)
    j[16], j[17] = [-0.08, 0.50, 0], [0.28, 0.50, 0]   # shoulders (mid_sh y=0.5)
    rest = {"joints": j}

    # Z-up "t-shirt": x width 0.6, y depth 0.24, z height 0.5 (metres)
    rng = np.random.RandomState(0)
    g = rng.uniform(-1, 1, (300, 3)) * [0.3, 0.12, 0.25]
    out = _prealign_garment(g, rest, "upper_body__X_Tshirt")

    # scale preserved: the horizontal width is still ~0.6 (x is unchanged after Z-up→Y-up)
    assert (out[:, 0].max() - out[:, 0].min()) == pytest.approx(g[:, 0].max() - g[:, 0].min(), rel=1e-9)
    # hang: top edge = shoulder + DEFAULT_HANG_PAD (-0.06 → 0.44). A positive pad raised the
    # collar to chin level and bound it to the neck/head in K-NN (2026-08-09 diagnosis).
    assert out[:, 1].max() == pytest.approx(0.50 + DEFAULT_HANG_PAD, abs=1e-9)
    # x/z centred on the pelvis
    assert (out[:, 0].max() + out[:, 0].min()) / 2 == pytest.approx(0.1, abs=1e-9)
    assert (out[:, 2].max() + out[:, 2].min()) / 2 == pytest.approx(0.05, abs=1e-9)

    # lower body hangs from the waist (top edge = pelvis_y + DEFAULT_HANG_PAD)
    low = _prealign_garment(g, rest, "lower_body__X_Trousers")
    assert low[:, 1].max() == pytest.approx(0.0 + DEFAULT_HANG_PAD, abs=1e-9)
