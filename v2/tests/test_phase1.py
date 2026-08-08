"""Faz 1 saf yardımcılarının lokal testleri (GPU/pytorch3d/diffusers gerektirmez)."""

import numpy as np
import pytest

from meshvton2.conditioning.person import apply_agnostic
from meshvton2.conditioning.render import center_unit, zup_to_yup
from meshvton2.model.flux_tryon import (
    composite_outside_mask,
    crop_half,
    make_side_canvas,
    make_side_mask,
)


def test_side_canvas_roundtrip():
    left = np.full((8, 6, 3), 10, np.uint8)
    right = np.full((8, 6, 3), 200, np.uint8)
    canvas = make_side_canvas(left, right)
    assert canvas.shape == (8, 12, 3)
    assert np.array_equal(crop_half(canvas, "left"), left)
    assert np.array_equal(crop_half(canvas, "right"), right)
    with pytest.raises(ValueError):
        make_side_canvas(left, np.zeros((4, 6, 3), np.uint8))


def test_side_mask_reference_half_is_zero():
    mask = np.full((8, 6), 255, np.uint8)
    cmask = make_side_mask(mask)
    assert cmask.shape == (8, 12)
    assert cmask[:, :6].sum() == 0  # referans yarısı asla inpaint edilmez
    assert (cmask[:, 6:] == 255).all()


def test_composite_outside_mask_preserves_original():
    pred = np.full((32, 32, 3), 255, np.uint8)
    orig = np.zeros((32, 32, 3), np.uint8)
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 255
    out = composite_outside_mask(pred, orig, mask)
    assert out[0, 0].tolist() == [0, 0, 0]        # maske dışı = orijinal
    assert out[16, 16].tolist() == [255, 255, 255]  # maske içi = tahmin


def test_apply_agnostic():
    img = np.full((10, 10, 3), 50, np.uint8)
    mask = np.zeros((10, 10), np.uint8)
    mask[2:5, 2:5] = 255
    out = apply_agnostic(img, mask)
    assert out[3, 3].tolist() == [128, 128, 128]
    assert out[0, 0].tolist() == [50, 50, 50]
    assert img[3, 3].tolist() == [50, 50, 50]  # girdi mutasyonu yok


def test_zup_to_yup():
    v = np.array([[1.0, 2.0, 3.0]], np.float32)
    assert zup_to_yup(v).tolist() == [[1.0, 3.0, -2.0]]


def test_center_unit():
    v = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    out = center_unit(v)
    assert np.allclose(out.mean(axis=0), 0)
    assert np.abs(out).max() == pytest.approx(1.0)


def test_find_sibling_texture(tmp_path):
    from meshvton2.conditioning.garment import find_sibling_texture

    obj = tmp_path / "model.obj"
    obj.write_text("v 0 0 0\n")
    assert find_sibling_texture(obj) is None
    (tmp_path / "random.png").write_bytes(b"x")
    assert find_sibling_texture(obj).endswith("random.png")  # klasördeki tek görüntü
    (tmp_path / "model.png").write_bytes(b"x")
    assert find_sibling_texture(obj).endswith("model.png")  # aynı ad öncelikli


def test_pick_garments_nested_categories(tmp_path):
    """CLOTH3D Drive düzeni: garments_3d/kategori/giysi/model.obj (+texture)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "build_golden_set", Path(__file__).parents[1] / "scripts" / "build_golden_set.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    (tmp_path / "upper_body/g1").mkdir(parents=True)
    (tmp_path / "upper_body/g1/model.obj").write_text("v 0 0 0\n")
    (tmp_path / "upper_body/g1/tex.png").write_bytes(b"x")
    (tmp_path / "dresses/g2").mkdir(parents=True)
    (tmp_path / "dresses/g2/model.obj").write_text("v 0 0 0\n")  # texturesiz — yine de dahil edilmeli
    (tmp_path / "dresses/g3").mkdir(parents=True)
    (tmp_path / "dresses/g3/model.obj").write_text("v 0 0 0\n")
    (tmp_path / "dresses/g3/g3.png").write_bytes(b"x")

    picked = mod.pick_garments(tmp_path, None)
    ids = [g.id for g in picked]
    assert "upper_body__g1" in ids and "dresses__g3" in ids
    assert "dresses__g2" in ids                          # texture şart değil (KALICI kural)
    assert ids[0] == "upper_body__g1"                    # upper_body öncelikli
    g1 = picked[ids.index("upper_body__g1")]
    assert g1.mesh == "upper_body/g1/model.obj" and g1.category == "upper_body"
    g2 = picked[ids.index("dresses__g2")]
    assert g2.texture is None


def test_garment_ref_from_person(tmp_path):
    """Ürün fotoğrafı yoksa referans kişinin üzerindeki giysiden kesilir (beyaz zemin)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "preprocess_vitonhd", Path(__file__).parents[1] / "scripts" / "preprocess_vitonhd.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    img = np.full((100, 80, 3), 30, np.uint8)
    parse = np.zeros((50, 40), np.uint8)   # düşük çözünürlüklü parse (gerçekte 512x384)
    parse[10:30, 10:30] = 4                # upper_clothes bölgesi
    img[20:60, 20:60] = (200, 40, 40)      # giysi pikselleri kırmızı

    ref = mod.garment_ref_from_person(img, parse)
    assert ref is not None and ref.ndim == 3
    assert (ref == 255).all(axis=2).any()          # beyaz zemin var
    assert (ref[..., 0] > 150).sum() > 100         # giysi pikselleri taşınmış

    assert mod.garment_ref_from_person(img, np.zeros((50, 40), np.uint8)) is None  # giysi yok
