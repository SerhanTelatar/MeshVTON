"""Metrik doğruluğu + harness uçtan uca sıhhati (Faz 0 çıkış kriteri)."""

import numpy as np
import pytest
from PIL import Image

from meshvton2.eval import metrics as M
from meshvton2.eval import harness
from meshvton2.eval.golden_set import (
    GoldenGarment,
    GoldenManifest,
    GoldenPerson,
    load_manifest,
)


def _img(seed=0, size=(64, 48)):
    return np.random.RandomState(seed).randint(0, 255, (*size, 3), dtype=np.uint8)


# --------------------------- metrikler --------------------------- #


def test_identity_metrics():
    img = _img()
    assert M.compute_ssim(img, img) == pytest.approx(1.0)
    assert M.compute_psnr(img, img) == float("inf")
    mask = np.zeros((64, 48), bool)
    mask[20:40, 10:30] = True
    assert M.compute_ssim(img, img, mask=mask) == pytest.approx(1.0)


def test_metrics_degrade_on_difference():
    a, b = _img(0), _img(1)
    assert M.compute_ssim(a, b) < 0.5
    assert M.compute_psnr(a, b) < 15


def test_delta_e_zero_for_same_color():
    img = np.full((32, 32, 3), (200, 30, 60), dtype=np.uint8)
    mask = np.ones((32, 32), bool)
    assert M.garment_delta_e(img, mask, img.copy(), ref_mask=mask) == pytest.approx(0.0, abs=1e-6)
    other = np.full((32, 32, 3), (30, 200, 60), dtype=np.uint8)
    assert M.garment_delta_e(img, mask, other, ref_mask=mask) > 20


def test_silhouette_iou():
    a = np.zeros((10, 10), bool)
    a[2:8, 2:8] = True
    assert M.silhouette_iou(a, a) == pytest.approx(1.0)
    b = np.zeros((10, 10), bool)
    assert M.silhouette_iou(a, b) == pytest.approx(0.0)
    assert M.silhouette_iou(b, b) is None  # boş∪boş tanımsız — None, 0 değil


def test_empty_mask_returns_none_not_zero():
    """v1 dersi: hesaplanamayan metrik 0.0 YALANI dönmez."""
    img = _img()
    empty = np.zeros((64, 48), bool)
    assert M.compute_ssim(img, img, mask=empty) is None
    assert M.garment_delta_e(img, empty, img) is None


# --------------------------- manifest --------------------------- #


def _tiny_manifest(root):
    return GoldenManifest(
        root=root,
        persons=[GoldenPerson(id="p1", image="persons/p1.png", source="vitonhd_test")],
        garments=[GoldenGarment(id="g1", mesh="g1/model.obj", texture="g1/tex.png")],
        angles=[0, 180],
    )


def test_manifest_roundtrip(tmp_path):
    m = _tiny_manifest(tmp_path)
    m.save(tmp_path / "manifest.json")
    loaded = load_manifest(tmp_path / "manifest.json")
    assert [c for c in loaded.items()] == [c for c in m.items()]
    assert len(list(loaded.items())) == 2  # 1 combo × 2 açı


# --------------------------- harness --------------------------- #


def test_harness_end_to_end(tmp_path):
    m = _tiny_manifest(tmp_path)
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()

    pred = _img(0)
    person = _img(1)
    mask = np.zeros((64, 48), np.uint8)
    mask[20:40, 10:30] = 255
    # 0°: tüm aux dosyalarıyla; 180°: kasıtlı eksik tahmin
    stem = pred_dir / "p1_g1_000"
    Image.fromarray(pred).save(stem.with_suffix(".png"))
    Image.fromarray(person).save(f"{stem}.person.png")
    Image.fromarray(mask).save(f"{stem}.mask.png")
    Image.fromarray(mask).save(f"{stem}.sil.png")
    Image.fromarray(pred).save(f"{stem}.ref.png")
    Image.fromarray(pred).save(f"{stem}.gt.png")

    summary = harness.evaluate(m, pred_dir)
    assert summary["total"] == 2 and summary["found"] == 1
    assert summary["missing"] == ["p1_g1_180"]
    assert summary["overall"]["gt_ssim"]["mean"] == pytest.approx(1.0)  # pred == gt
    assert summary["overall"]["silhouette_iou"]["mean"] == pytest.approx(1.0)  # mask == sil
    assert summary["overall"]["outside_mask_ssim"]["mean"] is not None

    out = tmp_path / "report.json"
    harness.write_report(summary, out)
    assert out.exists() and out.with_suffix(".md").exists()
    assert "gt_ssim" in out.with_suffix(".md").read_text()
