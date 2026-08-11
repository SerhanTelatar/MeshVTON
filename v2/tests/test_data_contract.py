"""Synthetic data directory contract test (runs locally with the stub builder;
on Colab the same test runs against the real pipeline — conftest picks MESHVTON2_STUB automatically)."""

import csv
import json

import numpy as np

from meshvton2.conditioning.builder import GarmentAsset
from meshvton2.synth.generate import VIEWS, generate

SIZE = (64, 48)


def _dummy_assets(n=2):
    rng = np.random.RandomState(7)
    return [
        GarmentAsset(
            garment_id=f"upper_body__g{i}",
            verts=rng.randn(30, 3).astype(np.float32),
            faces=rng.randint(0, 30, (40, 3)).astype(np.int64),
            uv=rng.rand(30, 2).astype(np.float32),
            texture=rng.randint(0, 255, (16, 16, 3), dtype=np.uint8),
        )
        for i in range(n)
    ]


def test_generate_contract(tmp_path):
    stats = generate(_dummy_assets(), tmp_path, num_samples=3, size=SIZE, seed=1, log=lambda *_: None)
    assert stats["written"] == 3 and not stats["failed"]

    with (tmp_path / "pairs.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["sample_id", "garment_id"]
    assert len(rows) == 4  # header + 3

    for sample_id, garment_id in rows[1:]:
        sd = tmp_path / sample_id
        assert (sd / "appearance_ref.png").exists()
        meta = json.loads((sd / "meta.json").read_text())
        assert meta["garment_id"] == garment_id
        assert len(meta["betas"]) == 10 and len(meta["body_pose"]) == 63
        assert set(meta["views"]) == {str(v) for v in VIEWS}
        for v in VIEWS:
            vd = sd / f"view_{v:03d}"
            for f in ("gt", "agnostic", "mask", "normal", "depth_sil"):
                assert (vd / f"{f}.png").exists(), f"{sample_id}/view_{v:03d}/{f}.png missing"
        # the view cameras must differ from one another (the orbit really rotates)
        views = meta["views"]
        assert views["0"] != views["180"]


def test_generate_appends_pairs(tmp_path):
    generate(_dummy_assets(1), tmp_path, num_samples=1, size=SIZE, seed=1, log=lambda *_: None)
    generate(_dummy_assets(1), tmp_path, num_samples=1, size=SIZE, seed=2, log=lambda *_: None)
    with (tmp_path / "pairs.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 3  # a single header + 2 samples (different seed → different sample_id)
    assert rows[1][0] != rows[2][0]
