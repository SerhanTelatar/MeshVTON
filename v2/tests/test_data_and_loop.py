"""Dataset sözleşme okuyucuları + eğitim döngüsü testleri (saf torch, lokal)."""

import numpy as np
import pytest
import torch

from meshvton2.conditioning.builder import GarmentAsset
from meshvton2.data.datasets import BundleDataset, MixedDataset, SingleViewDataset
from meshvton2.data.items import discover_synth_items, load_item
from meshvton2.synth.generate import VIEWS, generate
from meshvton2.training.loop import TrainConfig, TrainLoop, cosine_warmup

SIZE = (64, 48)


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    """Stub builder'la küçük sentetik set — dataset'ler bunu okur."""
    root = tmp_path_factory.mktemp("synth")
    rng = np.random.RandomState(3)
    assets = [
        GarmentAsset(
            garment_id=f"g{i}", verts=rng.randn(20, 3).astype(np.float32),
            faces=rng.randint(0, 20, (30, 3)).astype(np.int64),
            uv=rng.rand(20, 2).astype(np.float32),
            texture=rng.randint(0, 255, (8, 8, 3), dtype=np.uint8),
        )
        for i in range(2)
    ]
    stats = generate(assets, root, num_samples=4, size=SIZE, seed=5, log=lambda *_: None)
    assert stats["written"] == 4
    return root


def test_discover_and_load(synth_root):
    items = discover_synth_items(synth_root)
    assert len(items) == 4 * len(VIEWS)
    d = load_item(items[0], size=SIZE)
    assert d["gt_rgb"].shape == (3, *SIZE) and d["inpaint_mask"].shape == (1, *SIZE)
    assert d["gt_rgb"].min() >= -1.001 and d["gt_rgb"].max() <= 1.001
    assert set(torch.unique(d["inpaint_mask"]).tolist()) <= {0.0, 1.0}


def test_single_view_dataset(synth_root):
    ds = SingleViewDataset(discover_synth_items(synth_root), size=SIZE)
    assert len(ds) == 16
    assert ds[3]["control_normal"].shape == (3, *SIZE)


def test_bundle_dataset(synth_root):
    items = discover_synth_items(synth_root)
    groups = [items[i : i + 4] for i in range(0, len(items), 4)]
    ds = BundleDataset(groups, size=SIZE)
    b = ds[0]
    assert b["gt_rgb"].shape == (4, 3, *SIZE)  # (V,C,H,W)
    assert b["appearance_ref"].shape == (4, 3, *SIZE)
    # aynı örneğin görüşleri AYNI referansı paylaşır
    assert torch.equal(b["appearance_ref"][0], b["appearance_ref"][3])


def test_mixed_dataset_ratio_and_determinism():
    a = [{"src": "a"}] * 100
    b = [{"src": "b"}] * 100

    class L(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    m = MixedDataset(L(a), L(b), primary_ratio=0.7, length=2000, seed=1)
    srcs = [m[i]["src"] for i in range(2000)]
    ratio = srcs.count("a") / len(srcs)
    assert 0.65 < ratio < 0.75
    assert srcs == [m[i]["src"] for i in range(2000)]  # deterministik


def test_cosine_warmup_shape():
    assert cosine_warmup(0, 10, 100) == 0.0
    assert cosine_warmup(10, 10, 100) == pytest.approx(1.0)
    assert cosine_warmup(100, 10, 100) == pytest.approx(0.0, abs=1e-9)
    assert cosine_warmup(55, 10, 100) > cosine_warmup(90, 10, 100)


def test_train_loop_learns_and_resumes(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    target_w = torch.tensor([[1.0, -2.0, 0.5, 3.0]])

    def step_fn(batch):
        x = batch["x"]
        return ((model(x) - x @ target_w.T) ** 2).mean()

    loader = [{"x": torch.randn(8, 4)} for _ in range(10)]
    cfg = TrainConfig(max_steps=60, lr=0.05, warmup_steps=5, ckpt_every=30,
                      log_every=1000, out_dir=str(tmp_path))
    loop = TrainLoop(
        model.parameters(), step_fn, cfg,
        state_provider=lambda: {"w": model.state_dict()},
        state_loader=lambda s: model.load_state_dict(s["w"]),
        log=lambda *_: None,
    )
    first_loss = float(step_fn(loader[0]))
    loop.run(loader)
    assert float(step_fn(loader[0])) < first_loss * 0.5  # öğreniyor
    assert (tmp_path / "ckpt_000060.pt").exists() and (tmp_path / "latest.pt").exists()

    # resume: yeni loop, latest'ten devam — adım sayısı ve ağırlıklar geri gelir
    model2 = torch.nn.Linear(4, 1)
    loop2 = TrainLoop(
        model2.parameters(), step_fn, cfg,
        state_provider=lambda: {"w": model2.state_dict()},
        state_loader=lambda s: model2.load_state_dict(s["w"]),
        log=lambda *_: None,
    )
    loop2.maybe_resume()
    assert loop2.step == 60
    assert torch.equal(model2.weight, model.weight)


def test_train_loop_rejects_frozen_params():
    lin = torch.nn.Linear(2, 2)
    for p in lin.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        TrainLoop(lin.parameters(), lambda b: torch.tensor(0.0),
                  TrainConfig(max_steps=1), log=lambda *_: None)


def test_latents_fast_path(synth_root):
    """precompute latent'leri varsa dataset VAE'siz sözlük döner; yoksa net hata."""
    items = discover_synth_items(synth_root)
    it = items[0]
    with pytest.raises(FileNotFoundError, match="precompute"):
        ds_missing = SingleViewDataset(items, size=SIZE, use_latents=True)
        _ = ds_missing[0]

    fake = {k: torch.randn(16, 8, 6, dtype=torch.bfloat16)
            for k in ("gt_lat", "masked_lat", "normal_lat", "depth_sil_lat", "ref_lat")}
    torch.save(fake, it.item_dir / "latents.pt")
    d = SingleViewDataset([it], size=SIZE, use_latents=True)[0]
    assert set(fake) <= set(d) and d["inpaint_mask"].shape == (1, *SIZE)
    assert torch.equal(d["gt_lat"], fake["gt_lat"])
    (it.item_dir / "latents.pt").unlink()  # fixture'ı diğer testler için temiz bırak


def test_checkpoint_rotation(tmp_path):
    """100 adımda bir ckpt + rotasyon: yalnız son keep_last kalır (Drive kotası)."""
    model = torch.nn.Linear(2, 1)
    loader = [{"x": torch.randn(4, 2)}] * 5
    cfg = TrainConfig(max_steps=50, ckpt_every=10, keep_last=2, log_every=1000,
                      out_dir=str(tmp_path))
    loop = TrainLoop(model.parameters(),
                     lambda b: (model(b["x"]) ** 2).mean(), cfg,
                     state_provider=lambda: {"w": model.state_dict()},
                     state_loader=lambda s: model.load_state_dict(s["w"]),
                     log=lambda *_: None)
    loop.run(loader)
    cks = sorted(p.name for p in tmp_path.glob("ckpt_*.pt"))
    assert cks == ["ckpt_000040.pt", "ckpt_000050.pt"]  # 10..30 silindi
    assert (tmp_path / "latest.pt").read_text().endswith("ckpt_000050.pt")
