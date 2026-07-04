"""Faz 4 çekirdek testleri: flow matching, zero-init kontrol embedder, token paketleme.
Hepsi saf torch — GPU/FLUX gerektirmez."""

import pytest
import torch

from meshvton2.model.control_embedder import ControlXEmbedder, attach_control_embedder
from meshvton2.model.reference_tokens import (
    concat_reference,
    make_img_ids,
    pack_latents,
    unpack_latents,
)
from meshvton2.training.flow_matching import (
    apply_shift,
    resolution_shift,
    rf_interpolate,
    rf_loss,
    rf_target,
    sample_logit_normal_t,
    x0_from_v,
)

# ------------------------------ flow matching ------------------------------ #


def test_rf_endpoints_and_recovery():
    x0, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    assert torch.equal(rf_interpolate(x0, noise, torch.zeros(2)), x0)
    assert torch.equal(rf_interpolate(x0, noise, torch.ones(2)), noise)
    t = torch.tensor([0.3, 0.8])
    x_t = rf_interpolate(x0, noise, t)
    assert torch.allclose(x0_from_v(x_t, rf_target(x0, noise), t), x0, atol=1e-6)


def test_rf_loss_zero_at_perfect_prediction():
    x0, noise = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    assert rf_loss(rf_target(x0, noise), x0, noise).item() == pytest.approx(0.0)
    assert rf_loss(torch.zeros_like(x0), x0, noise).item() > 0


def test_rf_loss_token_mask_excludes_reference():
    """Referans token'larındaki hata kayba girmemeli."""
    x0, noise = torch.randn(1, 6, 4), torch.randn(1, 6, 4)
    v = rf_target(x0, noise).clone()
    v[:, 3:] += 100.0  # son 3 token (referans) tamamen yanlış
    mask = torch.tensor([[True, True, True, False, False, False]])
    assert rf_loss(v, x0, noise, token_mask=mask).item() == pytest.approx(0.0, abs=1e-10)
    assert rf_loss(v, x0, noise).item() > 1.0  # masksız olsa cezalanırdı


def test_logit_normal_and_shift():
    g = torch.Generator().manual_seed(0)
    t = sample_logit_normal_t(4096, generator=g)
    assert ((t > 0) & (t < 1)).all()
    assert 0.4 < t.mean().item() < 0.6  # mean=0 → medyan 0.5
    s = resolution_shift(3072)
    assert s > 1.0  # uzun dizi → gürültüye kaydır
    ts = apply_shift(t, s)
    assert (ts >= t - 1e-6).all()  # shift t'yi 1'e yaklaştırır
    assert torch.allclose(apply_shift(t, 1.0), t)  # s=1 kimlik


# --------------------------- kontrol embedder --------------------------- #


def _embedder():
    orig = torch.nn.Linear(384, 64)
    return ControlXEmbedder(orig, control_in_features=128), orig


def test_zero_init_is_identity():
    """v1 dersinin kilidi: init'te kontrollü çıktı, stok çıktıyla BİT-EŞ."""
    emb, orig = _embedder()
    x = torch.randn(2, 10, 384)
    ctrl = torch.randn(2, 10, 128) * 100  # ne olursa olsun etkisiz olmalı
    assert torch.equal(emb(torch.cat([x, ctrl], -1)), orig(x))
    assert torch.equal(emb(x), orig(x))  # kontrolsüz çağrı da stok


def test_only_control_proj_trains():
    emb, _ = _embedder()
    x = torch.randn(2, 5, 384 + 128)
    emb(x).sum().backward()
    assert emb.control_proj.weight.grad is not None
    assert emb.original.weight.grad is None and not emb.original.weight.requires_grad
    names = [n for n, p in emb.named_parameters() if p.requires_grad]
    assert names == ["control_proj.weight"]


def test_embedder_rejects_wrong_width():
    emb, _ = _embedder()
    with pytest.raises(ValueError):
        emb(torch.randn(1, 4, 400))


def test_sidecar_roundtrip(tmp_path):
    emb, _ = _embedder()
    with torch.no_grad():
        emb.control_proj.weight += 0.5
    p = str(tmp_path / "ctrl.pt")
    emb.save_sidecar(p)
    emb2, _ = _embedder()
    emb2.load_sidecar(p)
    assert torch.equal(emb.control_proj.weight, emb2.control_proj.weight)


def test_attach_idempotent():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = torch.nn.Linear(384, 64)

    tr = Fake()
    a = attach_control_embedder(tr, 128)
    b = attach_control_embedder(tr, 128)
    assert a is b and isinstance(tr.x_embedder, ControlXEmbedder)


# --------------------------- referans token'ları --------------------------- #


def test_pack_unpack_roundtrip():
    x = torch.randn(2, 16, 12, 10)
    tokens = pack_latents(x)
    assert tokens.shape == (2, 6 * 5, 64)
    assert torch.equal(unpack_latents(tokens, 12, 10), x)
    with pytest.raises(ValueError):
        pack_latents(torch.randn(1, 4, 7, 8))  # tek H


def test_img_ids_and_reference_concat():
    tids = make_img_ids(12, 10, frame_idx=0)
    rids = make_img_ids(12, 10, frame_idx=1)
    assert tids.shape == (30, 3)
    assert (tids[:, 0] == 0).all() and (rids[:, 0] == 1).all()
    assert tids[:, 1].max() == 5 and tids[:, 2].max() == 4

    tgt, ref = torch.randn(2, 30, 64), torch.randn(2, 30, 64)
    tokens, ids, mask = concat_reference(tgt, tids, ref, rids)
    assert tokens.shape == (2, 60, 64) and ids.shape == (60, 3)
    assert mask[:30].all() and not mask[30:].any()
    assert torch.equal(tokens[:, :30], tgt) and torch.equal(tokens[:, 30:], ref)
