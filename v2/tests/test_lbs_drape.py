"""lbs_drape.py tests: a synthetic cylinder body + garment ring (no SMPL-X required).

The visual drape QA in the Phase 2 exit criteria (real CLOTH3D + SMPL-X) happens on Colab;
these tests lock in the invariants of the binding/application maths.
"""

import numpy as np
import pytest

from meshvton2.conditioning.lbs_drape import (
    GarmentBinding,
    apply_binding,
    bind_garment,
    push_clearance,
    vertex_normals,
)


def cylinder(radius=1.0, height=2.0, n_seg=24, n_rows=8):
    """A simple open cylinder mesh (a stand-in for the body)."""
    verts, faces = [], []
    for r in range(n_rows):
        y = height * r / (n_rows - 1) - height / 2
        for s in range(n_seg):
            a = 2 * np.pi * s / n_seg
            verts.append([radius * np.cos(a), y, radius * np.sin(a)])
    for r in range(n_rows - 1):
        for s in range(n_seg):
            a, b = r * n_seg + s, r * n_seg + (s + 1) % n_seg
            c, d = a + n_seg, b + n_seg
            faces += [[a, c, b], [b, c, d]]  # outward-facing winding (CCW)
    return np.array(verts, np.float64), np.array(faces, np.int64)


BODY_V, BODY_F = cylinder()
GARMENT = cylinder(radius=1.05, height=1.0, n_rows=4)[0]  # a ring wrapping the body


def test_vertex_normals_point_outward():
    n = vertex_normals(BODY_V, BODY_F)
    radial = BODY_V.copy()
    radial[:, 1] = 0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    assert ((n * radial).sum(1) > 0.9).all()  # on a cylinder the normals point radially outward


def test_rest_pose_roundtrip():
    """If the body did not deform, the garment comes back unchanged."""
    binding = bind_garment(GARMENT, BODY_V, BODY_F)
    out = apply_binding(binding, BODY_V)
    assert np.abs(out - GARMENT).max() < 1e-9


def test_translation_follows():
    binding = bind_garment(GARMENT, BODY_V, BODY_F)
    t = np.array([0.3, -0.1, 0.5])
    out = apply_binding(binding, BODY_V + t)
    assert np.abs(out - (GARMENT + t)).max() < 1e-9


def test_rigid_rotation_follows():
    """If the body rotates rigidly, the garment follows with the same rigid rotation."""
    binding = bind_garment(GARMENT, BODY_V, BODY_F)
    a = np.deg2rad(40)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    out = apply_binding(binding, BODY_V @ R.T)
    assert np.abs(out - GARMENT @ R.T).max() < 1e-6


def test_radial_scale_follows():
    """If the body inflates (a stand-in for a beta change), the garment is pushed outward."""
    binding = bind_garment(GARMENT, BODY_V, BODY_F)
    fat = BODY_V.copy()
    fat[:, [0, 2]] *= 1.2
    out = apply_binding(binding, fat)
    r_out = np.linalg.norm(out[:, [0, 2]], axis=1)
    assert (r_out > np.linalg.norm(GARMENT[:, [0, 2]], axis=1) + 0.1).all()


def test_push_clearance():
    """A garment vertex sunk inside is pushed out to the clearance distance; the depth is reported."""
    sunk = GARMENT.copy()
    sunk[:, [0, 2]] *= 0.9 / 1.05  # pull the radius INSIDE the body (~0.15 depth)
    fixed, ratio, depth = push_clearance(sunk, BODY_V, BODY_F, clearance=0.02)
    assert ratio > 0.9  # almost all of them were corrected
    assert 0.05 < depth < 0.3  # mean penetration ~0.14m (a deep-sinking signal)
    n = vertex_normals(BODY_V, BODY_F)
    from meshvton2.conditioning.lbs_drape import _knn

    idx, _ = _knn(fixed, BODY_V, 1)
    signed = ((fixed - BODY_V[idx[:, 0]]) * n[idx[:, 0]]).sum(1)
    assert (signed > 0.019).all()
    # vertices that are outside are left untouched, the depth is ~0
    _, ratio0, depth0 = push_clearance(GARMENT, BODY_V, BODY_F, clearance=0.02)
    assert ratio0 < 0.05 and depth0 < 0.01


def test_binding_save_load_roundtrip(tmp_path):
    binding = bind_garment(GARMENT, BODY_V, BODY_F)
    p = tmp_path / "g.lbs.npz"
    binding.save(str(p))
    loaded = GarmentBinding.load(str(p))
    out_a = apply_binding(binding, BODY_V * 1.1)
    out_b = apply_binding(loaded, BODY_V * 1.1)
    assert np.array_equal(out_a, out_b)


def test_weights_normalized():
    binding = bind_garment(GARMENT, BODY_V, BODY_F)
    assert np.allclose(binding.weights.sum(1), 1.0)
    assert (binding.weights >= 0).all()
