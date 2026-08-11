"""Garment drape: K-NN binding to the body surface + local frame transfer.

v1's `_geometric_align` only did centre+scale (no pose/shape tracking).
The v2 approach: garment vertices are bound to the K nearest vertices of the REST body;
each bond's offset is stored in the body vertex's local frame (normal + tangent).
When the body deforms under any (β,θ) the frames are rebuilt and the offsets are
re-applied — this carries both the β blend shape and the LBS pose automatically (no
per-joint transform matrix needed; the vertices smplx returns are enough).

Flow:
    binding = bind_garment(garment_rest, body_rest_verts, body_faces)   # once, cached
    draped  = apply_binding(binding, body_posed_verts)                  # for each (β,θ)
    draped  = push_clearance(draped, body_posed_verts, body_faces)      # resolve interpenetration

Known limits (deliberate, per the plan): no drape physics for loose skirts/dresses
(the Phase 6 Blender upgrade); the initial garment set is upper body. A normal-agreement
filter on the arms (rejecting dot<0 neighbours) reduces body-grabbing errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_NEIGHBORS = 4


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (V,3), unit length."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # weighted by 2*area
    n = np.zeros_like(verts)
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def _tangent_partners(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """A deterministic neighbour vertex for each vertex (tangent direction reference).
    Because it comes from the topology it rotates WITH the body — tying the frame to a
    global axis would break the drape under rigid rotation (rotation-equivariance requirement)."""
    partner = np.full(n_verts, -1, dtype=np.int64)
    for col in range(3):
        src = faces[:, col]
        dst = faces[:, (col + 1) % 3]
        empty = partner[src] == -1
        # many faces write to the same vertex; with repeated np assignment the last one wins — deterministic
        partner[src[empty]] = dst[empty]
    partner[partner == -1] = np.arange(n_verts)[partner == -1]  # isolated vertex (should not happen)
    return partner


def _frames(verts: np.ndarray, normals: np.ndarray, partners: np.ndarray) -> np.ndarray:
    """An orthonormal frame per vertex (V,3,3): columns [t1, t2, n].
    Tangent = the projection of the edge to the neighbour vertex onto the tangent plane (mesh-intrinsic)."""
    n = normals
    edge = verts[partners] - verts
    t1 = edge - (edge * n).sum(1, keepdims=True) * n
    bad = np.linalg.norm(t1, axis=1) < 1e-10  # edge parallel to the normal (degenerate)
    if bad.any():
        ref = np.tile(np.array([1.0, 0.0, 0.0]), (bad.sum(), 1))
        ref[np.abs(n[bad, 0]) > 0.9] = [0.0, 0.0, 1.0]
        t1[bad] = ref - (ref * n[bad]).sum(1, keepdims=True) * n[bad]
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=2)  # (V,3,3)


def _knn(query: np.ndarray, ref: np.ndarray, k: int, chunk: int = 2048):
    """Brute-force K-NN (no scipy). query (Q,3), ref (R,3) -> idx (Q,k), dist (Q,k)."""
    idx = np.empty((len(query), k), dtype=np.int64)
    dist = np.empty((len(query), k), dtype=np.float64)
    for s in range(0, len(query), chunk):
        q = query[s : s + chunk]
        d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)  # (q,R)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        pd = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(pd, axis=1)
        idx[s : s + chunk] = np.take_along_axis(part, order, axis=1)
        dist[s : s + chunk] = np.sqrt(np.take_along_axis(pd, order, axis=1))
    return idx, dist


# --------------------------------------------------------------------------- #
# Binding and application
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GarmentBinding:
    nn_idx: np.ndarray        # (Vg,K) body vertex indices
    weights: np.ndarray       # (Vg,K) 1/d² normalized
    local_offsets: np.ndarray  # (Vg,K,3) offset in the rest frame
    body_faces: np.ndarray    # (Fb,3) for the normal computations

    def save(self, path: str) -> None:
        """Atomic write (tmp+rename): parallel workers share the same cache;
        a half-written .npz was raising 'File is not a zip file'."""
        import os

        tmp = f"{path}.tmp.{os.getpid()}.npz"
        np.savez_compressed(
            tmp, nn_idx=self.nn_idx, weights=self.weights,
            local_offsets=self.local_offsets, body_faces=self.body_faces,
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "GarmentBinding":
        d = np.load(path)
        return cls(d["nn_idx"], d["weights"], d["local_offsets"], d["body_faces"])


def bind_garment(
    garment_verts: np.ndarray,
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    *,
    k: int = K_NEIGHBORS,
    normal_filter: bool = True,
) -> GarmentBinding:
    """Binds the garment to the rest body. normal_filter: penalizes neighbours facing
    away from the garment vertex's approximate normal (so an arm vertex does not grab the torso)."""
    g = np.asarray(garment_verts, np.float64)
    b = np.asarray(body_verts, np.float64)
    n_body = vertex_normals(b, body_faces)

    kk = min(k * 3, len(b)) if normal_filter else k  # wide candidate pool, filtered afterwards
    idx, dist = _knn(g, b, kk)

    if normal_filter:
        # garment vertex normal approximation: the direction of the nearest body normal is the reference
        approx_n = n_body[idx[:, 0]]
        agree = (n_body[idx] * approx_n[:, None, :]).sum(-1)  # (Vg,kk)
        penalized = np.where(agree > 0.0, dist, np.inf)  # drop neighbours facing the other way
        order = np.argsort(penalized, axis=1)[:, :k]
        idx = np.take_along_axis(idx, order, axis=1)
        dist = np.take_along_axis(penalized, order, axis=1)
        # for a vertex whose candidates were all dropped, fall back to the true unfiltered distance
        fallback = np.linalg.norm(g[:, None] - b[idx], axis=-1)
        dist = np.where(np.isfinite(dist), dist, fallback)

    w = 1.0 / np.maximum(dist, 1e-9) ** 2
    w /= w.sum(axis=1, keepdims=True)

    partners = _tangent_partners(np.asarray(body_faces, np.int64), len(b))
    frames = _frames(b, n_body, partners)  # (Vb,3,3)
    offsets_world = g[:, None, :] - b[idx]                       # (Vg,K,3)
    f = frames[idx]                                              # (Vg,K,3,3)
    local = np.einsum("vkij,vkj->vki", f.transpose(0, 1, 3, 2), offsets_world)
    return GarmentBinding(idx.astype(np.int64), w, local, np.asarray(body_faces, np.int64))


def apply_binding(binding: GarmentBinding, body_verts_posed: np.ndarray) -> np.ndarray:
    """Applies the binding to a deformed body -> (Vg,3) draped garment."""
    b = np.asarray(body_verts_posed, np.float64)
    n_body = vertex_normals(b, binding.body_faces)
    partners = _tangent_partners(binding.body_faces, len(b))
    frames = _frames(b, n_body, partners)
    f = frames[binding.nn_idx]                                   # (Vg,K,3,3)
    off = np.einsum("vkij,vkj->vki", f, binding.local_offsets)   # frame to world
    pts = b[binding.nn_idx] + off                                # (Vg,K,3)
    return (pts * binding.weights[..., None]).sum(axis=1)


def push_clearance(
    garment_verts: np.ndarray,
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    *,
    clearance: float = 0.008,
) -> tuple[np.ndarray, float, float]:
    """Pushes garment vertices that sank into the body out along the nearest body
    normal to the clearance distance.

    Returns: (corrected verts, pushed fraction, mean penetration depth [m]).
    NOTE: some pushing is NORMAL on a fitted garment (cloth touches skin) — the
    corruption signal is not HOW MANY vertices were pushed but how DEEPLY (mm=normal,
    cm=the garment is passing through the body).
    clearance 4mm→8mm (2026-08-09): the push is VERTEX-based only, so even with every
    vertex outside, the body can leak THROUGH large triangles (skin-colored holes on the
    shoulder/chest in QA). Cloth already sits a few mm off the skin."""
    g = np.asarray(garment_verts, np.float64).copy()
    b = np.asarray(body_verts, np.float64)
    n_body = vertex_normals(b, body_faces)
    idx, _ = _knn(g, b, 1)
    nb, nn = b[idx[:, 0]], n_body[idx[:, 0]]
    signed = ((g - nb) * nn).sum(1)  # signed distance along the normal
    inside = signed < clearance
    depth = float(np.clip(clearance - signed[inside], 0, None).mean()) if inside.any() else 0.0
    g[inside] += (clearance - signed[inside])[:, None] * nn[inside]
    return g, float(inside.mean()), depth
