"""Giysi drape'i: gövde yüzeyine K-NN bağlama + yerel çerçeve transferi.

v1'in `_geometric_align`'ı yalnız merkez+ölçek yapıyordu (poz/şekil takibi yok).
v2 yaklaşımı: giysi vertex'leri REST gövdesinin en yakın K vertex'ine bağlanır;
her bağda offset, gövde vertex'inin yerel çerçevesinde (normal + teğet) saklanır.
Gövde herhangi bir (β,θ) ile deforme olunca çerçeveler yeniden kurulur ve offset
geri uygulanır — β blend-shape'i de LBS pozunu da otomatik taşır (per-joint
transform matrisi gerektirmez; smplx'in verdiği vertex'ler yeter).

Akış:
    binding = bind_garment(garment_rest, body_rest_verts, body_faces)   # 1 kez, cache'lenir
    draped  = apply_binding(binding, body_posed_verts)                  # her (β,θ) için
    draped  = push_clearance(draped, body_posed_verts, body_faces)      # iç içe girmeyi çöz

Bilinen sınırlar (plan gereği bilinçli): bol etek/elbisede sarkma fiziği yok
(Faz 6 Blender yükseltmesi); başlangıç giysi seti üst-giyim. Kollarda normal
uyumu filtresi (dot<0 komşu reddi) gövde-kapma hatasını azaltır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_NEIGHBORS = 4


# --------------------------------------------------------------------------- #
# Geometri yardımcıları
# --------------------------------------------------------------------------- #


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Alan-ağırlıklı vertex normalleri (V,3), birim uzunluk."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # alan*2 ağırlıklı
    n = np.zeros_like(verts)
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def _tangent_partners(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """Her vertex için deterministik bir komşu vertex (teğet yönü referansı).
    Topolojiden geldiği için gövdeyle BİRLİKTE döner — çerçeveyi global eksene
    bağlamak rijit rotasyonda drape'i bozar (rotasyon-eşdeğişimi şartı)."""
    partner = np.full(n_verts, -1, dtype=np.int64)
    for col in range(3):
        src = faces[:, col]
        dst = faces[:, (col + 1) % 3]
        empty = partner[src] == -1
        # aynı vertex'e çok yüz yazar; np tekrarlı atamada sonuncu kazanır — deterministik
        partner[src[empty]] = dst[empty]
    partner[partner == -1] = np.arange(n_verts)[partner == -1]  # izole vertex (olmamalı)
    return partner


def _frames(verts: np.ndarray, normals: np.ndarray, partners: np.ndarray) -> np.ndarray:
    """Her vertex için ortonormal çerçeve (V,3,3): sütunlar [t1, t2, n].
    Teğet = komşu vertex'e giden kenarın teğet düzleme izdüşümü (mesh-intrinsik)."""
    n = normals
    edge = verts[partners] - verts
    t1 = edge - (edge * n).sum(1, keepdims=True) * n
    bad = np.linalg.norm(t1, axis=1) < 1e-10  # kenar normale paralel (dejenere)
    if bad.any():
        ref = np.tile(np.array([1.0, 0.0, 0.0]), (bad.sum(), 1))
        ref[np.abs(n[bad, 0]) > 0.9] = [0.0, 0.0, 1.0]
        t1[bad] = ref - (ref * n[bad]).sum(1, keepdims=True) * n[bad]
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=2)  # (V,3,3)


def _knn(query: np.ndarray, ref: np.ndarray, k: int, chunk: int = 2048):
    """Kaba-kuvvet K-NN (scipy'siz). query (Q,3), ref (R,3) -> idx (Q,k), dist (Q,k)."""
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
# Bağlama ve uygulama
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GarmentBinding:
    nn_idx: np.ndarray        # (Vg,K) gövde vertex indeksleri
    weights: np.ndarray       # (Vg,K) 1/d² normalize
    local_offsets: np.ndarray  # (Vg,K,3) rest çerçevesinde offset
    body_faces: np.ndarray    # (Fb,3) normal hesapları için

    def save(self, path: str) -> None:
        """Atomik yazım (tmp+rename): paralel işçiler aynı cache'i paylaşır;
        yarım yazılmış .npz 'File is not a zip file' hatası veriyordu."""
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
    """Giysiyi rest gövdesine bağlar. normal_filter: giysi vertex'inin yaklaşık
    normali ile zıt bakan komşuları cezalandırır (kol vertex'i gövde kapmasın)."""
    g = np.asarray(garment_verts, np.float64)
    b = np.asarray(body_verts, np.float64)
    n_body = vertex_normals(b, body_faces)

    kk = min(k * 3, len(b)) if normal_filter else k  # aday havuzu geniş, sonra filtre
    idx, dist = _knn(g, b, kk)

    if normal_filter:
        # giysi vertex normal yaklaşımı: en yakın gövde normalinin yönü referans
        approx_n = n_body[idx[:, 0]]
        agree = (n_body[idx] * approx_n[:, None, :]).sum(-1)  # (Vg,kk)
        penalized = np.where(agree > 0.0, dist, np.inf)  # zıt bakan komşuyu ele
        order = np.argsort(penalized, axis=1)[:, :k]
        idx = np.take_along_axis(idx, order, axis=1)
        dist = np.take_along_axis(penalized, order, axis=1)
        # tüm adayları elenen vertex için filtresiz gerçek mesafeye geri dön
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
    """Bağlamayı deforme gövdeye uygular -> (Vg,3) drape edilmiş giysi."""
    b = np.asarray(body_verts_posed, np.float64)
    n_body = vertex_normals(b, binding.body_faces)
    partners = _tangent_partners(binding.body_faces, len(b))
    frames = _frames(b, n_body, partners)
    f = frames[binding.nn_idx]                                   # (Vg,K,3,3)
    off = np.einsum("vkij,vkj->vki", f, binding.local_offsets)   # çerçeveden dünyaya
    pts = b[binding.nn_idx] + off                                # (Vg,K,3)
    return (pts * binding.weights[..., None]).sum(axis=1)


def push_clearance(
    garment_verts: np.ndarray,
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    *,
    clearance: float = 0.004,
) -> tuple[np.ndarray, float, float]:
    """Gövde içine giren giysi vertex'lerini en yakın gövde normali boyunca
    clearance mesafesine iter.

    Döner: (düzeltilmiş verts, itilen oran, ortalama penetrasyon derinliği [m]).
    NOT: oturan giyside bir miktar itme NORMALDİR (kumaş tene değer) — bozukluk
    sinyali itilen SAYI değil, ne kadar DERİNDEN itildiğidir (mm=normal, cm=
    giysi gövdenin içinden geçiyor demek)."""
    g = np.asarray(garment_verts, np.float64).copy()
    b = np.asarray(body_verts, np.float64)
    n_body = vertex_normals(b, body_faces)
    idx, _ = _knn(g, b, 1)
    nb, nn = b[idx[:, 0]], n_body[idx[:, 0]]
    signed = ((g - nb) * nn).sum(1)  # normal yönünde işaretli mesafe
    inside = signed < clearance
    depth = float(np.clip(clearance - signed[inside], 0, None).mean()) if inside.any() else 0.0
    g[inside] += (clearance - signed[inside])[:, None] * nn[inside]
    return g, float(inside.mean()), depth
