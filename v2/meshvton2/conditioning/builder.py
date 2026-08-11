"""build_conditioning() — the SINGLE source of conditioning generation (parity contract).

v1's most expensive lesson: when training preprocessing and inference conditioning came
from different code paths, the ControlNet got out-of-distribution input and ruined results.
v2 rule: training preprocessing (scripts/preprocess_vitonhd.py), the synthetic generator
(synth/generate.py, person_image=None mode) and inference (inference/run_tryon.py)
call build_conditioning() from this module WITHOUT EXCEPTION. tests/test_parity.py
calls it from both paths with the same input and enforces tensor equality.

This signature was frozen in Phase 0; adding fields is allowed, changing existing ones is not.

IMPLEMENTATION CHOICE: the real path (SMPL-X + LBS drape + pyrender) is the default.
The `MESHVTON2_STUB=1` env var selects the deterministic stub — ONLY for tests on a
dev machine without 3D dependencies (tests/conftest.py sets it automatically).
Production scripts call `assert_real_impl()`: v1's "silent placeholder" disaster is
structurally prevented.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

CANONICAL_SIZE = (1024, 768)  # (height, width) — same as configs/base.yaml

# Hang pad [m]: vertical offset of the garment's top edge relative to the shoulder
# (upper body) / pelvis (lower body) reference.
#
# +0.06 IS THE CORRECT VALUE, DO NOT CHANGE. Rationale: SMPL j[16]/j[17] is the shoulder
# JOINT centre, not the top of the shoulder (~6cm below) — a t-shirt collar/shoulder seam
# only sits on the shoulder with +6cm. The 2026-07-04 synthetic drape QA passed with this value.
#
# On 2026-08-09 -0.06 was briefly tried (based on the photo calibration in
# [[meshvton-v2-inference-alignment]]) and in the QA render the garment slid to the ARMPIT
# (the t-shirt looked like a strapless bustier) → reverted. That -0.06 was a patch
# compensating for an uncorrected person_square_bbox bug; the bbox is now correct on every path, so it is unnecessary.
DEFAULT_HANG_PAD = 0.06

# SEPARATE value for the PHOTO path. Empirical calibration on 2026-08-10
# (v2/scripts/calibrate_hang.py, the 5 people that passed the camera gate): +0.06 hangs the
# garment ~21 points HIGHER than the real garment (it rides over the face);
# mesh<->parser IoU is 0.295 at +0.06 and 0.480 at -0.12. Verified end to end:
# placement IoU (placement_iou.py) 0.5815 -> 0.6829, mesh specificity unchanged
# (+0.0236 -> +0.0219, same 10 combos). geo_iou DROPPED, but that metric targets the
# silhouette we hand it, so it cannot see a broken target — see placement_iou.py.
#
# The SYNTHETIC path STAYS at +0.06: there both body and garment come from the same
# SMPL-X and the 2026-07-04 drape QA passed with that value; when -0.06 was tried the
# garment slid into the armpit. The two paths need different values because HMR2 (photo)
# and SMPL-X (synthetic) bodies differ; no single global value fits both.
PHOTO_HANG_PAD = -0.12

# Garment scale for the PHOTO path. 2026-08-10 calibration (calibrate_scale.py, the same 5
# verified people, hang_pad=-0.12): mesh<->parser IoU is 0.481 at 1.00, peaking at 0.555
# at 1.25. The peak is NOT at the boundary (1.10-1.70 swept, it drops after 1.30) and it
# passes an independent consistency check: at 1.25 the silhouette covers 19.2% of the frame,
# the real garment area measured by the parser is 18.8% — the same point from two paths.
#
# Interpretation: CLOTH3D garments are systematically SMALL relative to the bodies HMR2
# estimates (synthetic/real body distribution mismatch). The SYNTHETIC path STAYS at 1.0 —
# there both body and garment come from the same SMPL-X, and rescaling turned out to be a
# mistake twice in the past (see the _prealign_garment lesson chain).
PHOTO_GARMENT_SCALE = 1.25

# use_texture: defaults to False per the PERMANENT RULE — the appearance ref is always flat
# grey (color/pattern fidelity is not a goal of this project). Its ONLY legitimate use: running
# the OLD checkpoint trained on 2026-07-06 with textured refs + a VITON-HD mix (stage1_july/
# ckpt_004000.pt) — those weights are used to seeing textured refs and would get OUT-of-
# distribution input from a grey ref. NEVER set True for new/textureless checkpoints.


def implementation() -> str:
    """"real" | "stub" — read from the env on every call (for test isolation)."""
    return "stub" if os.environ.get("MESHVTON2_STUB") == "1" else "real"


def assert_real_impl() -> None:
    """Production entry points (synth generator, preprocessing, run_tryon) call this."""
    if implementation() != "real":
        raise RuntimeError(
            "build_conditioning is in STUB mode (MESHVTON2_STUB=1) — forbidden in production. "
            "This is the antidote to v1's 'an untrained placeholder ran silently' bug."
        )


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraSpec:
    """Serializable camera: full-frame intrinsics + world->camera transform."""

    K: tuple  # 3x3, nested tuple
    R: tuple  # 3x3
    T: tuple  # 3
    source: str  # "photo" | "orbit:90" | "synth" ...

    def to_dict(self) -> dict:
        return {"K": self.K, "R": self.R, "T": self.T, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSpec":
        as_t = lambda x: tuple(tuple(r) if isinstance(r, (list, tuple)) else r for r in x)
        return cls(K=as_t(d["K"]), R=as_t(d["R"]), T=tuple(d["T"]), source=d["source"])


@dataclass(frozen=True)
class PhotoView:
    """The photo's own camera (derived from HMR2 pred_cam)."""


@dataclass(frozen=True)
class OrbitView:
    """The photo camera rotated around the pelvis vertical axis."""

    azim_deg: int  # 0/90/180/270 (0 = same direction as the photo camera)


ViewSpec = PhotoView | OrbitView


@dataclass(frozen=True)
class GarmentAsset:
    """Loaded garment asset. Filled by garment.py::load_garment_asset in Phase 2."""

    garment_id: str
    verts: np.ndarray          # (V,3) float32
    faces: np.ndarray          # (F,3) int64
    uv: np.ndarray | None      # (V,2) float32
    texture: np.ndarray | None  # (Ht,Wt,3) uint8
    lbs_cache: str | None = None  # path to garment_id.lbs.npz (Phase 2)


@dataclass(frozen=True)
class ConditioningBundle:
    """Output of build_conditioning. All tensors CPU float32, (C,H,W)."""

    agnostic_rgb: torch.Tensor      # (3,H,W) [-1,1]
    inpaint_mask: torch.Tensor      # (1,H,W) {0,1}
    control_normal: torch.Tensor    # (3,H,W) [-1,1] camera-space scene normals (body+garment)
    control_depth_sil: torch.Tensor  # (3,H,W) [-1,1]: [depth, depth, garment silhouette]
    appearance_ref: torch.Tensor    # (3,H,W) [-1,1] SHADED grey cloth render (NEVER textured)
    camera: CameraSpec
    meta: dict = field(default_factory=dict)  # contains meta["gt_rgb"] in synth mode

    def __post_init__(self):
        h, w = self.inpaint_mask.shape[-2:]
        for name in ("agnostic_rgb", "control_normal", "control_depth_sil", "appearance_ref"):
            t = getattr(self, name)
            if t.shape != (3, h, w):
                raise ValueError(f"{name}: expected (3,{h},{w}), got {tuple(t.shape)}")
            if t.dtype != torch.float32:
                raise ValueError(f"{name}: must be float32, got {t.dtype}")
        if self.inpaint_mask.shape != (1, h, w):
            raise ValueError(f"inpaint_mask: expected (1,{h},{w}), got {tuple(self.inpaint_mask.shape)}")
        uniq = torch.unique(self.inpaint_mask)
        if not torch.all((uniq == 0) | (uniq == 1)):
            raise ValueError("inpaint_mask must be binary {0,1}")


# --------------------------------------------------------------------------- #
# Single-source function
# --------------------------------------------------------------------------- #


def build_conditioning(
    person_image: np.ndarray | None,
    smplx_params: dict[str, Any],
    garment: GarmentAsset | None,
    view: ViewSpec,
    *,
    size: tuple[int, int] = CANONICAL_SIZE,
    device: str = "cpu",
    person_prep: Any | None = None,  # PersonPrep — REQUIRED in photo mode (agnostic+mask source)
    appearance_ref_image: np.ndarray | None = None,  # REQUIRED in garment=None mode (product photo)
    geometry_mask: bool = True,  # photo+mesh: mask = parse ∪ dilate(garment silhouette); False = parse only
    hang_pad: float = DEFAULT_HANG_PAD,  # garment top-edge hang pad [m] (see DEFAULT_HANG_PAD)
    garment_scale: float = 1.0,  # 1.0 = CLOTH3D metric scale (see _prealign_garment)
    use_texture: bool = False,  # False BY RULE — see the use_texture note above
) -> ConditioningBundle:
    """Builds the conditioning bundle.

    Args:
        person_image: (H,W,3) uint8 RGB photo; None => synthetic mode
            (no real photo, the GT render is returned as meta["gt_rgb"]).
        smplx_params: betas(10), body_pose(63), global_orient(3), transl(3),
            pred_cam(3: s,tx,ty), bbox(4: x,y,w,h) — HMR2 adapter contract.
        garment: loaded garment asset (with its LBS cache) OR None = "real-data
            mode": there is no mesh for the garment the person is WEARING (VITON-HD
            training) → geometry is body-only, the garment silhouette comes from the
            parse, the appearance reference from appearance_ref_image (product photo).
            Valid only in photo mode; required for supervision consistency (the GT garment cannot be a random mesh).
        view: PhotoView() = the photo's camera; OrbitView(azim) = rotated.
        size: (height, width); the only valid value is CANONICAL_SIZE, tests may use a
            small size.

    Returns:
        ConditioningBundle — identical for training, synthetic generation and inference.
    """
    if person_image is not None:
        person_image = np.ascontiguousarray(person_image)
        if person_image.ndim != 3 or person_image.shape[2] != 3 or person_image.dtype != np.uint8:
            raise ValueError("person_image must be (H,W,3) uint8 RGB")
    required = {"betas", "body_pose", "global_orient", "pred_cam", "bbox"}
    missing = required - set(smplx_params)
    if missing:
        raise ValueError(f"smplx_params missing fields: {sorted(missing)} (hmr2_adapter must return pred_cam+bbox)")
    if not isinstance(view, (PhotoView, OrbitView)):
        raise TypeError(f"view must be PhotoView|OrbitView, got {type(view)}")
    if garment is None:
        if person_image is None:
            raise ValueError("Synthetic mode (person_image=None) cannot be built without a garment mesh")
        if appearance_ref_image is None:
            raise ValueError("appearance_ref_image (product photo) is required in garment=None mode")

    if implementation() == "stub":
        return _build_impl_stub(
            person_image, smplx_params, garment, view,
            size=size, device=device, appearance_ref_image=appearance_ref_image,
        )
    return _build_impl_real(
        person_image, smplx_params, garment, view, size=size, device=device,
        person_prep=person_prep, appearance_ref_image=appearance_ref_image,
        geometry_mask=geometry_mask, hang_pad=hang_pad, garment_scale=garment_scale,
        use_texture=use_texture,
    )


# --------------------------------------------------------------------------- #
# Real implementation (Phase 2): SMPL-X + LBS drape + screen-space render
# --------------------------------------------------------------------------- #


def _to_tensor01(img01: np.ndarray) -> torch.Tensor:
    """(H,W,3) [0,1] float -> (3,H,W) float32 [-1,1]."""
    return torch.from_numpy(np.ascontiguousarray(img01.transpose(2, 0, 1))).float() * 2.0 - 1.0


def _prealign_garment(gverts: np.ndarray, rest: dict, garment_id: str = "",
                      hang_pad: float = DEFAULT_HANG_PAD, scale: float = 1.0) -> np.ndarray:
    """Pre-binding alignment: NO SCALING, only axis conversion + hang alignment.

    Lesson chain (caught in QA): (1) a body-width scale measured the T-pose arm
    span and inflated the garment 3x; (2) a shoulder-heuristic scale shrank an
    already CORRECTLY scaled CLOTH3D garment and pushed it inside the body
    (clearance ~0.9). Reality: CLOTH3D garments are in metre scale and modelled
    against the SMPL body — rescaling was a mistake in every case. Now: convert
    Z-up→Y-up, centre x/z on the pelvis, align vertically by 'hang': tops/dresses
    hang from the shoulder (top edge ≈ shoulder line + pad), lower body from the waist (top edge ≈ pelvis + pad)."""
    from meshvton2.conditioning.render import zup_to_yup

    j = rest["joints"]
    pelvis = j[0]
    mid_sh = (j[16] + j[17]) / 2.0

    g = zup_to_yup(np.asarray(gverts, np.float64))  # metric scale is PRESERVED
    # The scale=1.0 default preserves the "rescaling was ALWAYS a mistake" lesson above.
    # A value other than 1.0 is ONLY for empirical calibration
    # (v2/scripts/calibrate_scale.py): on the photo path the body comes from HMR2, so
    # CLOTH3D's SMPL-referenced scale may not fit exactly. The scale is applied around the
    # garment's OWN centre; the centring + hang alignment that follows is unaffected.
    if scale != 1.0:
        c = (g.max(axis=0) + g.min(axis=0)) / 2.0
        g = (g - c) * scale + c
    g[:, 0] += pelvis[0] - (g[:, 0].max() + g[:, 0].min()) / 2.0
    g[:, 2] += pelvis[2] - (g[:, 2].max() + g[:, 2].min()) / 2.0
    # Alignment uses the garment's TOP EDGE (max y) — the collar rim on a t-shirt, the
    # chest band on a strapless top. A single global offset cannot be ideal for every garment
    # type; the default is chosen for the t-shirt/top family (see DEFAULT_HANG_PAD).
    hang_y = (pelvis[1] + hang_pad) if "lower_body" in garment_id else (mid_sh[1] + hang_pad)
    g[:, 1] += hang_y - g[:, 1].max()  # rest body is y-UP: top edge = max(y)
    return g


def _get_binding(garment: GarmentAsset, body_model, hang_pad: float = DEFAULT_HANG_PAD,
                 scale: float = 1.0) -> "GarmentBinding":  # noqa: F821
    """Loads the garment binding from cache, or builds it on the rest body and caches it.
    The cache key INCLUDES hang_pad AND scale — a different hang/scale is a different binding
    (the cache used to only apply at +0.06; when the default changed it silently turned off;
    without scale in the key a calibration sweep would read the first scale's cache)."""
    from meshvton2.conditioning.lbs_drape import GarmentBinding, bind_garment

    cache_path = None
    if garment.lbs_cache:
        p = Path(garment.lbs_cache)
        key = f".hp{int(round(hang_pad * 1000)):+04d}"
        if scale != 1.0:
            key += f".sc{int(round(scale * 1000)):04d}"
        cache_path = p.with_name(f"{p.stem}{key}{p.suffix}")
        if cache_path.exists():
            try:
                return GarmentBinding.load(cache_path)
            except Exception:  # half-written cache from a parallel worker race — rebuild
                pass
    rest = body_model.rest()
    aligned = _prealign_garment(garment.verts, rest, garment.garment_id,
                                hang_pad=hang_pad, scale=scale)
    binding = bind_garment(aligned, rest["verts"], rest["faces"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        binding.save(cache_path)
    return binding


ATR_GARMENT_LABELS = (4, 7)  # upper_clothes, dress — garment silhouette from the parse (garment=None mode)


def _build_impl_real(
    person_image, smplx_params, garment, view, *, size, device,
    person_prep=None, appearance_ref_image=None, geometry_mask=True, hang_pad=DEFAULT_HANG_PAD,
    garment_scale=1.0, use_texture=False,
):
    import cv2

    from meshvton2.conditioning import camera as cam_mod
    from meshvton2.conditioning.body import get_body_model
    from meshvton2.conditioning.lbs_drape import apply_binding, push_clearance
    from meshvton2.conditioning.render import (
        force_textureless,
        render_appearance_ref,
        render_geometry,
        render_textured_scene,
    )

    if person_image is not None and person_prep is None:
        raise ValueError(
            "person_prep is required in photo mode (output of PersonPreprocessor.process) — "
            "agnostic+mask come from there; the builder does not load a parser."
        )
    hgt, wdt = size

    # 1) Body + camera (NO azimuth estimation: the photo camera comes from pred_cam, the rest are orbits)
    body_model = get_body_model()
    body = body_model(smplx_params)
    cam0 = cam_mod.photo_camera(smplx_params, size)
    cam = cam0 if isinstance(view, PhotoView) else cam_mod.orbit_camera(cam0, body["pelvis"], view.azim_deg)

    meta: dict[str, Any] = {"view": "photo" if isinstance(view, PhotoView) else f"orbit:{view.azim_deg}"}

    if garment is not None:
        # 2) Drape: rest binding (cached) -> apply to the posed body -> clearance
        binding = _get_binding(garment, body_model, hang_pad=hang_pad, scale=garment_scale)
        gverts = apply_binding(binding, body["verts"])
        gverts, clearance_ratio, pen_depth = push_clearance(gverts, body["verts"], body["faces"])
        # Explosion detector: how much bigger is the draped garment than the body?
        # (clearance CANNOT catch this — an exploded garment is outside the body;
        # QA showed a shattered garment 3-4x the size, this metric is its gate)
        diag = lambda v: float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
        extent_ratio = diag(gverts) / max(diag(body["verts"]), 1e-8)

        # 3) Screen-space geometry + appearance reference (mesh render)
        geo = render_geometry(body["verts"], body["faces"], gverts, garment.faces, cam, size=size)
        garment_sil = geo["garment_sil"].astype(np.float32)
        # PERMANENT RULE: the system runs TEXTURELESS — the appearance ref is ALWAYS a
        # SHAPED render with flat grey cloth (not a flat grey card: the garment's form
        # reaches the model, color/pattern information never does), whether or not
        # garment.texture exists (see project rule: appearance fidelity is not a goal).
        # use_texture=True is ONLY for the old textured checkpoint (see the constant's note)
        g_app = garment if use_texture else force_textureless(garment)
        appearance01 = render_appearance_ref(g_app, size=size).astype(np.float32) / 255.0
        meta.update(garment_id=garment.garment_id, clearance_ratio=clearance_ratio,
                    penetration_depth=pen_depth, drape_extent_ratio=extent_ratio)
    else:
        # Real-data mode: body-only geometry; the garment silhouette comes from the PARSE,
        # the appearance reference from the product photo (supervision consistency).
        geo = render_geometry(body["verts"], body["faces"], None, None, cam, size=size)
        parse = np.asarray(person_prep.parse)
        parse = cv2.resize(parse, (wdt, hgt), interpolation=cv2.INTER_NEAREST)
        garment_sil = np.isin(parse, ATR_GARMENT_LABELS).astype(np.float32)
        ref = cv2.resize(appearance_ref_image, (wdt, hgt), interpolation=cv2.INTER_AREA)
        appearance01 = ref.astype(np.float32) / 255.0
        meta.update(garment_id="real_worn_garment", clearance_ratio=None)

    depth_sil01 = np.stack([geo["depth"], geo["depth"], garment_sil], axis=2)

    # 4) Agnostic + mask
    if person_image is None:  # synthetic mode: derive from the GT render
        skin = np.full((len(body["verts"]), 3), (0.80, 0.62, 0.52))
        # The GT garment is ALWAYS grey (the SAME look as the appearance ref).
        # 2026-08-09 diagnosis: with a textured GT the ref stayed grey → the same input
        # sometimes got a patterned, sometimes a grey target; since the pattern is
        # UNPREDICTABLE from the input, the flow-matching loss optimum is the conditional
        # MEAN = a pale/semi-transparent smear. The task only becomes learnable once the target is grey too.
        gt = render_textured_scene(body["verts"], body["faces"], skin, gverts,
                                   g_app, cam, size=size)
        kernel = np.ones((wdt // 30, wdt // 30), np.uint8)
        mask_u8 = cv2.dilate((geo["garment_sil"] * 255).astype(np.uint8), kernel)
        agnostic = gt.copy()
        agnostic[mask_u8 > 127] = (128, 128, 128)
        meta["gt_rgb"] = _to_tensor01(gt.astype(np.float32) / 255.0)
    else:
        agnostic = person_prep.agnostic
        mask_u8 = person_prep.mask
        if agnostic.shape[:2] != (hgt, wdt):
            raise ValueError(f"person_prep size {agnostic.shape[:2]} != target {(hgt, wdt)}")
        if garment is not None and geometry_mask:
            # ALIGNMENT FIX: the parse mask marks the person's OLD garment; the mesh
            # silhouette may be wider/shifted → the model gets trapped inside the mask
            # and leaves a "ghost" of the geometry outside. Mask = parse ∪ dilate(sil)
            # (the synthetic training mask is dilate(sil) too — same regime, see above).
            from meshvton2.conditioning.person import apply_agnostic

            kernel = np.ones((wdt // 30, wdt // 30), np.uint8)
            sil_u8 = cv2.dilate((geo["garment_sil"] * 255).astype(np.uint8), kernel)
            mask_u8 = np.maximum(mask_u8, sil_u8)
            agnostic = apply_agnostic(person_prep.image, mask_u8)
            meta["mask_source"] = "parse+garment_sil"

    return ConditioningBundle(
        agnostic_rgb=_to_tensor01(agnostic.astype(np.float32) / 255.0),
        inpaint_mask=torch.from_numpy((mask_u8 > 127).astype(np.float32)).unsqueeze(0),
        control_normal=_to_tensor01(geo["normal"]),
        control_depth_sil=_to_tensor01(depth_sil01),
        appearance_ref=_to_tensor01(appearance01),
        camera=cam,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Stub implementation (only for the 3D-dependency-free test environment; see assert_real_impl)
# --------------------------------------------------------------------------- #


def _stable_seed(person_image, smplx_params, garment: GarmentAsset | None, view: ViewSpec, size, appearance_ref_image=None) -> int:
    """Deterministic seed from all inputs — the basis of the parity test."""
    h = hashlib.sha256()
    h.update(b"none" if person_image is None else person_image.tobytes())
    for key in sorted(k for k in smplx_params if k in ("betas", "body_pose", "global_orient", "transl", "pred_cam", "bbox")):
        h.update(key.encode())
        h.update(np.asarray(smplx_params[key], dtype=np.float64).tobytes())
    if garment is not None:
        h.update(garment.garment_id.encode())
        h.update(garment.verts.astype(np.float64).tobytes())
    else:
        h.update(b"nogarment")
        h.update(np.ascontiguousarray(appearance_ref_image).tobytes())
    view_tag = "photo" if isinstance(view, PhotoView) else f"orbit:{view.azim_deg}"
    h.update(view_tag.encode())
    h.update(json.dumps(size).encode())
    return int.from_bytes(h.digest()[:8], "little")


def _build_impl_stub(person_image, smplx_params, garment, view, *, size, device, appearance_ref_image=None) -> ConditioningBundle:
    hgt, wdt = size
    gen = torch.Generator().manual_seed(
        _stable_seed(person_image, smplx_params, garment, view, size, appearance_ref_image)
    )

    def rand_img() -> torch.Tensor:
        return (torch.rand(3, hgt, wdt, generator=gen) * 2 - 1).float()

    mask = (torch.rand(1, hgt, wdt, generator=gen) > 0.7).float()
    view_tag = "photo" if isinstance(view, PhotoView) else f"orbit:{view.azim_deg}"
    camera = CameraSpec(
        K=tuple(map(tuple, np.eye(3).tolist())),
        R=tuple(map(tuple, np.eye(3).tolist())),
        T=(0.0, 0.0, 2.7),
        source=view_tag if person_image is not None else f"synth:{view_tag}",
    )
    meta: dict[str, Any] = {"stub": True, "garment_id": garment.garment_id if garment else "real_worn_garment"}
    if person_image is None:
        meta["gt_rgb"] = rand_img()
    return ConditioningBundle(
        agnostic_rgb=rand_img(),
        inpaint_mask=mask,
        control_normal=rand_img(),
        control_depth_sil=rand_img(),
        appearance_ref=rand_img(),
        camera=camera,
        meta=meta,
    )
