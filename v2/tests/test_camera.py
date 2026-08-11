"""camera.py maths tests (pure numpy — no GPU/data required).

Note: the reprojection-IoU test that forms the Phase 2 exit criterion (SMPL-X silhouette vs
the parser mask, IoU >= 0.70) needs real data and runs on Colab; the ones here lock in the
geometric invariants of the transform.
"""

import numpy as np
import pytest

from meshvton2.conditioning.builder import CameraSpec
from meshvton2.conditioning.camera import (
    camera_center,
    orbit_camera,
    photo_camera,
    project,
    weak_persp_to_full,
)

SIZE = (1024, 768)  # (H,W)


def _params(s=0.9, tx=0.0, ty=0.0, bbox=(0, 0, 768, 1024)):
    return {
        "pred_cam": np.array([s, tx, ty], np.float32),
        "bbox": np.array(bbox, np.float32),
    }


def test_weak_persp_geometry():
    t, focal = weak_persp_to_full(np.array([0.9, 0.0, 0.0]), np.array([0, 0, 768, 1024]), SIZE)
    assert t[2] > 0 and focal > 0
    # the person is bigger in frame (large s) -> the camera is closer (small tz)
    t2, _ = weak_persp_to_full(np.array([1.8, 0.0, 0.0]), np.array([0, 0, 768, 1024]), SIZE)
    assert t2[2] < t[2]
    # invalid input fails fast
    with pytest.raises(ValueError):
        weak_persp_to_full(np.array([0.0, 0, 0]), np.array([0, 0, 768, 1024]), SIZE)


def test_photo_camera_centered_person_projects_to_center():
    """A centred bbox + tx=ty=0 -> the body centre (the origin) lands at the image centre."""
    cam = photo_camera(_params(), SIZE)
    uv = project(cam, np.zeros((1, 3)))
    assert np.allclose(uv[0], [SIZE[1] / 2, SIZE[0] / 2], atol=1e-6)
    assert cam.source == "photo"


def test_offcenter_bbox_shifts_projection():
    cam_l = photo_camera(_params(bbox=(0, 0, 384, 1024)), SIZE)     # person on the left
    cam_c = photo_camera(_params(), SIZE)
    u_l = project(cam_l, np.zeros((1, 3)))[0, 0]
    u_c = project(cam_c, np.zeros((1, 3)))[0, 0]
    assert u_l < u_c  # left bbox -> projection on the left


def test_orbit_identity_and_period():
    cam = photo_camera(_params(), SIZE)
    pivot = np.array([0.0, 0.2, 0.1])
    for deg in (0.0, 360.0):
        o = orbit_camera(cam, pivot, deg)
        _, R0, T0 = (np.asarray(x) for x in (cam.K, cam.R, cam.T))
        _, R1, T1 = (np.asarray(x) for x in (o.K, o.R, o.T))
        assert np.allclose(R0, R1, atol=1e-12) and np.allclose(T0, T1, atol=1e-9)


def test_orbit_preserves_pivot_distance_and_projection():
    """The pivot lands on the same pixel at every orbit angle, the camera-pivot distance is constant."""
    cam = photo_camera(_params(), SIZE)
    pivot = np.array([0.05, 0.3, -0.02])
    d0 = np.linalg.norm(camera_center(cam) - pivot)
    uv0 = project(cam, pivot[None])
    for deg in (90, 180, 270):
        o = orbit_camera(cam, pivot, deg)
        assert np.linalg.norm(camera_center(o) - pivot) == pytest.approx(d0, abs=1e-9)
        assert np.allclose(project(o, pivot[None]), uv0, atol=1e-6)
        assert o.source == f"orbit:{deg}"


def test_orbit_180_views_opposite_side():
    """A point in front of the pivot lands behind it at 180° (the depth order flips)."""
    cam = photo_camera(_params(), SIZE)
    pivot = np.zeros(3)
    front = np.array([[0.0, 0.0, -0.2]])  # offset towards the camera
    _, R, T = (np.asarray(x) for x in (cam.K, cam.R, cam.T))
    z_front = (front @ R.T + T)[0, 2]
    o = orbit_camera(cam, pivot, 180)
    _, R2, T2 = (np.asarray(x) for x in (o.K, o.R, o.T))
    z_front_back = (front @ R2.T + T2)[0, 2]
    z_pivot = T[2]
    assert z_front < z_pivot            # in front: closer than the pivot
    assert z_front_back > z_pivot       # at 180°: farther than the pivot


def test_orbit_90_270_symmetric():
    """A point in front of the pivot lands on opposite sides of the centre at 90° and 270°
    (the symmetry is exact because it stays at the same depth)."""
    cam = photo_camera(_params(), SIZE)
    pivot = np.zeros(3)
    front = np.array([[0.0, 0.0, -0.3]])  # the camera side of the pivot
    u90 = project(orbit_camera(cam, pivot, 90), front)[0, 0]
    u270 = project(orbit_camera(cam, pivot, 270), front)[0, 0]
    cx = SIZE[1] / 2
    assert abs(u90 - cx) > 1.0  # clearly away from the centre
    assert (u90 - cx) == pytest.approx(-(u270 - cx), abs=1e-6)


def test_project_behind_camera_is_nan():
    cam = photo_camera(_params(), SIZE)
    _, _, T = (np.asarray(x) for x in (cam.K, cam.R, cam.T))
    behind = np.array([[0.0, 0.0, -(T[2] + 1.0)]])
    assert np.isnan(project(cam, behind)).all()


def test_camera_spec_serializable():
    cam = photo_camera(_params(), SIZE)
    assert CameraSpec.from_dict(cam.to_dict()) == cam


def test_detect_person_bbox_is_square_and_centered():
    """HMR2 expects a square crop — a rectangular bbox broke the aspect ratio and shifted the camera."""
    from meshvton2.conditioning.body import detect_person_bbox

    bbox = detect_person_bbox(np.zeros((1024, 768, 3), np.uint8))
    x0, y0, x1, y1 = bbox
    assert (x1 - x0) == (y1 - y0) == 1024  # square, side = max(H,W)
    assert (x0 + x1) / 2 == pytest.approx(384) and (y0 + y1) / 2 == pytest.approx(512)
