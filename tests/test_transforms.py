"""Tests for core.transforms.

Conventions under test (OpenGL, as in the gluPerspective / gluLookAt manual pages):
- matrices multiply a column vector on their right,
- the camera looks down its own negative z axis,
- clip-space depth after the perspective divide runs from -1 (near) to +1 (far).
"""

import numpy as np
import pytest

from physengine.core.transforms import look_at, pers_proj


def apply(m, p):
    """Apply a 4x4 matrix to a 3-vector (homogeneous w = 1), return the 4-vector."""
    return m @ np.append(np.asarray(p, dtype=float), 1.0)


def apply_divide(m, p):
    """Apply and perform the perspective divide, return the 3-vector."""
    v = apply(m, p)
    return v[:3] / v[3]


# --- look_at -----------------------------------------------------------------


def test_look_at_maps_eye_to_origin():
    m = look_at(eye=[3.0, -2.0, 5.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    assert np.allclose(apply(m, [3.0, -2.0, 5.0]), [0.0, 0.0, 0.0, 1.0])


def test_look_at_maps_target_onto_negative_z_axis():
    eye = np.array([3.0, -2.0, 5.0])
    target = np.array([1.0, 1.0, -1.0])
    m = look_at(eye, target, up=[0.0, 1.0, 0.0])
    dist = np.linalg.norm(target - eye)
    assert np.allclose(apply(m, target), [0.0, 0.0, -dist, 1.0])


def test_look_at_is_rigid():
    """The upper-left 3x3 block must be a proper rotation: orthonormal, determinant +1."""
    m = look_at(eye=[1.0, 2.0, 3.0], target=[-4.0, 0.5, 2.0], up=[0.0, 0.0, 1.0])
    r = m[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(m[3], [0.0, 0.0, 0.0, 1.0])


def test_look_at_up_direction_maps_to_positive_y():
    """A point directly above the eye (along `up`) must land at positive view-space y."""
    eye = [0.0, 0.0, 5.0]
    m = look_at(eye, target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    v = apply(m, [0.0, 1.0, 5.0])
    assert v[1] > 0.0
    assert np.isclose(v[0], 0.0) and np.isclose(v[2], 0.0)


def test_look_at_right_handed():
    """Camera at +z looking at the origin with y up: world +x must appear on the right (+x)."""
    m = look_at(eye=[0.0, 0.0, 5.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    v = apply(m, [1.0, 0.0, 0.0])
    assert v[0] > 0.0


# --- pers_proj ---------------------------------------------------------------


def test_perspective_matches_gluPerspective_reference():
    """Explicit reference: fov = 90 deg, aspect = 1, near = 1, far = 3.

    From the gluPerspective manual page, with f = 1 / tan(45 deg) = 1:
      [1 0  0   0]
      [0 1  0   0]
      [0 0 -2  -3]   (far+near)/(near-far) = 4/-2 = -2 ; 2*far*near/(near-far) = 6/-2 = -3
      [0 0 -1   0]
    """
    m = pers_proj(np.pi / 2, 1.0, 1.0, 3.0)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, -3.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    assert np.allclose(m, expected)


@pytest.mark.parametrize("near,far", [(0.1, 100.0), (1.0, 3.0), (0.01, 1000.0)])
def test_perspective_near_and_far_map_to_clip_minus_one_and_plus_one(near, far):
    m = pers_proj(np.radians(60.0), 16.0 / 9.0, near, far)
    assert np.isclose(apply_divide(m, [0.0, 0.0, -near])[2], -1.0)
    assert np.isclose(apply_divide(m, [0.0, 0.0, -far])[2], 1.0)


def test_perspective_point_straight_ahead_has_zero_xy():
    m = pers_proj(np.radians(45.0), 1.5, 0.1, 50.0)
    xyz = apply_divide(m, [0.0, 0.0, -7.0])
    assert np.allclose(xyz[:2], [0.0, 0.0])


def test_perspective_frustum_edge_maps_to_clip_boundary():
    """At depth d, a point at height d*tan(fov/2) lies on the top edge -> clip y = +1,
    and a point at x = aspect*d*tan(fov/2) lies on the right edge -> clip x = +1."""
    fov, aspect, d = np.radians(70.0), 1.25, 4.0
    m = pers_proj(fov, aspect, 0.1, 100.0)
    half_h = d * np.tan(fov / 2)
    assert np.isclose(apply_divide(m, [0.0, half_h, -d])[1], 1.0)
    assert np.isclose(apply_divide(m, [aspect * half_h, 0.0, -d])[0], 1.0)


def test_perspective_depth_is_monotonic():
    """Farther points must get larger clip z (needed for depth testing to work)."""
    m = pers_proj(np.radians(60.0), 1.0, 0.1, 100.0)
    depths = [0.2, 1.0, 5.0, 20.0, 90.0]
    clip_z = [apply_divide(m, [0.0, 0.0, -d])[2] for d in depths]
    assert np.all(np.diff(clip_z) > 0.0)


# --- combined ----------------------------------------------------------------


def test_view_projection_puts_target_at_screen_centre():
    eye = np.array([2.0, 3.0, 6.0])
    target = np.array([-1.0, 0.5, 0.0])
    view = look_at(eye, target, up=[0.0, 1.0, 0.0])
    proj = pers_proj(np.radians(50.0), 16.0 / 9.0, 0.1, 100.0)
    xyz = apply_divide(proj @ view, target)
    assert np.allclose(xyz[:2], [0.0, 0.0])
    assert -1.0 < xyz[2] < 1.0
