"""Tests of the black-hole ray tracers (GLSL, run in a standalone GL context).

Units G = c = M = 1. Reference values and their sources:

- Shadow of a Schwarzschild hole seen by a static observer at r:
  sin(alpha) = sqrt(27) M sqrt(1 - 2M/r) / r (Synge 1966, MNRAS 131, 463).
- Final directions of light rays: exact quadrature of
  dphi/du = 1 / sqrt(1/b^2 - u^2 + 2u^3) (schwarzschild.escape_angle).
- Face-on disc: every ray reaches the disc plane after a quarter turn
  (phi = pi/2); the photons have no angular momentum about the axis, so
  g = sqrt(1 - 3M/r) / sqrt(1 - 2M/r_cam) (schwarzschild.disc_redshift).
- Kerr shadow seen from far away: the critical curve of Bardeen (1973), in
  kerr.shadow_edge. Kerr with a -> 0 must reproduce Schwarzschild.
- A body on a Kerr circular orbit where a disc pixel's ray lands must show
  up in that pixel with that pixel's redshift (two independent codes).
"""

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from dataclasses import replace  # noqa: E402

from physengine.modules import kerr  # noqa: E402
from physengine.modules.blackhole_renderer import BlackHoleRenderer, TraceSettings  # noqa: E402
from physengine.modules.kerr_bodies import (  # noqa: E402
    KerrBodySystem,
    kerr_circular_state,
    kerr_images,
)
from physengine.modules.schwarzschild import escape_angle, shadow_angular_radius  # noqa: E402


@pytest.fixture(scope="module")
def renderer():
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as error:
        pytest.skip(f"no OpenGL 4.3 context: {error}")
    r = BlackHoleRenderer()
    r.setup(ctx)
    yield r
    r.release()
    ctx.release()


def frame(e, up=(0, 1, 0)):
    e = np.array(e, float)
    f = -e / np.linalg.norm(e)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return e, np.stack([r, u, -f], 1)


def pixel_dirs(rot, fov, aspect, w, h):
    x = (np.arange(w) + 0.5) / w * 2 - 1
    y = (np.arange(h) + 0.5) / h * 2 - 1
    X, Y = np.meshgrid(x, y)
    t = np.tan(fov / 2)
    d = np.stack([X * aspect * t, Y * t, -np.ones_like(X)], -1) @ rot.T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


@pytest.mark.parametrize("r0", [4.0, 30.0, 100.0])
def test_shadow_radius(renderer, r0):
    a = shadow_angular_radius(r0)
    fov = min(2.6, 2.4 * a)
    w = 2001
    e, rot = frame((0.0, 0.0, r0))
    renderer.trace(
        (w, 1), e, rot, fov, 1.0, TraceSettings(show_disc=False, step=0.03), output_mode=1
    )
    row = renderer.read()[0]
    ang = np.arctan(np.abs(((np.arange(w) + 0.5) / w * 2 - 1) * np.tan(fov / 2)))
    captured = row[:, 3] == 0
    pixel = ang[w // 2 + 1] - ang[w // 2]
    assert abs(ang[captured].max() - a) < 2 * pixel


def test_escape_directions(renderer):
    r0, fov, n = 30.0, 1.0, 121
    e, rot = frame(np.array([0.0, 0.3, 1.0]) / np.linalg.norm([0.0, 0.3, 1.0]) * r0)
    renderer.trace((n, n), e, rot, fov, 1.0, TraceSettings(show_disc=False), output_mode=1)
    img = renderer.read()
    d = pixel_dirs(rot, fov, 1.0, n, n)
    ch = e / r0
    worst = 0.0
    for j in range(0, n, 8):
        for i in range(0, n, 8):
            dd = d[j, i]
            alpha = np.arccos(np.clip(dd @ ch, -1, 1))
            ref = escape_angle(r0, alpha)
            got = img[j, i]
            if ref is None:
                assert got[3] == 0.0
                continue
            assert got[3] == 1.0
            et = dd - (dd @ ch) * ch
            et /= np.linalg.norm(et)
            want = np.cos(ref) * ch + np.sin(ref) * et
            g3 = got[:3] / np.linalg.norm(got[:3])
            worst = max(
                worst, np.degrees(np.arctan2(np.linalg.norm(np.cross(want, g3)), want @ g3))
            )
    assert worst < 0.01


def test_face_on_disc_redshift(renderer):
    r0 = 40.0
    e, rot = frame((0.0, r0, 0.0), up=(0, 0, 1))
    renderer.trace((101, 101), e, rot, 1.4, 1.0, TraceSettings(r_out=30.0), output_mode=1)
    img = renderer.read()
    disc = img[..., 3] == 2.0
    assert disc.sum() > 100
    r, g = img[..., 0][disc], img[..., 1][disc]
    assert np.allclose(g, np.sqrt(1 - 3 / r) / np.sqrt(1 - 2 / r0), rtol=1e-5)


def test_doppler_side(renderer):
    # Camera on +z; the disc spins about +y, so its gas at -x (left) comes towards us.
    e, rot = frame((0.0, 0.5, 50.0))
    renderer.trace((201, 201), e, rot, 0.9, 1.0, TraceSettings(r_out=25.0), output_mode=1)
    img = renderer.read()
    disc = img[..., 3] == 2.0
    g = img[..., 1]
    assert g[:, :100][disc[:, :100]].max() > 1.2 > 0.6 > g[:, 101:][disc[:, 101:]].min()


@pytest.mark.parametrize("pitch", [0.1, 0.06])  # multiples of the step 0.02
def test_mirror_symmetry_when_the_disc_crossing_is_on_a_step_boundary(renderer, pitch):
    # Regression: the crossing angle used to pick the next crossing, half a
    # turn later, when it fell exactly on a step boundary (a vertical seam).
    e, rot = frame(np.array([0.0, np.sin(pitch), np.cos(pitch)]) * 30.0)
    renderer.trace((300, 200), e, rot, 1.0, 1.5, TraceSettings(step=0.02), output_mode=1)
    img = renderer.read()
    disc = (img[..., 3] == 2) & (img[..., 3][:, ::-1] == 2)
    r = img[..., 0]
    assert np.abs(r - r[:, ::-1])[disc].max() < 0.02


def test_kerr_with_tiny_spin_is_schwarzschild(renderer):
    e, rot = frame(np.array([0.0, np.sin(0.3), np.cos(0.3)]) * 30.0)
    st0 = TraceSettings(r_out=25.0)
    renderer.trace((151, 101), e, rot, 1.0, 1.5, st0, output_mode=1)
    A = renderer.read().copy()
    renderer.trace(
        (151, 101), e, rot, 1.0, 1.5, replace(st0, spin=1e-4, r_in=kerr.isco(1e-4)), output_mode=1
    )
    B = renderer.read().copy()
    assert np.mean(A[..., 3] == B[..., 3]) > 0.995
    disc = (A[..., 3] == 2) & (B[..., 3] == 2) & (B[..., 1] > 0)
    assert np.median(np.abs(A[..., 0] - B[..., 0])[disc]) < 0.02
    assert np.median(np.abs(A[..., 1][disc] / B[..., 1][disc] - 1)) < 1e-3


@pytest.mark.parametrize("a, incl_deg", [(0.9, 90.0), (0.5, 90.0), (0.9, 60.0)])
def test_kerr_shadow_matches_bardeen(renderer, a, incl_deg):
    incl, r0, W = np.radians(incl_deg), 1000.0, 601
    e, rot = frame(np.array([0.0, np.cos(incl), np.sin(incl)]) * r0)
    fov = 2 * np.arctan(9.0 / r0)
    st = TraceSettings(show_disc=False, spin=a, r_in=kerr.isco(a), step=0.01, max_steps=20000)
    renderer.trace((W, W), e, rot, fov, 1.0, st, output_mode=1)
    captured = renderer.read()[..., 3] == 0
    xs = ((np.arange(W) + 0.5) / W * 2 - 1) * np.tan(fov / 2) * r0
    alpha_b, beta_b = kerr.shadow_edge(a, incl, 4000)
    for beta in (2.0, 3.5):
        row = np.argmin(np.abs(xs - beta))
        cols = np.nonzero(captured[row])[0]
        sel = np.abs(beta_b - beta) < 0.03
        # Finite distance (r0 = 1000) shrinks the shadow by ~0.5%, plus a pixel.
        assert xs[cols.min()] == pytest.approx(alpha_b[sel].min(), abs=0.15)
        assert xs[cols.max()] == pytest.approx(alpha_b[sel].max(), abs=0.15)


def test_kerr_body_lands_in_its_disc_pixel(renderer):
    a, r0, W, fov = 0.9, 30.0, 151, 1.0
    e, rot = frame(np.array([0.0, np.sin(0.3), np.cos(0.3)]) * r0)
    renderer.trace(
        (W, W),
        e,
        rot,
        fov,
        1.0,
        TraceSettings(spin=a, r_in=kerr.isco(a), r_out=25.0, step=0.01, max_steps=20000),
        output_mode=1,
    )
    img = renderer.read()
    d_all = pixel_dirs(rot, fov, 1.0, W, W)
    rng = np.random.default_rng(0)
    disc = np.argwhere(img[..., 3] == 2.0)
    bodies, vels, dirs, g_shader = [], [], [], []
    for j, i in disc[rng.choice(len(disc), 15, replace=False)]:
        d = d_all[j, i]
        st = kerr.photon_from_camera(e, d, a)
        out = kerr.trace_photon(*st[:7], a, kerr.isco(a), 25.0, 600.0, 0.005, 200000)
        if out[0] != 2:
            continue
        s, c = kerr_circular_state(out[1], a)
        s[2] = out[3]
        k = KerrBodySystem([s], [c], a)
        bodies.append(k.x[0])
        vels.append(k.w[0])
        dirs.append(d)
        g_shader.append(img[j, i, 1])
    dirs = np.array(dirs)
    guess = dirs + rng.normal(0, 0.009, dirs.shape)  # half a degree off
    d, vis, solid, g, delay = kerr_images(
        e, np.array(bodies), np.array(vels), guess, a, kerr.isco(a), 25.0, False
    )
    assert vis.all()
    ang = np.degrees(np.arccos(np.clip(np.sum(d * dirs, 1), -1, 1)))
    assert ang.max() < 0.01
    assert np.allclose(g, g_shader, rtol=5e-3)
