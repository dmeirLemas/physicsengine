"""Tests of the black-hole module's physics on the CPU: Schwarzschild and Kerr
formulas, body geodesics, histories, lensed images of bodies, the star model.

Units G = c = M = 1. Reference values and their sources:

- Light bending, Schwarzschild: deflection 4M/b + 15 pi M^2 / (4 b^2) + ...
  for large impact parameter b (e.g. Weinberg 1972, sec. 8.5); shadow seen
  from far away: b_c = sqrt(27) M (Synge 1966, MNRAS 131, 463).
- Thin-disc flux (Shakura & Sunyaev 1973, no torque at r_in): peak at
  r = 49/36 r_in. Relativistic (Page & Thorne 1974): zero at the isco, peak at
  r = 9.55 M for a = 0, Newtonian shape at large r.
- Kerr isco (Bardeen, Press & Teukolsky 1972, eq. 2.21): 6, 4.233, 2.321,
  1.237 M for a = 0, 0.5, 0.9, 0.998.
- Massive bodies: pericentre advance per orbit = 2 int L/r^2 / sqrt(E^2 - V) dr
  - 2 pi with V = (1 - 2/r)(1 + L^2/r^2) (Hartle 2003, ch. 9); circular
  orbits stable for r > 6M; a body from rest at infinity is captured if
  L < 4M (Chandrasekhar 1983, sec. 19). Free fall from rest at r: E = sqrt(1 - 2/r).
- Kerr circular prograde orbit: Omega = 1 / (r^1.5 + a) (Bardeen et al.
  1972). Lense-Thirring: the orbit plane precesses at 2 a / r^3, i.e.
  4 pi a / r^1.5 per orbit (weak field).
- Light-travel time along a radial ray: difference of r + 2 ln(r/2 - 1)
  (tortoise coordinate).
- Plummer sphere in equilibrium: 2K / |W| = 1; half-mass radius 1.305 a
  (cut at 5 a: 1.25 a). Tidal disruption: about half the debris is bound
  (Rees 1988, Nature 333, 523).
"""

import numpy as np
import pytest

from physengine.modules import kerr
from physengine.modules.blackhole_images import RayTable, body_images
from physengine.modules.blackhole_scenes import (
    plummer_star,
    star_in_hole_units,
    tidal_radius,
)
from physengine.modules.blackhole_sph import Gas, plummer_energy
from physengine.modules.blackhole_star import SWALLOWED, TORN
from physengine.modules.geodesics import (
    BodySystem,
    History,
    bound_orbit,
    circular_orbit,
    energy,
    newtonian_mutual,
    orbit_normal,
    orbit_state,
    precession_per_orbit,
)
from physengine.modules.kerr_bodies import (
    KerrBodySystem,
    kerr_capture_L,
    kerr_circular_state,
    kerr_images,
    local_velocity_from_schwarzschild,
    state_from_local,
)
from physengine.modules.schwarzschild import (
    B_CRITICAL,
    deflection_angle,
    disc_flux_shape,
    escape_angle,
    impact_parameter,
    shadow_angular_radius,
)

# ------------------------------------------------------- Schwarzschild formulas


@pytest.mark.parametrize("b", [1000.0, 300.0])
def test_weak_field_deflection(b):
    series = 4.0 / b + 15.0 * np.pi / (4.0 * b * b)
    assert deflection_angle(b) == pytest.approx(series, rel=2e-3 * 100 / b)


def test_shadow_from_far_away_is_sqrt27():
    r = 1e6
    assert impact_parameter(r, shadow_angular_radius(r)) == pytest.approx(B_CRITICAL, rel=1e-5)


def test_escape_angle_far_observer_is_pi_plus_deflection():
    r, b = 1e7, 50.0
    alpha = np.pi - np.arcsin(b * np.sqrt(1 - 2 / r) / r)
    assert escape_angle(r, alpha) - np.pi == pytest.approx(deflection_angle(b) - b / r, abs=1e-6)


def test_newtonian_disc_flux_peak():
    r = np.linspace(6.0, 20.0, 140001)
    assert r[np.argmax(disc_flux_shape(r, 6.0))] == pytest.approx(49 / 36 * 6, abs=1e-3)


# ------------------------------------------------------------- Kerr formulas


@pytest.mark.parametrize("a, r_isco", [(0.0, 6.0), (0.5, 4.233), (0.9, 2.321), (0.998, 1.237)])
def test_kerr_isco(a, r_isco):
    assert kerr.isco(a) == pytest.approx(r_isco, abs=1e-3)


@pytest.mark.parametrize("a, peak", [(1e-9, 9.554), (0.9, 3.442)])
def test_page_thorne_flux(a, peak):
    r, F = kerr.disc_flux_table(a, r_out=3000.0, n=30000)
    assert F[0] == 0.0
    assert r[np.argmax(F)] == pytest.approx(peak, abs=0.01)
    ratio = F / disc_flux_shape(r, kerr.isco(a)).clip(1e-300)
    late = np.interp([1000.0, 2500.0], r, ratio)
    assert late[0] == pytest.approx(late[1], rel=0.03)  # Newtonian shape far out


def test_kerr_photon_integrator_at_zero_spin_matches_schwarzschild():
    a, r0 = 1e-6, 30.0
    cam = np.array([0.0, 0.0, r0])
    for alpha in np.linspace(0.3, 2.6, 8):
        d = np.array([np.sin(alpha) * 0.6, np.sin(alpha) * 0.8, np.cos(alpha)])
        st = kerr.photon_from_camera(cam, d, a)
        out = kerr.trace_photon(*st[:7], a, 0.0, -1.0, 5000.0, 0.005, 200000)
        ref = escape_angle(r0, alpha)
        if ref is None:
            assert out[0] == 0
            continue
        assert out[0] == 1
        got = kerr.escape_direction(out[1], out[2], out[3], out[5], out[6], st[5], a)
        ch = cam / r0
        et = d - (d @ ch) * ch
        et /= np.linalg.norm(et)
        want = np.cos(ref) * ch + np.sin(ref) * et
        assert np.degrees(np.arctan2(np.linalg.norm(np.cross(want, got)), want @ got)) < 0.01


# ------------------------------------------------------------ bodies (exact)


@pytest.mark.parametrize("r_p, r_a", [(10.0, 25.0), (8.0, 12.0)])
def test_precession_and_conservation(r_p, r_a):
    E, L = bound_orbit(r_p, r_a)
    x, w = orbit_state(r_p, L, E, inward=False)
    s = BodySystem([x], [w])
    s.ETA = 0.002
    T = 2 * np.pi * ((r_p + r_a) / 2) ** 1.5
    dt = T / 2000
    rs, angs = [], []
    for _ in range(int(3.3 * 2000)):
        s.advance(dt)
        rs.append(s.r()[0])
        angs.append(np.arctan2(s.x[0, 2], s.x[0, 0]))
    rs, angs = np.array(rs), np.unwrap(angs)
    i = next(i for i in range(1, len(rs) - 1) if rs[i] < rs[i - 1] and rs[i] <= rs[i + 1])
    j = next(j for j in range(i + 10, len(rs) - 1) if rs[j] < rs[j - 1] and rs[j] <= rs[j + 1])

    def refine(k):
        a_, b_, c_ = rs[k - 1], rs[k], rs[k + 1]
        return angs[k] + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) * (angs[k + 1] - angs[k - 1]) / 2

    advance = abs(refine(j) - refine(i)) - 2 * np.pi
    assert advance == pytest.approx(precession_per_orbit(r_p, r_a), rel=1e-4)
    assert energy(s.x, s.w)[0] == pytest.approx(E, rel=1e-11)
    assert np.linalg.norm(np.cross(s.x[0], s.w[0])) == pytest.approx(L, rel=1e-11)


@pytest.mark.parametrize("r, stays", [(7.0, True), (5.5, False)])
def test_isco(r, stays):
    E, L = circular_orbit(r)
    x, w = orbit_state(r, L * (1 - 1e-6), E)
    s = BodySystem([x], [w])
    s.advance(20 * 2 * np.pi * r**1.5)
    assert bool(s.alive[0]) == stays


def test_free_fall_energy():
    s = BodySystem([[10.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    assert energy(s.x, s.w)[0] == pytest.approx(np.sqrt(0.8))
    s.advance(40.0)
    assert energy(s.x, s.w)[0] == pytest.approx(np.sqrt(0.8), rel=1e-9)


def test_history_round_trip():
    E, L = bound_orbit(10.0, 25.0)
    x, w = orbit_state(25.0, L, E, normal=orbit_normal(0.5))
    s = BodySystem([x], [w])
    h = History(s, 0.5, 400)
    s.advance(50.0)
    xp, wp, ok = h.state_at(s, np.array([0]), np.array([-100.0]))
    assert ok[0]
    f = BodySystem(xp, wp)
    f.advance(100.0)
    assert np.linalg.norm(f.x[0] - x) < 1e-5  # float32 storage


# --------------------------------------------------------------- Kerr bodies


def test_kerr_bodies_at_zero_spin_match_schwarzschild():
    E, L = bound_orbit(10.0, 25.0)
    x, w = orbit_state(25.0, L, E, normal=orbit_normal(np.radians(30)))
    s = BodySystem([x], [w])
    st, c = state_from_local(x, local_velocity_from_schwarzschild(x, w), 1e-7)
    assert c[0] == pytest.approx(E, rel=1e-9)
    assert np.sqrt(c[1] ** 2 + c[2]) == pytest.approx(L, rel=1e-9)
    k = KerrBodySystem([st], [c], 1e-7)
    s.advance(200.0)
    k.advance(200.0)
    assert np.linalg.norm(k.x[0] - s.x[0]) < 1e-5


def test_kerr_circular_orbit():
    a, r = 0.9, 4.0
    st, c = kerr_circular_state(r, a)
    k = KerrBodySystem([st], [c], a)
    k.advance(200.0)
    assert k.st[0, 0] == pytest.approx(r, abs=1e-7)
    assert k.st[0, 2] / 200.0 == pytest.approx(1 / (r**1.5 + a), rel=1e-7)


def test_kerr_inside_isco_plunges():
    st, c = kerr_circular_state(2.1, 0.9, nudge=1e-4)
    k = KerrBodySystem([st], [c], 0.9)
    k.advance(500.0)
    assert not k.alive[0]


def test_lense_thirring():
    a, r = 0.9, 30.0
    st, c = kerr_circular_state(r, a, inclination=np.radians(45))
    k = KerrBodySystem([st], [c], a)

    def node():
        h = np.cross(k.x[0], k.w[0])
        return np.arctan2(h[0], h[2])

    n0 = node()
    k.advance(5 * 2 * np.pi * r**1.5)
    assert (node() - n0) / 5 == pytest.approx(4 * np.pi * a / r**1.5, rel=0.1)


def test_kerr_time_reversal():
    E, L = bound_orbit(10.0, 25.0)
    x, w = orbit_state(25.0, L, E, normal=orbit_normal(0.5))
    a = 0.9
    st, c = state_from_local(x, local_velocity_from_schwarzschild(x, w), a)
    x0 = KerrBodySystem([st], [c], a).x[0]
    mirror = KerrBodySystem([[st[0], st[1], -st[2], -st[3], -st[4]]], [c], a)
    mirror.advance(100.0)
    b = mirror.st[0]
    forward = KerrBodySystem([[b[0], b[1], -b[2], -b[3], -b[4]]], [c], a)
    forward.advance(100.0)
    assert np.linalg.norm(forward.x[0] - x0) < 1e-9


def test_capture_threshold_at_zero_spin_is_four():
    assert kerr_capture_L(1e-6, 20.0, 30.0) == pytest.approx(4.0, abs=2e-3)


# ---------------------------------------------------------- images of bodies


def _trace_u(r0, alpha, phi_end, h=1e-4):
    u = 1 / r0
    v = -u * np.sqrt(1 - 2 * u) * np.cos(alpha) / np.sin(alpha)
    n = int(phi_end / h)
    h = phi_end / n
    for _ in range(n):
        k1 = (v, 3 * u * u - u)
        k2 = (v + h / 2 * k1[1], 3 * (u + h / 2 * k1[0]) ** 2 - (u + h / 2 * k1[0]))
        k3 = (v + h / 2 * k2[1], 3 * (u + h / 2 * k2[0]) ** 2 - (u + h / 2 * k2[0]))
        k4 = (v + h * k3[1], 3 * (u + h * k3[0]) ** 2 - (u + h * k3[0]))
        u += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return u


@pytest.fixture(scope="module")
def image_case():
    rng = np.random.default_rng(1)
    r0 = 30.0
    cam = np.array([0.0, 0.3, 1.0])
    cam *= r0 / np.linalg.norm(cam)
    pos = rng.normal(size=(30, 3))
    pos /= np.linalg.norm(pos, axis=1)[:, None]
    pos *= rng.uniform(4, 35, 30)[:, None]
    rays = RayTable(r0)
    im = body_images(
        rays,
        cam,
        pos,
        np.zeros_like(pos),
        np.sqrt(1 - 2 / np.linalg.norm(pos, axis=1)),
        disc_on=False,
    )
    return r0, cam, pos, im


def test_image_rays_reach_the_bodies(image_case):
    r0, cam, pos, im = image_case
    ch = cam / r0
    for k in range(0, 30, 3):
        for o in (0, 1):
            if not im.visible[k, o]:
                continue
            d = im.dirs[k, o]
            alpha = np.arccos(np.clip(d @ ch, -1, 1))
            e = d - (d @ ch) * ch
            e /= np.linalg.norm(e)
            D = np.arccos(np.clip(pos[k] @ ch / np.linalg.norm(pos[k]), -1, 1))
            phi = D if o == 0 else 2 * np.pi - D
            x = (np.cos(phi) * ch + np.sin(phi) * e) / _trace_u(r0, alpha, phi)
            assert np.linalg.norm(x - pos[k]) / np.linalg.norm(pos[k]) < 2e-4


def test_static_body_redshift(image_case):
    r0, cam, pos, im = image_case
    r = np.linalg.norm(pos, axis=1)[:, None]
    expected = np.sqrt(1 - 2 / r) / np.sqrt(1 - 2 / r0)
    assert np.allclose(
        im.g[im.visible], np.broadcast_to(expected, im.g.shape)[im.visible], rtol=1e-12
    )


def test_flat_space_apparent_size():
    rays = RayTable(2000.0)
    cam = np.array([0, 0, 2000.0])
    pos = np.array([[-600.0, 50.0, 1200.0]])
    im = body_images(rays, cam, pos, np.zeros((1, 3)), np.ones(1), disc_on=False)
    assert im.solid[0, 0] * np.linalg.norm(pos[0] - cam) ** 2 == pytest.approx(1.0, abs=0.01)


def test_radial_light_delay_is_tortoise_difference():
    def tortoise(r):
        return r + 2 * np.log(r / 2 - 1)

    rays = RayTable(50.0)
    im = body_images(
        rays,
        np.array([0, 0, 50.0]),
        np.array([[1e-3, 0, 10.0]]),
        np.zeros((1, 3)),
        np.ones(1),
        disc_on=False,
    )
    assert im.visible[0, 0]
    assert im.delay[0, 0] == pytest.approx(tortoise(50.0) - tortoise(10.0), rel=1e-6)


def test_kerr_images_at_zero_spin_match_schwarzschild(image_case):
    r0, cam, pos, im = image_case
    take = slice(0, 10)
    p = pos[take]
    guess = np.vstack([im.dirs[take, 0], im.dirs[take, 1]])
    d, vis, solid, g, delay = kerr_images(
        cam, np.vstack([p, p]), np.zeros((20, 3)), guess, 1e-6, 6.0, 20.0, False
    )
    for o in (0, 1):
        rows = slice(o * 10, (o + 1) * 10)
        ok = vis[rows] & im.visible[take, o]
        assert ok.sum() >= im.visible[take, o].sum() - 1
        ang = np.degrees(np.arccos(np.clip(np.sum(d[rows][ok] * im.dirs[take, o][ok], 1), -1, 1)))
        assert ang.max() < 0.01
        assert np.allclose(g[rows][ok], im.g[take, o][ok], rtol=1e-6)
        assert np.allclose(delay[rows][ok], im.delay[take, o][ok], atol=0.05)


# --------------------------------------------------------------- the star


def test_plummer_star_equilibrium():
    rng = np.random.default_rng(0)
    p, v = plummer_star(20000, 1.0, 1.0, rng)
    r = np.sort(np.linalg.norm(p, axis=1))
    assert r[len(r) // 2] == pytest.approx(1.25, abs=0.03)
    sub = rng.choice(20000, 2000, replace=False)
    ps, vs, m = p[sub], v[sub], 1.0 / 2000
    d = np.linalg.norm(ps[:, None] - ps[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    ratio = 2 * 0.5 * m * (vs**2).sum() / (0.5 * m * m * (1 / d).sum())
    assert ratio == pytest.approx(1.0, abs=0.06)


def _star_run(scene, log_mass, star, until, cooling="Fast"):
    from physengine.modules.blackholemodule import BlackHoleModule

    m = BlackHoleModule()
    m.scene_name, m.hole_mass_log, m.star, m.gas_cooling = scene, log_mass, star, cooling
    m.speed = 100.0
    m.load()
    while m.bodies.t < until:
        m.step(m.dt)
    return m


def test_tidal_radius_of_the_sun():
    # r_t = R (M / m)^(1/3) with R_sun = 6.957e5 km, G M_sun / c^2 = 1.4766 km:
    # 47.1 G M / c^2 for 1e6 solar masses, and it scales as M^(-2/3), so it
    # reaches the horizon (2) at the Hills mass, (47.1 / 2)^1.5 * 1e6 = 1.1e8.
    m, radius = star_in_hole_units("Sun-like", 1e6)
    assert tidal_radius(radius, m) == pytest.approx(47.1, rel=2e-3)
    m, radius = star_in_hole_units("Sun-like", 1.14e8)
    assert tidal_radius(radius, m) == pytest.approx(2.0, rel=1e-2)


def test_star_is_torn_or_swallowed_whole_depending_on_the_masses():
    # A bound orbit with L = 3.8, just under the capture limit. The Sun at
    # the default 3.2e5 solar-mass hole (radius 1.5 M, r_t = 102 M) is torn
    # into a strand; measured, 78% is swallowed by t = 300 M and 81% by
    # 450 M, the rest stays in orbit. A white dwarf at 1e6 solar masses
    # (r_t = 0.7 M, inside the horizon) goes in whole, and so does the Sun at
    # 3e8 solar masses (above the Hills mass).
    torn = _star_run("Star torn apart", 5.5, "Sun-like", 600.0)
    assert torn.intact_star.state == TORN
    assert 0.7 < np.mean(~torn.bodies.alive) < 0.9
    for log_mass, star in ((6.0, "White dwarf"), (8.5, "Sun-like")):
        whole = _star_run("Star torn apart", log_mass, star, 400.0)
        assert whole.intact_star.state == SWALLOWED
        assert not whole.bodies.alive.any()


def test_gas_star_stays_in_hydrostatic_balance():
    # The Plummer sphere is the n = 5 polytrope: gas at rest in it with
    # u = m / (4 sqrt(r^2 + a^2)) (gamma = 5/3) is in balance, so its
    # half-mass radius must stay put (measured: within 6% over 10 dynamical
    # times sqrt(a^3 / m), energy to 0.4%); without pressure it falls to
    # less than half within 2.5.
    n, a = 2000, 1.0
    x0, _ = plummer_star(n, 1.0, a, np.random.default_rng(1))
    mass = np.full(n, 1.0 / n)
    alive = np.ones(n, bool)
    eps, dt = 0.1, 0.01

    def half_mass_radius(with_gas, t_end):
        x, v = x0.copy(), np.zeros_like(x0)
        gas = Gas(n)
        gas.start(np.arange(n), plummer_energy(x, 1.0, a), 2.5 * a / np.cbrt(n))
        grav = np.zeros_like(x)

        def acc():
            newtonian_mutual(x, mass, alive, eps, grav)
            return grav + gas.accelerations(x, v, mass, alive) if with_gas else grav.copy()

        def total_energy():
            d2 = np.sum((x[:, None] - x[None]) ** 2, axis=2) + eps * eps
            pot = -0.5 * (np.sum(1.0 / np.sqrt(d2)) - n / eps) / n**2
            return 0.5 * np.sum(mass * np.sum(v**2, 1)) + pot + np.sum(mass * gas.u)

        e0 = total_energy()
        a_now = acc()
        for _ in range(int(t_end / dt)):
            v += 0.5 * dt * a_now
            gas.heat(0.5 * dt, np.arange(n))
            x += dt * v
            a_now = acc()
            v += 0.5 * dt * a_now
            gas.heat(0.5 * dt, np.arange(n))
        drift = abs(total_energy() / e0 - 1.0)
        return np.median(np.linalg.norm(x - x.mean(0), axis=1)), drift

    r0 = np.median(np.linalg.norm(x0, axis=1))
    r_gas, drift = half_mass_radius(True, 10.0)
    assert r_gas == pytest.approx(r0, rel=0.1)
    assert drift < 0.01
    assert half_mass_radius(False, 2.5)[0] < 0.6 * r0


def test_gas_debris_is_torn_heated_and_partly_eaten():
    # The Sun on the eccentric orbit (pericentre 8 M) at 1e6 solar masses:
    # released at once (r_t = 47 M > apocentre), stretched at the first
    # passage (t ~ 340 M), shocked from the second (t ~ 1000 M) on. Measured
    # with fast cooling: 2% eaten by t = 2000 M, 7% by 3000 M, and the gas a
    # thousand times hotter than the star was.
    m = _star_run("Star torn apart, with gas", 6.0, "Sun-like", 3000.0)
    b = m.bodies
    assert m.intact_star.state == TORN
    assert 0.03 < np.mean(~b.alive) < 0.15
    u_star = np.median(plummer_energy(m.intact_star.info.dx, 1e-6, 0.471 / 2.5))
    assert np.median(m.gas.u[b.alive]) > 100.0 * u_star


def test_gas_ring_drains_slowly_and_is_not_flung_out():
    # A ring of gas on circular orbits at 12 M, held at thin-disc
    # temperature, with shear viscosity. Angular momentum moves outward: the
    # inner gas reaches the innermost stable orbit and falls in, the outer
    # gas spreads. Measured at viscosity 1: 4.3% eaten by t = 1000 M, the
    # outermost particle at 27 M (nothing thrown out), median radius 13 M.
    # At viscosity 0.2 a third as much drains (2.4% against 7.3% by 1500 M).
    from physengine.modules.blackholemodule import BlackHoleModule

    def run(viscosity):
        m = BlackHoleModule()
        m.scene_name = "Gas ring"
        m.gas_viscosity = viscosity
        m.speed = 100.0
        m.load()
        while m.bodies.t < 1000.0:
            m.step(m.dt)
        return m.bodies

    b = run(1.0)
    eaten = np.mean(~b.alive)
    r = b.r()[b.alive]
    assert 0.02 < eaten < 0.08
    assert r.max() < 45.0
    assert 12.2 < np.median(r) < 14.5
    assert np.mean(~run(0.2).alive) < 0.6 * eaten
