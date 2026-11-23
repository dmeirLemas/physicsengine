"""Tests of the 1PN (EIH) gravity of the N-body module (modules/postnewtonian.py).

Units: AU, years, solar masses, G = 4 pi^2, c = 63 241.08 AU/yr.

Reference values and their sources:

- Test body around a mass M at rest, harmonic coordinates (e.g. Poisson &
  Will 2014, Gravity, ch. 9):
      a = -G M / r^2 [ (1 - 4 G M / (r c^2) + v^2 / c^2) n - 4 (n.v) v / c^2 ],
      E = v^2/2 - G M / r + (3/8 v^4 + 3/2 G M v^2 / r + 1/2 G^2 M^2 / r^2) / c^2.
- Perihelion advance 6 pi G M / (c^2 a (1 - e^2)) per orbit; for Mercury
  (a = 0.387098 AU, e = 0.205630) 42.98 arcseconds per century.
- The 1PN energy is conserved up to 2PN terms, (v/c)^4 relative.
"""

import numpy as np
import pytest

from physengine.modules.nbody_binary import advance_relative, invariants, mean_motion, peters
from physengine.modules.nbody_scenes import M_SGR_A
from physengine.modules.nbodymodule import RELATIVITY, NBodyModule, force_1pn
from physengine.modules.nbodystate import Integrator, NBodyState, acc
from physengine.modules.postnewtonian import C_AU_PER_YEAR as C
from physengine.modules.postnewtonian import (
    eih_acc,
    eih_energy,
    inspiral_time,
    quadrupole_ddot,
    rr_acc,
    schwarzschild_radius,
)

G = 4 * np.pi**2


def test_test_particle_acceleration():
    x = np.array([0.7, -0.4, 0.3])
    v = np.array([4.0, 6.0, -1.0])
    a = eih_acc(np.array([[0, 0, 0], x]), np.array([[0, 0, 0], v]), np.array([1.0, 0.0]), G, C)[1]
    r = np.linalg.norm(x)
    n = x / r
    ref = -G / r**2 * ((1 - 4 * G / (r * C**2) + v @ v / C**2) * n - 4 * (n @ v) * v / C**2)
    newton = -G / r**2 * n
    assert np.linalg.norm(a - ref) < 1e-6 * np.linalg.norm(ref - newton)


def test_test_particle_energy():
    x = np.array([0.7, 0.2, 0.1])
    v = np.array([1.0, 5.0, -2.0])
    m = 1e-12
    E = eih_energy(np.array([[0, 0, 0], x]), np.array([[0, 0, 0], v]), np.array([1.0, m]), G, C) / m
    r, v2 = np.linalg.norm(x), v @ v
    pn = (3 / 8 * v2**2 + 1.5 * G * v2 / r + 0.5 * G**2 / r**2) / C**2
    assert E - (0.5 * v2 - G / r) == pytest.approx(pn, rel=1e-6)


def test_energy_conserved_to_2pn_order():
    c = C / 300  # 1PN terms ~1e-3
    s = NBodyState(3)
    s.pos[:] = [[0, 0, 0], [1.0, 0, 0], [0, 2.0, 0.3]]
    s.vel[:] = [[0, -0.05, 0], [0, 6.0, 0.4], [-4.2, 0, 0]]
    s.mass[:] = [1.0, 3e-3, 1e-3]
    integ = Integrator(lambda p, m, v: eih_acc(p, v, m, G, c), velocity_dependent=True)
    e0 = eih_energy(s.pos, s.vel, s.mass, G, c)
    worst = 0.0
    for _ in range(2000):
        integ.advance(s, 1e-3)
        worst = max(worst, abs(eih_energy(s.pos, s.vel, s.mass, G, c) / e0 - 1))
    assert worst < 5e-5  # against 1PN terms of 1e-3


def _lrl(s):
    rel = s.pos[1] - s.pos[0]
    v = s.vel[1] - s.vel[0]
    return np.cross(v, np.cross(rel, v)) / (G * s.mass.sum()) - rel / np.linalg.norm(rel)


def test_mercury_perihelion():
    a, e = 0.387098, 0.205630
    r_p = a * (1 - e)
    v_p = np.sqrt(G * (1 + e) / r_p)
    P = 2 * np.pi * np.sqrt(a**3 / G)
    turn = {}
    for c in (C, 1e30):  # 1PN, and Newton with the same steps
        s = NBodyState(2)
        s.pos[:] = [[0, 0, 0], [r_p, 0, 0]]
        s.vel[:] = [[0, 0, 0], [0, v_p, 0]]
        s.mass[:] = [1.0, 1.66e-7]
        integ = Integrator(lambda p, m, v, c=c: eih_acc(p, v, m, G, c), velocity_dependent=True)
        A0 = _lrl(s)
        for _ in range(5 * 2000):
            integ.advance(s, P / 2000)
        A1 = _lrl(s)
        turn[c] = (np.arctan2(A1[1], A1[0]) - np.arctan2(A0[1], A0[0])) / 5
    arcsec_per_century = np.degrees(turn[C] - turn[1e30]) * 3600 * 100 / P
    assert arcsec_per_century == pytest.approx(42.98, abs=0.05)


def test_module_orbit_near_a_black_hole_matches_fine_reference():
    m = NBodyModule()
    m.scene_name = "Star near a black hole"
    m.gravity = RELATIVITY
    m.load_scene(m.scene_name)
    p0, v0, ms = m.nb.pos.copy(), m.nb.vel.copy(), m.nb.mass.copy()
    P = np.sqrt(50.0**3 / M_SGR_A)
    # Two orbits. The module slows down near the hole (nbodymodule.SLOW_MOTION),
    # so its steps are not all dt long: compare at the time it has reached.
    while m.elapsed < 2 * P:
        m.step(m.dt)
    s = NBodyState(2)
    s.pos[:], s.vel[:], s.mass[:] = p0, v0, ms
    ref = Integrator(force_1pn, velocity_dependent=True)
    h = m.elapsed / 20000
    for _ in range(20000):
        ref.advance(s, h)

    def angle(pos):
        rel = pos[1] - pos[0]
        return np.degrees(np.arctan2(rel[1], rel[0]))

    assert angle(m.nb.pos) == pytest.approx(angle(s.pos), abs=0.01)


def test_black_hole_radius():
    # 10 solar masses: 2 G M / c^2 = 29.5 km = 1.97e-7 AU.
    assert schwarzschild_radius(10.0, G) * 1.495978707e8 == pytest.approx(29.53, abs=0.02)


def test_radiation_reaction_matches_peters():
    # Peters (1964): a circular binary shrinks at
    # da/dt = -64 G^3 m1 m2 M / (5 c^5 a^3). Newtonian gravity plus the 2.5PN
    # force, with c lowered to 300 AU/yr so that 20 orbits show the decay.
    c, m1, m2, a0 = 300.0, 1.0, 0.5, 1.0
    total = m1 + m2
    v = np.sqrt(G * total / a0)
    s = NBodyState(2)
    s.pos[:] = [[-m2 / total * a0, 0, 0], [m1 / total * a0, 0, 0]]
    s.vel[:] = [[0, -m2 / total * v, 0], [0, m1 / total * v, 0]]
    s.mass[:] = [m1, m2]

    def semi_major_axis():
        r = np.linalg.norm(s.pos[1] - s.pos[0])
        return 1.0 / (2.0 / r - np.sum((s.vel[1] - s.vel[0]) ** 2) / (G * total))

    integ = Integrator(
        lambda p, m, vel: acc(p, m, G, 0.0) + rr_acc(p, vel, m, G, c), velocity_dependent=True
    )
    period = 2 * np.pi * np.sqrt(a0**3 / (G * total))
    start = semi_major_axis()
    for _ in range(20 * 400):
        integ.advance(s, period / 400)
    rate = (semi_major_axis() - start) / (20 * period)
    assert rate == pytest.approx(-64 * G**3 * m1 * m2 * total / (5 * c**5 * a0**3), rel=2e-3)
    np.testing.assert_allclose(s.mass @ s.vel, 0.0, atol=1e-12)  # momentum conserved


def test_quadrupole_of_a_circular_binary():
    # Equal masses m on a circle of radius d/2 at angular speed w, at phase 0
    # (on the x axis, moving along y): Q_xx - Q_yy = (m d^2 / 2) cos(2 w t),
    # so its second derivative is -2 m d^2 w^2 = -8 * (m v^2) with v = w d / 2
    # each body's speed, and Qdd_xy = 0 at that instant.
    m, d, w = 2.0, 3.0, 0.7
    pos = np.array([[d / 2, 0, 0], [-d / 2, 0, 0.0]])
    vel = np.array([[0, w * d / 2, 0], [0, -w * d / 2, 0.0]])
    centripetal = -(w**2) * pos
    q = quadrupole_ddot(pos, vel, centripetal, np.array([m, m]))
    assert q[0, 0] - q[1, 1] == pytest.approx(-2 * m * d**2 * w**2)
    assert q[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert np.trace(q) == pytest.approx(0.0, abs=1e-12)


def test_black_hole_merger_scene():
    # Two holes of 5e9 solar masses, 20 G M / c^2 apart. Peters' time,
    # 5 c^5 a^4 / (256 G^3 m1 m2 M) = 19.5 yr; with the 1PN terms the
    # module takes 18.8 yr (measured). The merged hole has lost 4.8% of the
    # mass to gravitational waves.
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = "Black hole merger"
    m.load_scene(m.scene_name)
    separation = np.linalg.norm(m.nb.pos[1] - m.nb.pos[0])
    assert inspiral_time(5e9, 5e9, separation, G) == pytest.approx(19.5, abs=0.1)
    steps = 0
    while m.nb.N == 2 and steps < 25 * 240:
        m.step(m.dt)
        steps += 1
    assert steps * m.dt == pytest.approx(18.8, abs=0.5)
    assert m.nb.mass[0] == pytest.approx(1e10 * (1 - 0.048), rel=1e-9)


# ------------------------------------------------ the analytic orbit of a pair
#
# nbody_binary.advance_relative: the 1PN orbit of a bound pair in closed form
# (quasi-Keplerian, Damour & Deruelle 1985) with Peters' decay. References:
# - periapsis advance per radial period: 6 pi G M / (c^2 a (1 - e^2));
# - the 1PN first integrals of the relative motion (Blanchet 2014, sec. 7.1);
#   the energy is postnewtonian.eih_energy of the two bodies / reduced mass;
# - Peters (1964): a circular orbit shrinks as a(t) = a0 (1 - t / T)^(1/4),
#   T = 5 c^5 a0^4 / (256 G^3 m1 m2 M); for small e, e is proportional to
#   a^(19/12) (eq. 5.11 to lowest order in e).


def _pair_at_apocentre(m1, m2, a, e):
    total = m1 + m2
    r = a * (1 + e)
    v = np.sqrt(G * total * (1 - e) / r)
    return np.array([r, 0.0, 0.0]), np.array([0.0, v, 0.0])


def test_analytic_orbit_keeps_the_1pn_energy_and_is_reversible():
    m1, m2, c = 1.0, 0.5, 100.0  # G M / (a c^2) = 6e-3
    total, mu = m1 + m2, m1 * m2 / (m1 + m2)
    x0, v0 = _pair_at_apocentre(m1, m2, 1.0, 0.3)
    eps0, h0, _, _ = invariants(x0, v0, G * total, mu / total, c**-2)
    # eps is the code's own 1PN energy of the two bodies at rest as a whole.
    pos = np.array([m2 / total * x0, -m1 / total * x0])
    vel = np.array([m2 / total * v0, -m1 / total * v0])
    assert mu * eps0 == pytest.approx(eih_energy(pos, vel, np.array([m1, m2]), G, c), rel=1e-13)
    x, v, code = advance_relative(x0, v0, m1, m2, G, c**-2, 5.3, 0.0, False)
    assert code == 1
    eps, h, _, _ = invariants(x, v, G * total, mu / total, c**-2)
    assert eps == pytest.approx(eps0, rel=1e-13) and h == pytest.approx(h0, rel=1e-13)
    # Back by the same time: the same state. And in 1000 steps: the same.
    xb, vb, _ = advance_relative(x, v, m1, m2, G, c**-2, -5.3, 0.0, False)
    # (At a turning point the radial speed is the root of a rounding error:
    # 1e-7 of the speed.)
    np.testing.assert_allclose(xb, x0, atol=1e-12)
    np.testing.assert_allclose(vb, v0, atol=1e-6)
    xs, vs = x0, v0
    for _ in range(1000):
        xs, vs, _ = advance_relative(xs, vs, m1, m2, G, c**-2, 5.3e-3, 0.0, False)
    np.testing.assert_allclose(xs, x, atol=1e-9)


def test_analytic_orbit_is_keplers_without_relativity():
    # 1 / c^2 = 0: after one period 2 pi sqrt(a^3 / (G M)) the pair is back;
    # after half of it, at the periapsis a (1 - e) on the other side.
    m1, m2, a, e = 1.0, 0.25, 1.3, 0.6
    x0, v0 = _pair_at_apocentre(m1, m2, a, e)
    period = 2 * np.pi * np.sqrt(a**3 / (G * (m1 + m2)))
    x, v, _ = advance_relative(x0, v0, m1, m2, G, 0.0, period, 0.0, False)
    np.testing.assert_allclose(x, x0, atol=1e-7)
    np.testing.assert_allclose(v, v0, atol=1e-6)
    x, v, _ = advance_relative(x0, v0, m1, m2, G, 0.0, 0.5 * period, 0.0, False)
    np.testing.assert_allclose(x, [-a * (1 - e), 0.0, 0.0], atol=1e-7)


def test_analytic_orbit_precesses_at_the_1pn_rate():
    # One radial period from the apocentre: back at the apocentre, turned by
    # 6 pi G M / (c^2 a (1 - e^2)). c = 1000 AU/yr: G M / (a c^2) = 4e-5, so
    # the formula's a and e are the orbit's to that accuracy; within 0.1%.
    m1, m2, a, e, c = 1.0, 1e-6, 1.0, 0.5, 1000.0
    total = m1 + m2
    x0, v0 = _pair_at_apocentre(m1, m2, a, e)
    eps, h, _, _ = invariants(x0, v0, G * total, m1 * m2 / total**2, c**-2)
    period = 2 * np.pi / mean_motion(eps, G * total, m1 * m2 / total**2, c**-2)
    x, v, _ = advance_relative(x0, v0, m1, m2, G, c**-2, period, 0.0, False)
    assert np.linalg.norm(x) == pytest.approx(np.linalg.norm(x0), rel=1e-9)
    expected = 6 * np.pi * G * total / (c**2 * a * (1 - e**2))
    assert np.arctan2(x[1], x[0]) == pytest.approx(expected, rel=1e-3)

    # And against the direct integration of the EIH equations (6000 steps
    # per orbit) over 6.3 orbits at c = 200 (G M / (a c^2) = 1e-3): the
    # position angle agrees to 3% of the precession accumulated, 0.156 rad
    # (measured 1.4%: the 2PN terms the direct integration carries along).
    c = 200.0
    s = NBodyState(2)
    s.pos[:] = [[0, 0, 0], x0]
    s.vel[:] = [[0, 0, 0], v0]
    s.mass[:] = [m1, m2]
    rel0 = s.pos[1] - s.pos[0], s.vel[1] - s.vel[0]
    eps, h, _, _ = invariants(*rel0, G * total, m1 * m2 / total**2, c**-2)
    period = 2 * np.pi / mean_motion(eps, G * total, m1 * m2 / total**2, c**-2)
    integ = Integrator(lambda p, m, vel: eih_acc(p, vel, m, G, c), velocity_dependent=True)
    steps = int(6.3 * 6000)
    for _ in range(steps):
        integ.advance(s, period / 6000)
    x, v, _ = advance_relative(*rel0, m2, m1, G, c**-2, steps * period / 6000, 0.0, False)
    direct = s.pos[1] - s.pos[0]
    precession = 6.3 * 6 * np.pi * G * total / (c**2 * a * (1 - e**2))
    difference = np.arctan2(x[1], x[0]) - np.arctan2(direct[1], direct[0])
    assert abs((difference + np.pi) % (2 * np.pi) - np.pi) < 0.03 * precession
    assert np.linalg.norm(x) == pytest.approx(np.linalg.norm(direct), rel=2e-3)


def test_peters_decay():
    # Circular: a(t) = a0 (1 - t / T)^(1/4). Nearly circular: e ~ a^(19/12).
    m1, m2, a0, c = 1.0, 0.5, 1.0, 300.0
    kgw = G**3 * m1 * m2 * (m1 + m2) / c**5
    T = inspiral_time(m1, m2, a0, G, c)
    a, e, used, stopped = peters(a0, 0.0, 0.5 * T, kgw, 0.0)
    assert (e, used, stopped) == (0.0, 0.5 * T, False)
    assert a == pytest.approx(a0 * 0.5**0.25, rel=1e-6)
    a, e, used, stopped = peters(a0, 0.01, 0.5 * T, kgw, 0.0)
    assert e == pytest.approx(0.01 * (a / a0) ** (19 / 12), rel=1e-3)
    # It stops where the periapsis reaches the given distance: for a
    # circular orbit after T (1 - (r / a0)^4).
    a, e, used, stopped = peters(a0, 0.0, 2 * T, kgw, 0.3)
    assert stopped and a == pytest.approx(0.3, rel=0.03)
    assert used == pytest.approx(T * (1 - 0.3**4), rel=2e-3)


def test_decaying_analytic_orbit_loses_energy_like_the_direct_integration():
    # The 2.5PN force with the 1PN orbit, integrated directly over 10 orbits
    # (G M / (a c^2) = 1.5e-3, e = 0.5), against the analytic decay over the
    # same time: the energy lost agrees within 2% (measured 0.3%; Peters'
    # rates with the orbit's actual size and shape).
    m1, m2, a, e, c = 1.0, 0.5, 1.0, 0.5, 200.0
    total, nu = m1 + m2, m1 * m2 / (m1 + m2) ** 2
    x0, v0 = _pair_at_apocentre(m1, m2, a, e)
    eps0, h0, _, _ = invariants(x0, v0, G * total, nu, c**-2)
    period = 2 * np.pi / mean_motion(eps0, G * total, nu, c**-2)
    s = NBodyState(2)
    s.pos[:] = [m2 / total * x0, -m1 / total * x0]
    s.vel[:] = [m2 / total * v0, -m1 / total * v0]
    s.mass[:] = [m1, m2]
    integ = Integrator(
        lambda p, m, vel: eih_acc(p, vel, m, G, c) + rr_acc(p, vel, m, G, c),
        velocity_dependent=True,
    )
    for _ in range(10 * 3000):
        integ.advance(s, period / 3000)
    direct = invariants(s.pos[0] - s.pos[1], s.vel[0] - s.vel[1], G * total, nu, c**-2)[0]
    x, v, code = advance_relative(x0, v0, m1, m2, G, c**-2, 10 * period, 1e-3, True)
    analytic = invariants(x, v, G * total, nu, c**-2)[0]
    assert code == 1
    assert direct - eps0 < -1e-5 * abs(eps0)
    assert analytic - eps0 == pytest.approx(direct - eps0, rel=0.02)
