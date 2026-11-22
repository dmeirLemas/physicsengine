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
