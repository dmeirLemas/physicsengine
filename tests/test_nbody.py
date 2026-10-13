"""Tests for the direct-summation acceleration in modules.nbody.

Units: AU, years, solar masses, so G = 4 pi^2 (Kepler's third law,
P^2 = a^3 for P in years and a in AU around one solar mass).

Reference values:
- Newton's law of gravitation, a = G M / r^2 directed toward the other body.
  Sun (M = 1) at the origin, Earth (m = 3.0e-6) at (1, 0, 0), r = 1:
    Earth: (-4 pi^2, 0, 0)          ~ (-39.478, 0, 0)
    Sun:   (+4 pi^2 * 3.0e-6, 0, 0) ~ (+1.184e-4, 0, 0)
- Newton's third law: pairwise forces cancel, so sum_i m_i a_i = 0 for any
  configuration of an isolated system.
"""

import numpy as np

from physengine.modules.nbody import (
    NBodyState,
    acc,
    energy,
    sun_earth_scenario,
    total_angular_momentum,
    total_momentum,
)

G = 4.0 * np.pi**2


def test_sun_earth_acceleration():
    state = NBodyState(2)
    state.pos[:] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    state.vel[:] = [[0.0, 0.0, 0.0], [0.0, 2.0 * np.pi, 0.0]]
    state.mass[:] = [1.0, 3.0e-6]

    a = acc(state.pos, state.mass, G, 0.0)

    expected = np.array(
        [
            [G * 3.0e-6, 0.0, 0.0],
            [-G, 0.0, 0.0],
        ]
    )
    # atol handles the components that are exactly zero; it sits far below
    # the smallest nonzero expected value (1.2e-4).
    np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-15)


def test_total_force_vanishes():
    rng = np.random.default_rng(seed=12345)
    n = 5
    pos = rng.uniform(-10.0, 10.0, size=(n, 3))
    mass = rng.uniform(1e-6, 1.0, size=n)

    a = acc(pos, mass, G, 0.0)

    forces = mass[:, None] * a
    total = forces.sum(axis=0)
    scale = np.abs(forces).max()
    np.testing.assert_allclose(total, np.zeros(3), atol=1e-12 * scale)


def test_total_momentum_zero():
    nb = sun_earth_scenario()
    totmom = total_momentum(nb)

    np.testing.assert_allclose(totmom, np.zeros(3), atol=1e-12)


def test_total_angular_momentum():
    nb = sun_earth_scenario()
    totmom = total_angular_momentum(nb)

    np.testing.assert_allclose(totmom, np.array([0, 0, 1.88495276e-5]), atol=1e-12)


def test_total_energy():
    nb = sun_earth_scenario()
    pos = nb.pos
    mass = nb.mass
    vel = nb.vel
    totenergy = energy(pos, vel, mass, G, 0.0)

    np.testing.assert_allclose(totenergy, -5.9218e-5, atol=1e-9)
