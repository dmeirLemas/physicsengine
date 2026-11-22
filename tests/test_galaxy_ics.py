"""Tests of the galaxy initial conditions (modules/galacy_ics.py and the
galaxy module's load()).

Model units: G = 1, 1 kpc, 1e10 M_sun, 207.4 km/s, 4.715 Myr.

Reference values and their sources:

- Hernquist sphere (Hernquist 1990, ApJ 356, 359): enclosed mass
  M(<r) = M r^2 / (r + a)^2 (eq. 3), potential -G M / (r + a) (eq. 2), and
  the isotropic velocity dispersion (eq. 10)
      sigma^2 = G M / (12 a) [12 x (1 + x)^3 ln((1 + x) / x)
                              - x / (1 + x) (25 + 52 x + 42 x^2 + 12 x^3)],
      x = r / a.
  Truncated at r_max, the median radius is a s / (1 - s) with
  s = (r_max / (r_max + a)) / sqrt(2), and the mass fraction inside a is
  (1/4) / (r_max / (r_max + a))^2.
- Exponential disc: M(<R) / M = 1 - (1 + R/R_d) e^(-R/R_d); with R_d = 3
  and the cut at R_max = 18, the median R is 4.952 kpc and the fraction
  inside R_d is 0.2689 (both computed from that formula).
- sech^2 vertical profile: mean |z| = z_0 ln 2.
- Isothermal sheet: sigma_z^2 = pi G Sigma z_0 (Spitzer 1942).
- Toomre stability parameter for stars: Q = sigma_R kappa / (3.36 G Sigma)
  (Toomre 1964); the model sets Q = 1.5 at R = 2.5 R_d.
- Epicycle approximation: sigma_phi / sigma_R = kappa / (2 Omega); for a
  flat rotation curve kappa / Omega = sqrt(2) (Binney & Tremaine 2008,
  eq. 3.80 and 4.316).
- Circular speed of the composite galaxy from the analytic enclosed masses
  of the three components (spherical approximation): 0.918, 0.972, 0.967,
  0.917 at r = 2, 5, 8, 15 kpc.
- Virial theorem: in equilibrium 2K / |W| = 1.
- Midplane rotation curve: for a spherical mass distribution the in-plane
  circular speed equals sqrt(G M(<r) / r). A flattened exponential disc pulls
  harder in its own plane than the same mass spread over a sphere beyond about
  R_d, and less inside (Freeman 1970, ApJ 160, 811; Binney & Tremaine 2008,
  fig. 2.17), so the measured curve lies above the spherical one at 5-15 kpc.
"""

import numpy as np
import pytest

from physengine.modules.galacy_ics import (
    circular_velocity,
    disc_kinematics,
    enclosed_mass_and_potential,
    epicyclic_frequencies,
    hernquist_dispersion,
    ics,
    isotropic_velocities,
    midplane_rotation_curve,
    sample_disc,
)
from physengine.modules.galaxy_components import BULGE, DISC, HALO
from physengine.modules.galaxymodule import EPS, SINGLE, GalaxyModule


def hernquist_sigma_exact(r, M=1.0, a=1.0):
    x = r / a
    s2 = (
        M
        / (12 * a)
        * (
            12 * x * (1 + x) ** 3 * np.log((1 + x) / x)
            - x / (1 + x) * (25 + 52 * x + 42 * x**2 + 12 * x**3)
        )
    )
    return np.sqrt(s2)


@pytest.fixture(scope="module")
def isolated_sphere():
    """A big, effectively untruncated Hernquist sphere, M = a = 1."""
    return ics(200_000, 1.0, 1.0, 1e4, np.random.default_rng(3))


@pytest.fixture(scope="module")
def galaxy():
    m = GalaxyModule()
    m.scene_name = SINGLE
    m.tracers = "12000"
    m.load()
    return m


# ------------------------------------------------------------------ positions


@pytest.mark.parametrize("M, a, r_max", [(50.0, 30.0, 300.0), (1.0, 0.6, 6.0)])
def test_hernquist_sampler(M, a, r_max):
    n = 40_000
    pos, mass = ics(n, M, a, r_max, np.random.default_rng(0))
    r = np.linalg.norm(pos, axis=1)

    assert pos.shape == (n, 3)
    assert mass.sum() == pytest.approx(M)
    assert r.max() <= r_max

    u_max = (r_max / (r_max + a)) ** 2
    s = np.sqrt(u_max / 2)
    assert np.median(r) == pytest.approx(a * s / (1 - s), rel=0.02)
    assert np.mean(r < a) == pytest.approx(0.25 / u_max, abs=0.01)

    # Isotropic: unit vectors average to zero, and each axis has the same spread.
    unit = pos / r[:, None]
    np.testing.assert_allclose(unit.mean(axis=0), 0.0, atol=0.02)
    np.testing.assert_allclose(np.mean(unit**2, axis=0), 1 / 3, atol=0.01)


def test_disc_sampler():
    n = 40_000
    pos, mass = sample_disc(n, 5.0, 3.0, 0.3, 18.0, np.random.default_rng(0))
    R = np.hypot(pos[:, 0], pos[:, 1])

    assert pos.shape == (n, 3)
    assert not np.isnan(pos).any()
    assert mass.sum() == pytest.approx(5.0)
    assert R.max() <= 18.0
    assert np.median(R) == pytest.approx(4.952, rel=0.02)
    assert np.mean(R < 3.0) == pytest.approx(0.2689, abs=0.01)
    assert np.mean(np.abs(pos[:, 2])) == pytest.approx(0.3 * np.log(2), rel=0.02)
    assert np.mean(pos[:, 2] < 0) == pytest.approx(0.5, abs=0.01)


# ------------------------------------------------- potential and dispersion


def test_enclosed_mass_and_potential_of_hernquist_sphere(isolated_sphere):
    pos, mass = isolated_sphere
    r = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    m_enc, phi = enclosed_mass_and_potential(r, pos, mass)
    np.testing.assert_allclose(m_enc, r**2 / (r + 1) ** 2, rtol=0.02)
    np.testing.assert_allclose(phi, -1.0 / (r + 1.0), rtol=0.005)


def test_jeans_dispersion_matches_hernquist(isolated_sphere):
    pos, mass = isolated_sphere
    r = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    sigma = hernquist_dispersion(r, 1.0, 1e4, pos, mass)
    np.testing.assert_allclose(sigma, hernquist_sigma_exact(r), rtol=0.01)


def test_isotropic_velocities(isolated_sphere):
    pos, mass = isolated_sphere
    rng = np.random.default_rng(5)
    vel = isotropic_velocities(pos, 1.0, 1e4, pos, mass, rng)
    r = np.linalg.norm(pos, axis=1)

    _, phi = enclosed_mass_and_potential(r, pos, mass)
    assert np.all(np.linalg.norm(vel, axis=1) <= 0.95 * np.sqrt(-2 * phi) + 1e-12)

    # Away from the escape cut (which trims the fastest stars), the measured
    # spread per axis matches the exact dispersion.
    shell = (r > 0.5) & (r < 1.5)
    sigma_ref = np.mean(hernquist_sigma_exact(r[shell]))
    for axis in range(3):
        assert vel[shell, axis].std() == pytest.approx(sigma_ref, rel=0.03)


# ----------------------------------------------------------------- the galaxy


def test_galaxy_composition(galaxy):
    heavy = galaxy.nb.mass > 0
    assert heavy.sum() == 9000
    assert np.sum(heavy & (galaxy.kind == HALO)) == 4000
    assert np.sum(heavy & (galaxy.kind == BULGE)) == 1000
    assert np.sum(heavy & (galaxy.kind == DISC)) == 4000
    # Massless tracer stars: 10 000 in the disc, 2 000 in the bulge, none in the halo.
    assert np.sum(~heavy & (galaxy.kind == DISC)) == 10000
    assert np.sum(~heavy & (galaxy.kind == BULGE)) == 2000
    assert np.sum(~heavy & (galaxy.kind == HALO)) == 0
    assert galaxy.nb.mass.sum() == pytest.approx(56.0)
    assert not np.isnan(galaxy.nb.pos).any() and not np.isnan(galaxy.nb.vel).any()


def test_galaxy_is_centred_and_at_rest(galaxy):
    nb = galaxy.nb
    np.testing.assert_allclose(nb.mass @ nb.pos, 0.0, atol=1e-10)
    np.testing.assert_allclose(nb.mass @ nb.vel, 0.0, atol=1e-10)
    # Each component is centred on its own, so the bulge sits at the centre.
    bulge = nb.pos[galaxy.kind == BULGE]
    np.testing.assert_allclose(bulge.mean(axis=0), 0.0, atol=1e-10)


def test_galaxy_is_reproducible():
    a, b = GalaxyModule(), GalaxyModule()
    a.scene_name = b.scene_name = SINGLE
    a.load()
    b.load()
    np.testing.assert_array_equal(a.nb.pos, b.nb.pos)
    np.testing.assert_array_equal(a.nb.vel, b.nb.vel)


def test_rotation_curve(galaxy):
    r = np.array([2.0, 5.0, 8.0, 15.0])
    v = circular_velocity(r, galaxy.nb.pos, galaxy.nb.mass)
    np.testing.assert_allclose(v, [0.918, 0.972, 0.967, 0.917], rtol=0.04)


def test_epicyclic_ratio_of_flat_rotation_curve(galaxy):
    R = np.array([5.0, 7.5, 10.0])
    omega, kappa = epicyclic_frequencies(R, galaxy.nb.pos, galaxy.nb.mass)
    np.testing.assert_allclose(kappa / omega, np.sqrt(2), rtol=0.06)


def test_disc_kinematics_model(galaxy):
    nb = galaxy.nb
    M, R_d, z0, R_max = 5.0, 3.0, 0.3, 18.0
    R = np.array([3.0, 5.0, 7.5, 10.0])
    v_phi, s_R, s_phi, s_z = disc_kinematics(R, M, R_d, z0, R_max, nb.pos, nb.mass, Q_ref=1.5)
    omega, kappa = epicyclic_frequencies(R, nb.pos, nb.mass)

    sigma0 = M / (2 * np.pi * R_d**2) / (1 - 7 * np.exp(-6))
    surface = sigma0 * np.exp(-R / R_d)

    np.testing.assert_allclose(s_z**2, np.pi * surface * z0, rtol=1e-10)
    np.testing.assert_allclose(s_phi / s_R, kappa / (2 * omega), rtol=1e-10)
    # Q = 1.5 at the reference radius 2.5 R_d = 7.5 kpc, and above 1 elsewhere.
    Q = s_R * kappa / (3.36 * surface)
    assert Q[2] == pytest.approx(1.5, rel=1e-6)
    assert np.all(Q > 1.0)
    # Asymmetric drift: random motion means slower mean rotation than v_c.
    assert np.all(v_phi < circular_velocity(R, nb.pos, nb.mass))


def test_disc_velocities_follow_the_model(galaxy):
    nb = galaxy.nb
    disc = galaxy.kind == DISC
    p, v = nb.pos[disc], nb.vel[disc]
    R = np.hypot(p[:, 0], p[:, 1])
    v_R = (p[:, 0] * v[:, 0] + p[:, 1] * v[:, 1]) / R
    v_t = (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]) / R

    ring = (R > 6.5) & (R < 8.5)
    curve = midplane_rotation_curve(nb.pos, nb.mass, 18.0, EPS)
    model = disc_kinematics(
        np.array([7.5]), 5.0, 3.0, 0.3, 18.0, nb.pos, nb.mass, Q_ref=1.5, rotation_curve=curve
    )
    v_phi, s_R, s_phi, s_z = (x[0] for x in model)
    # ~400 stars in the ring: spreads to ~10%, mean rotation to a few percent.
    assert v_R[ring].std() == pytest.approx(s_R, rel=0.12)
    assert v_t[ring].std() == pytest.approx(s_phi, rel=0.12)
    assert v[ring, 2].std() == pytest.approx(s_z, rel=0.12)
    assert v_t[ring].mean() == pytest.approx(v_phi, rel=0.03)
    # The disc rotates counter-clockwise about +z.
    assert np.sum(v_t > 0) > 0.95 * len(v_t)


def test_galaxy_is_near_virial_equilibrium(galaxy):
    nb = galaxy.nb
    heavy = nb.mass > 0  # tracers have no mass: no potential or kinetic energy
    pos, mass, vel = nb.pos[heavy], nb.mass[heavy], nb.vel[heavy]
    n = len(mass)
    W = 0.0
    for i0 in range(0, n, 500):
        idx = np.arange(i0, min(i0 + 500, n))
        d = np.linalg.norm(pos[idx, None, :] - pos[None, :, :], axis=-1)
        d[np.arange(len(idx)), idx] = np.inf
        W -= 0.5 * np.sum(mass[idx, None] * mass[None, :] / d)
    K = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    # Measured 0.93: the escape-speed cut trims the fastest halo stars.
    assert 0.9 < 2 * K / abs(W) < 1.05


def test_midplane_curve_of_a_sphere_is_the_spherical_curve():
    pos, mass = ics(50_000, 1.0, 1.0, 50.0, np.random.default_rng(9))
    pos -= pos.mean(axis=0)
    R_grid, v_grid = midplane_rotation_curve(pos, mass, 10.0, 0.01)
    R = np.array([0.5, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(
        np.interp(R, R_grid, v_grid), circular_velocity(R, pos, mass), rtol=0.01
    )


def test_midplane_curve_of_the_galaxy_feels_the_flat_disc(galaxy):
    nb = galaxy.nb
    R_grid, v_grid = midplane_rotation_curve(nb.pos, nb.mass, 18.0, EPS)
    R = np.array([7.5, 10.0])
    ratio = np.interp(R, R_grid, v_grid) / circular_velocity(R, nb.pos, nb.mass)
    # Measured: +5.8% and +6.5%.
    assert np.all(ratio > 1.02) and np.all(ratio < 1.12)
