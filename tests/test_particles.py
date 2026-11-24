"""Tests of the particle-physics module (kinematics, decays, tracks, reconstruction).

Units: GeV, metres, nanoseconds, tesla; c = 1 in the kinematics.

Reference values and their sources (Particle Data Group, Review of Particle
Physics 2022, particle listings and the Kinematics review):

- Two-body decay momentum p* = sqrt((M^2 - (m1+m2)^2)(M^2 - (m1-m2)^2)) / 2M:
  K0_S -> pi+ pi-: 206 MeV; Lambda -> p pi-: 101 MeV (both quoted in the
  listings); Z -> mu+ mu-: 45.594 GeV (half the Z mass minus m_mu^2 / M_Z).
- Radius of curvature R [m] = p_T [GeV] / (0.2998 q B [T]): 3.336 m for
  1 GeV in 1 T.
- Mean decay length beta gamma c tau = (p / m) c tau: K0_S (c tau = 2.6844 cm,
  m = 0.497611) at 3 GeV: 16.18 cm.
- Z mass 91.1876 GeV, width 2.4952 GeV; K0_S mass 0.497611 GeV.
- Three-body phase space: the mass of two of the daughters is distributed as
  p*(M; m12, m3) p*(m12; m1, m2) (PDG eq. 49.20 integrated over angles).
- Material (PDG, "Passage of particles through matter", and the PDG's tables
  of atomic and nuclear properties): values quoted in the tests that use them.
- Higgs mass 125.25 GeV.
- Branching fractions: PDG particle listings, quoted in BRANCHING below.
- Momentum resolution of a tracker: Gluckstern 1963 / PDG "Particle detectors
  at accelerators", eqs. 35.63 to 35.65, quoted in the test that uses it.
- CMS open data (CERN Open Data Portal, record 545): values read from the
  files themselves, stated in the tests.
- Custom events (particle_custom.py), tests at the end of the file. PDG
  values: K0_S mass 0.497611 GeV, c tau 2.6844 cm, pi+ pi- 69.20%, pi0 pi0
  30.69%; Lambda mass 1.115683 GeV, c tau 7.89 cm; J/psi mass 3.0969 GeV;
  Z mass 91.1876 GeV, width 2.4952 GeV, e e : mu mu : tau tau = 3.3632 :
  3.3662 : 3.3696 %; tau mass 1.77686 GeV; Higgs mass 125.25 GeV. Formulas:
  mean decay length (p / m) c tau; radius p_T / (0.299792458 q B).
"""

import csv
import threading
import time

import numpy as np
import pytest

from physengine.modules import particle_custom as custom
from physengine.modules import particle_opendata
from physengine.modules.particle_custom import Entry, Setup
from physengine.modules.particle_data import A1, RHO
from physengine.modules.particle_data import PARTICLES as P
from physengine.modules.particle_events import (
    SCENES,
    Event,
    Hist,
    MassHistogram,
    _decay_time,
    _produce,
    bin_index,
    decay_momenta,
    generate,
    lever_arm,
    measured,
    momentum_resolution,
    quick_masses,
    vertex_pairs,
    visible_tracks,
    z_pair_masses,
)
from physengine.modules.particle_kinematics import (
    boost,
    breit_wigner_mass,
    four_momentum,
    leptonic_decay,
    mass,
    phase_space_decay,
    tau_decay_via_a1,
    tau_decay_via_rho,
    two_body_decay,
    two_body_momentum,
)
from physengine.modules.particle_material import (
    COIL,
    ECAL,
    HCAL,
    TRACKER,
    YOKE,
    bethe,
    em_shower,
    scattering_angle,
    scattering_step,
)
from physengine.modules.particle_tracks import (
    MAX_STEPS,
    C,
    helix_radius,
    helix_step,
    propagate,
)


@pytest.fixture(autouse=True)
def _default_custom_setup():
    # The Custom scene's setup is kept outside the module instances: every
    # test starts from, and leaves behind, the default one.
    custom.reset()
    try:
        yield
    finally:
        custom.reset()


def test_two_body_momenta():
    assert two_body_momentum(P["K0S"].mass, P["pi+"].mass, P["pi-"].mass) == pytest.approx(
        0.206, abs=5e-4
    )
    assert two_body_momentum(P["Lambda"].mass, P["p"].mass, P["pi-"].mass) == pytest.approx(
        0.101, abs=5e-4
    )
    assert two_body_momentum(91.1876, P["mu-"].mass, P["mu+"].mass) == pytest.approx(
        45.594, abs=1e-3
    )


@pytest.mark.parametrize(
    "parent, daughters",
    [
        ("K0S", ("pi+", "pi-")),
        ("tau-", ("mu-", "nu", "nu")),
        ("tau-", ("pi-", "pi+", "pi-", "nu")),
        ("tau-", ("pi-", "pi+", "pi-", "pi0", "pi0", "nu")),
        ("K+", ("pi0", "e+", "nu")),
    ],
)
def test_phase_space_decays_conserve_four_momentum(parent, daughters):
    rng = np.random.default_rng(1)
    p4 = four_momentum(P[parent].mass, [3.0, -1.0, 7.0])
    for _ in range(200):
        out = phase_space_decay(p4, [P[d].mass for d in daughters], rng)
        np.testing.assert_allclose(sum(out), p4, atol=1e-9)
        for q, d in zip(out, daughters, strict=True):
            # Mass shell: E^2 - p^2 = m^2 (compared as m^2: a massless
            # particle's m from rounding alone is ~1e-7).
            assert q[0] ** 2 - q[1:] @ q[1:] == pytest.approx(P[d].mass ** 2, abs=1e-9)


def _check_decay(out, parent, masses):
    """Energy and momentum conserved (sum of the daughters = parent, to rounding)
    and every daughter on its mass shell, E^2 - p^2 = m^2."""
    np.testing.assert_allclose(sum(out), parent, atol=1e-9)
    for q, m in zip(out, masses, strict=True):
        assert q[0] ** 2 - q[1:] @ q[1:] == pytest.approx(m * m, abs=1e-9)
        assert q[0] >= 0.0


def test_leptonic_and_resonant_decays_conserve_four_momentum():
    # Conservation laws only; the parent is a moving tau (m = 1.77686 GeV, PDG).
    rng = np.random.default_rng(21)
    tau = four_momentum(P["tau-"].mass, [3.0, -1.0, 7.0])
    m_mu, m_pi, m_pi0 = P["mu-"].mass, P["pi-"].mass, P["pi0"].mass
    for _ in range(200):
        _check_decay(leptonic_decay(tau, m_mu, rng), tau, [m_mu, 0.0, 0.0])
        _check_decay(tau_decay_via_rho(tau, m_pi, m_pi0, RHO, rng), tau, [m_pi, m_pi0, 0.0])
        out = tau_decay_via_a1(tau, m_pi, m_pi, m_pi, RHO, A1, rng)
        _check_decay(out, tau, [m_pi, m_pi, m_pi, 0.0])
    # Every decay the tables can ask for, through the function events use.
    for name in ("tau-", "tau+", "K+", "K-", "K0S", "Lambda", "pi0", "pi+"):
        parent = four_momentum(P[name].mass, [0.4, 2.0, -1.0])
        for daughters in P[name].modes()[1]:
            out = decay_momenta(name, parent, daughters, rng)
            _check_decay(out, parent, [P[d].mass for d in daughters])


def test_tau_decays_go_through_the_rho_and_the_a1():
    # tau -> pi pi0 nu is dominated by the rho(770): mass 775 MeV, width 149
    # MeV (PDG). Half of a Breit-Wigner lies within +-Gamma/2 = 75 MeV of the
    # pole; flat phase space would put 13% there, with a median of 1.03 GeV.
    # tau -> 3 pi nu goes through the a1(1260) (PDG: mass 1230 +- 40 MeV) to
    # rho pi; flat phase space would give a three-pion median of 1.32 GeV
    # and no rho in the pair.
    rng = np.random.default_rng(22)
    tau = four_momentum(P["tau-"].mass, [0.0, 0.0, 0.0])
    m_pi, m_pi0 = P["pi-"].mass, P["pi0"].mass
    pairs = np.array(
        [float(mass(sum(tau_decay_via_rho(tau, m_pi, m_pi0, RHO, rng)[:2]))) for _ in range(3000)]
    )
    assert np.median(pairs) == pytest.approx(0.775, abs=0.025)
    assert np.mean(np.abs(pairs - 0.775) < 0.075) > 0.45
    out = [tau_decay_via_a1(tau, m_pi, m_pi, m_pi, RHO, A1, rng) for _ in range(3000)]
    three = np.array([float(mass(sum(q[:3]))) for q in out])
    rho = np.array([float(mass(sum(q[:2]))) for q in out])
    assert 1.05 < np.median(three) < 1.30
    assert np.all(three < P["tau-"].mass)
    assert np.median(rho) == pytest.approx(0.775, abs=0.04)
    # And events use them: the mode's daughters come back in the table's order.
    out = decay_momenta("tau+", tau, ("pi+", "pi0", "nu"), rng)
    assert [round(float(abs(mass(q))), 3) for q in out] == [0.14, 0.135, 0.0]


# Branching fractions from the PDG particle listings (Review of Particle
# Physics 2022 and its 2023 update), as fractions. The tau's are the fit
# values with K0 excluded from the pion modes; pi K0bar nu (0.838%) is half
# K0_S and half K0_L. The six K+- modes are those of the PDG's constrained
# fit and add up to 99.995%.
BRANCHING = {
    "tau-": {
        ("mu-", "nu", "nu"): 0.1739,
        ("e-", "nu", "nu"): 0.1782,
        ("pi-", "nu"): 0.1082,
        ("K-", "nu"): 0.00696,
        ("pi-", "pi0", "nu"): 0.2549,
        ("K-", "pi0", "nu"): 0.00433,
        ("pi-", "pi0", "pi0", "nu"): 0.0926,
        ("pi-", "pi0", "pi0", "pi0", "nu"): 0.0104,
        ("pi-", "pi+", "pi-", "nu"): 0.0931,
        ("pi-", "pi+", "pi-", "pi0", "nu"): 0.0462,
        ("pi-", "K0S", "nu"): 0.00419,
        ("pi-", "K0L", "nu"): 0.00419,
    },
    "tau+": {("mu+", "nu", "nu"): 0.1739, ("pi+", "pi0", "nu"): 0.2549},
    "K+": {
        ("mu+", "nu"): 0.6356,
        ("pi+", "pi0"): 0.2067,
        ("pi+", "pi+", "pi-"): 0.05583,
        ("pi0", "e+", "nu"): 0.0507,
        ("pi0", "mu+", "nu"): 0.03352,
        ("pi+", "pi0", "pi0"): 0.01760,
    },
    "K-": {("mu-", "nu"): 0.6356, ("pi-", "pi0"): 0.2067},
    "K0S": {("pi+", "pi-"): 0.6920, ("pi0", "pi0"): 0.3069},
    "Lambda": {("p", "pi-"): 0.639, ("n", "pi0"): 0.358},
    "pi0": {("gamma", "gamma"): 0.98823, ("e+", "e-", "gamma"): 0.01174},
    "pi+": {("mu+", "nu"): 0.999877, ("e+", "nu"): 0.000123},
}


def test_decays_are_drawn_with_the_pdg_branching_fractions():
    rng = np.random.default_rng(23)
    for name, table in BRANCHING.items():
        probabilities, modes = P[name].modes()
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)
        assert min(probabilities) > 0.0
        for daughters, fraction in table.items():
            # The probability used for the draw is the PDG fraction itself
            # (to the 5e-5 by which the K and pi0 tables miss one).
            assert probabilities[modes.index(daughters)] == pytest.approx(fraction, rel=1e-4)
        for daughters in modes:  # every mode, the stand-in for the rest included
            assert sum(P[d].charge for d in daughters) == P[name].charge
            assert sum(P[d].mass for d in daughters) < P[name].mass
    # Every particle that decays in flight has a table that can be drawn from.
    for particle in P.values():
        if particle.decays and particle.width == 0.0:
            assert sum(particle.modes()[0]) == pytest.approx(1.0, abs=1e-12)
    # The draw itself, as events make it (rng.choice over modes()): 200000
    # tau decays; the binomial error on a fraction f is sqrt(f (1 - f) / n).
    probabilities, modes = P["tau-"].modes()
    n = 200000
    drawn = np.bincount(rng.choice(len(modes), p=probabilities, size=n), minlength=len(modes)) / n
    for daughters, fraction in BRANCHING["tau-"].items():
        error = np.sqrt(fraction * (1.0 - fraction) / n)
        assert drawn[modes.index(daughters)] == pytest.approx(fraction, abs=4.0 * error)


def test_kaons_in_events_decay_with_the_pdg_fractions():
    # End to end: K+ at rest decay at once; count what they decayed to.
    # PDG: mu nu 63.56%, pi pi0 20.67% (before the fix the first was drawn
    # at 75.5%). 600 decays: errors 2.0% and 1.7%; 4 sigma allowed.
    rng = np.random.default_rng(24)
    counts, n = {}, 600
    for _ in range(n):
        event = Event()
        kaon = four_momentum(P["K+"].mass, [0.0, 0.0, 0.0])
        _produce(event, "K+", kaon, np.zeros(3), 0.0, 3.8, rng)
        children = tuple(t.name for t in event.tracks if t.parent == 0 and t.origin == "decay")
        counts[children] = counts.get(children, 0) + 1
        # Energy and momentum conserved in the decay actually made.
        total = sum(t.p4 for t in event.tracks if t.parent == 0 and t.origin == "decay")
        np.testing.assert_allclose(total, kaon, atol=1e-9)
    assert counts[("mu+", "nu")] / n == pytest.approx(0.6356, abs=0.079)
    assert counts[("pi+", "pi0")] / n == pytest.approx(0.2067, abs=0.066)
    assert set(counts) <= set(BRANCHING["K+"])


def test_data_values():
    # PDG: pi0 mean life 8.43e-17 s, so c tau = 8.43e-17 * 2.998e8 = 25.3 nm.
    assert P["pi0"].ctau == pytest.approx(25.3e-9, rel=2e-3)
    # The lightest hadron is the neutral pion (134.98 MeV against 139.57).
    assert P["pi0"].mass < P["pi+"].mass == P["pi-"].mass


def test_boost_keeps_the_mass_and_dilates_energy():
    # A pion at rest seen from a frame where its parent moves with gamma = 5
    # along x: E = gamma m, p = gamma beta m.
    parent = four_momentum(1.0, [np.sqrt(24.0), 0.0, 0.0])  # gamma = 5
    pion = boost(four_momentum(P["pi+"].mass, [0.0, 0.0, 0.0]), parent)
    assert pion[0] == pytest.approx(5.0 * P["pi+"].mass)
    assert pion[1] == pytest.approx(np.sqrt(24.0) * P["pi+"].mass)
    assert float(mass(pion)) == pytest.approx(P["pi+"].mass)


def test_three_body_phase_space_distribution():
    rng = np.random.default_rng(2)
    M, m = 1.0, 0.1
    parent = four_momentum(M, [0.0, 0.0, 0.0])
    m12 = np.array(
        [float(mass(sum(phase_space_decay(parent, [m, m, m], rng)[:2]))) for _ in range(20000)]
    )
    x = np.linspace(2 * m, M - m, 4001)[1:-1]
    w = np.array([two_body_momentum(M, v, m) * two_body_momentum(v, m, m) for v in x])
    mean = (x * w).sum() / w.sum()
    spread = np.sqrt((x * x * w).sum() / w.sum() - mean**2)
    assert m12.mean() == pytest.approx(mean, abs=4 * spread / np.sqrt(len(m12)))
    assert m12.std() == pytest.approx(spread, rel=0.03)


def test_helix_radius_and_sense():
    # 1 GeV transverse momentum in 1 T: R = 1 / 0.2998 = 3.336 m.
    assert helix_radius(1.0, 1, 1.0) == pytest.approx(3.3356, abs=1e-4)
    p = four_momentum(P["mu+"].mass, [1.0, 0.0, 0.3])
    pos, points = np.zeros(3), []
    for _ in range(4000):
        pos, p = helix_step(pos, p, +1, 1.0, 0.05)
        points.append(pos.copy())
    points = np.array(points)
    assert (points[:, 1].max() - points[:, 1].min()) / 2 == pytest.approx(3.3356, abs=1e-3)
    assert np.linalg.norm(p[1:]) == pytest.approx(np.sqrt(1.0 + 0.09))  # |p| unchanged
    assert p[3] == pytest.approx(0.3)  # p_z unchanged
    # dp/dt = q v x B: a positive charge moving along +x in B along +z is
    # pushed towards -y (clockwise seen from +z).
    assert points[10, 1] < 0.0


def test_mean_decay_length():
    rng = np.random.default_rng(3)
    p4 = four_momentum(P["K0S"].mass, [3.0, 0.0, 0.0])
    speed = 3.0 / p4[0] * C
    lengths = [_decay_time(P["K0S"], p4, rng) * speed for _ in range(20000)]
    assert np.mean(lengths) == pytest.approx(3.0 / P["K0S"].mass * P["K0S"].ctau, rel=0.03)


def test_breit_wigner_width():
    # Half of a Breit-Wigner lies within +-Gamma/2 of the pole; cut at
    # +-10 Gamma that becomes atan(1) / atan(20) = 0.516.
    rng = np.random.default_rng(4)
    m = np.array([breit_wigner_mass(91.1876, 2.4952, rng) for _ in range(40000)])
    assert np.median(m) == pytest.approx(91.1876, abs=0.03)
    assert np.mean(np.abs(m - 91.1876) < 1.2476) == pytest.approx(0.516, abs=0.01)


def test_z_peak_is_reconstructed():
    # Z mass 91.1876 GeV, width 2.4952 GeV (PDG). The masses the Z bosons are
    # made with follow a Breit-Wigner cut at +-10 widths, whose central 68%
    # has the half-width (Gamma / 2) tan(0.68 atan(20)) = 1.2476 * 1.681 =
    # 2.10 GeV. The measured peak sits at the same place and is wider, by
    # the tracker's resolution (about 1% per muon at 3.8 T, more for muons
    # near the beam line: see the next test for the resolution itself).
    rng = np.random.default_rng(5)
    true, found = [], []
    for _ in range(4000):
        event = generate("Z -> mu+ mu-", 3.8, rng, quick=True)
        true.append(float(mass(event.bosons[0]["p4"])))
        found += [m for _, m in event.masses]
    half = lambda m: 0.5 * (np.percentile(m, 84) - np.percentile(m, 16))  # noqa: E731
    assert np.median(true) == pytest.approx(91.1876, abs=0.1)
    assert half(true) == pytest.approx(2.10, abs=0.12)
    assert np.median(found) == pytest.approx(91.1876, abs=0.15)
    assert half(true) < half(found) < half(true) + 1.0
    # Muons closer to the beam line than the tracker reaches give no mass:
    # most events do, not all.
    assert 0.8 < len(found) / 4000 < 0.97


def _gluckstern(pt, field, length=1.2, layers=13):
    """sigma(p_T) / p_T of a fast central track, written out independently of the
    module (PDG eqs. 35.63 to 35.65): 35 micrometre points, X0 = 2.77 m."""
    points = 35.0e-6 * pt / (0.299792458 * field * length**2) * np.sqrt(720.0 / (layers + 4))
    scattering = 0.016 / (0.299792458 * field * length) * np.sqrt(length / 2.77)
    return np.hypot(points, scattering)


def test_momentum_resolution_follows_the_field():
    # Gluckstern's formula for a 45 GeV muon across the field (cos(lambda) = 1,
    # beta = 1) in 3.8 T, with L = 1.2 m, N = 13, eps = 35 micrometres and
    # X0 = 2.77 m, by hand:
    #   points:     35e-6 * 45 / (0.29979 * 3.8 * 1.44) * sqrt(720 / 17) = 6.248e-3
    #   scattering: 0.016 / (0.29979 * 3.8 * 1.2) * sqrt(1.2 / 2.77)     = 7.704e-3
    #   together:   sqrt(6.248^2 + 7.704^2) e-3                          = 9.92e-3
    m_mu = P["mu-"].mass
    assert lever_arm(np.zeros(3), np.array([45.0, 0.0, 0.0]), -1, 3.8) == pytest.approx(1.2)
    assert momentum_resolution(45.0, 45.0, m_mu, 1.2, 3.8) == pytest.approx(9.92e-3, rel=2e-3)
    # Both parts go as 1 / B.
    assert momentum_resolution(45.0, 45.0, m_mu, 1.2, 1.9) == pytest.approx(2 * 9.92e-3, rel=2e-3)
    # A track leaving through the tracker's end (half length 2.8 m) at
    # eta = 2.5 has L = 2.8 / sinh(2.5) = 0.463 m; beyond eta = 3 fewer than
    # three layers are crossed and nothing is measured.
    forward = np.array([1.0, 0.0, np.sinh(2.5)])
    assert lever_arm(np.zeros(3), forward, 1, 3.8) == pytest.approx(0.463, abs=1e-3)
    far = np.array([1.0, 0.0, np.sinh(3.5)])
    length = lever_arm(np.zeros(3), far, 1, 3.8)
    assert momentum_resolution(1.0, np.cosh(3.5), m_mu, length, 3.8) is None
    # A tight circle is seen over at most its diameter: 0.1 GeV in 3.8 T has
    # R = 0.1 / (0.29979 * 3.8) = 0.0878 m.
    assert lever_arm(np.zeros(3), np.array([0.1, 0.0, 0.0]), 1, 3.8) == pytest.approx(
        0.1756, abs=1e-4
    )

    # The masses of generated events carry exactly this resolution: for a
    # pair of central muons (|eta| < 1: the whole radius is crossed) the
    # mass error is (sigma_1 (+) sigma_2) / 2 relative, each sigma from the
    # formula above (with 1 / cos(lambda) = cosh(eta) in the scattering).
    # The spread of (measured / true - 1) / expected must then be 1, at
    # both fields; lowering the field from 3.8 to 1 T widens the peak 3.8
    # times.
    rng = np.random.default_rng(25)
    spread = {}
    for field in (3.8, 1.0):
        pulls, errors = [], []
        while len(pulls) < 1500:
            event = generate("Z -> mu+ mu-", field, rng, quick=True)
            eta = [np.arcsinh(t.p4[3] / np.hypot(t.p4[1], t.p4[2])) for t in event.tracks]
            if max(np.abs(eta)) > 1.0 or not event.masses:
                continue
            sigma = []
            for t, e in zip(event.tracks, eta, strict=True):
                pt = np.hypot(t.p4[1], t.p4[2])
                points = 35.0e-6 * pt / (0.299792458 * field * 1.44) * np.sqrt(720.0 / 17.0)
                scattering = 0.016 / (0.299792458 * field * 1.2) * np.sqrt(1.2 * np.cosh(e) / 2.77)
                sigma.append(np.hypot(points, scattering))
            expected = 0.5 * np.hypot(*sigma)
            residual = event.masses[0][1] / float(mass(event.bosons[0]["p4"])) - 1.0
            pulls.append(residual / expected)
            errors.append(residual)
        # Central 68% (a few curvature measurements land far out).
        spread[field] = 0.5 * (np.percentile(errors, 84) - np.percentile(errors, 16))
        assert 0.5 * (np.percentile(pulls, 84) - np.percentile(pulls, 16)) == pytest.approx(
            1.0, abs=0.07
        )
    assert spread[1.0] / spread[3.8] == pytest.approx(3.8, rel=0.1)
    assert _gluckstern(45.0, 3.8) == pytest.approx(9.92e-3, rel=2e-3)


def test_k_short_is_reconstructed_from_its_pions():
    # K0_S mass 0.497611 GeV (PDG). Every pair of pions from a K0_S decay
    # vertex is used, identified by its parent and not by its mass: with the
    # true momenta at the vertex the pair has exactly the K0_S mass (energy
    # and momentum conservation, wherever in the field the decay happened);
    # as measured, the peak is centred there within the tracker's resolution.
    rng = np.random.default_rng(6)
    m_pi = P["pi+"].mass
    true, found, entries = [], [], 0
    for _ in range(120):
        event = generate("Strange particles", 3.8, rng)
        entries += len(event.masses)
        for parent, a, b in vertex_pairs(event):
            if event.tracks[parent].name != "K0S" or {a.name, b.name} != {"pi+", "pi-"}:
                continue
            true.append(float(mass(a.p4 + b.p4)))
            momenta = [measured(t, 3.8, rng) for t in (a, b)]
            if all(q is not None for q in momenta):
                found.append(float(mass(sum(four_momentum(m_pi, q) for q in momenta))))
    assert len(found) > 100 and entries >= len(found)
    np.testing.assert_allclose(true, 0.497611, atol=1e-9)
    assert np.median(found) == pytest.approx(0.497611, abs=0.001)
    assert 0.5 * (np.percentile(found, 84) - np.percentile(found, 16)) < 0.008


def test_tau_visible_mass_is_bounded_by_the_true_mass():
    # Z -> tau tau: the neutrinos carry energy away, so the visible decay
    # products have less energy than the Z and a smaller mass (the missing
    # four-momentum is a sum of light-like vectors). Each visible particle
    # must be counted once: not together with the pair it converted to or the
    # photons it radiated (that double counting gave masses above the Z's).
    rng = np.random.default_rng(26)
    conversions = 0
    for _ in range(60):
        event = generate("Z -> tau+ tau-", 3.8, rng)
        z = event.bosons[0]["p4"]
        seen = visible_tracks(event)
        indices = {id(t): i for i, t in enumerate(event.tracks)}
        chosen = {indices[id(t)] for t in seen}
        for t in seen:  # no ancestor of a visible particle is itself visible
            parent = t.parent
            while parent >= 0:
                assert parent not in chosen
                parent = event.tracks[parent].parent
        conversions += sum(t.end == "conversion" for t in seen)
        total = sum(t.p4 for t in seen)
        assert total[0] <= z[0] * (1.0 + 1e-9)
        assert float(mass(total)) <= float(mass(z)) * (1.0 + 1e-9)
        # As measured (charged momenta smeared by about 1%): still below,
        # with room for the resolution.
        for _, m in event.masses:
            assert m < 1.1 * float(mass(z))
    assert conversions > 0  # the case that used to be counted twice did occur


GENERATED = [name for name in SCENES if not name.startswith("Real data")]
needs_data = pytest.mark.skipif(
    not all(particle_opendata.available(name) for name in particle_opendata.FILES),
    reason="CMS open data not downloaded (python -m physengine.modules.particle_opendata)",
)


@pytest.mark.parametrize("scene", GENERATED)
def test_events_conserve_energy(scene):
    # Energy is only moved around: every particle ends by decaying or
    # converting (its energy goes to its daughters), or by showering,
    # stopping or leaving (it keeps what it has at the end); on the way it
    # leaves some by ionisation, and an electron gives some to photons,
    # which are particles of the event themselves. So the energy of what
    # came out of the collision equals the ionisation of all tracks plus
    # the final energy of every track that did not decay or convert.
    rng = np.random.default_rng(7)
    for _ in range(3 if scene == "Busy collision" else 8):
        event = generate(scene, 3.8, rng)
        primary = sum(t.p4[0] for t in event.tracks if t.parent < 0)
        ionisation = sum(t.ionisation for t in event.tracks)
        final = sum(t.p4_end[0] for t in event.tracks if t.end not in ("decay", "conversion"))
        assert ionisation + final == pytest.approx(primary, rel=1e-9)
        for t in event.tracks:
            if t.origin == "decay":
                assert event.tracks[t.parent].end == "decay"
                np.testing.assert_allclose(t.points[0], event.tracks[t.parent].points[-1])


def test_leptonic_decay_has_the_michel_spectrum():
    # mu -> e nu nu (or tau -> l nu nu) with the V - A matrix element: the
    # charged lepton's energy follows x^2 (3 - 2 x), x = 2 E / M, whose mean
    # is 7/10, i.e. <E> = 7 M / 20 (phase space alone gives M / 3), and whose
    # fraction above x = 0.5 is integral_0.5^1 2 x^2 (3 - 2 x) dx = [2 x^3 - x^4] = 0.8125.
    rng = np.random.default_rng(9)
    M = P["tau-"].mass
    parent = four_momentum(M, [0.0, 0.0, 0.0])
    x = np.array([2.0 * leptonic_decay(parent, 0.0, rng)[0][0] / M for _ in range(20000)])
    assert x.mean() == pytest.approx(0.7, abs=0.005)
    assert np.mean(x > 0.5) == pytest.approx(0.8125, abs=0.01)


def test_ionisation_loss():
    # Minimum ionisation of muons, from the PDG's tables of atomic and nuclear
    # properties (pdg.lbl.gov/AtomicNuclearProperties) [MeV cm^2 / g]:
    # silicon 1.664, lead tungstate 1.229, copper 1.403, aluminium 1.615,
    # iron 1.451. The tracker is taken as silicon and the hadron
    # calorimeter's brass as copper. (Before the density effect was
    # Sternheimer's, iron came out at 1.503.)
    reference = ((TRACKER, 1.664), (ECAL, 1.229), (HCAL, 1.403), (COIL, 1.615), (YOKE, 1.451))
    momenta = np.geomspace(0.15, 1.5, 400)  # the minimum is near beta gamma = 3.5
    for material, minimum in reference:
        loss = [bethe(material, P["mu-"].mass, p) * 10.0 / material.density for p in momenta]
        assert min(loss) == pytest.approx(minimum, rel=0.005)
        assert 0.25 < momenta[int(np.argmin(loss))] < 0.5
    # Lead tungstate: 1.229 MeV cm^2/g at rho = 8.30 is 10.20 MeV/cm.
    assert bethe(ECAL, P["mu-"].mass, 0.32) * 10.0 == pytest.approx(10.20, abs=0.05)
    # Slow particles lose more: 1 / beta^2.
    assert bethe(ECAL, P["p"].mass, 0.3) > 5.0 * bethe(ECAL, P["p"].mass, 3.0)


def test_scattering_angle():
    # Highland (PDG eq. 34.16): theta_0 = 13.6 MeV / (beta p) sqrt(t) (1 + 0.038 ln(t / beta^2)),
    # t = x / X0. By hand for a 1 GeV pion (m = 0.13957: beta = 0.9904) through
    # the whole tracker, 1.2 m at X0 = 2.77 m, t = 0.4332:
    #   13.6e-3 / 0.9904 * sqrt(0.4332) * (1 + 0.038 ln(0.4332 / 0.9809))
    #   = 13.732e-3 * 0.6582 * 0.96894 = 8.757 mrad.
    assert scattering_angle(TRACKER, 1.2, 1.0, 0.9904) == pytest.approx(8.757e-3, rel=2e-3)
    # In steps: the variances of 40 steps of 3 cm add up to that of the whole
    # (the logarithm is of the total thickness, not of each step) ...
    variance, crossed = 0.0, 0.0
    for _ in range(40):
        variance += scattering_step(crossed, crossed + 3.0 / 277.0, 1.0, 0.9904) ** 2
        crossed += 3.0 / 277.0
    assert np.sqrt(variance) == pytest.approx(8.757e-3, rel=2e-3)
    # ... whereas the formula applied to each 3 cm step on its own gives
    # sqrt(40) * 13.732e-3 * sqrt(0.01083) * (1 + 0.038 ln(0.01104)) = 7.49 mrad,
    # 15% too little (what the module did before).
    naive = np.sqrt(40.0) * scattering_angle(TRACKER, 0.03, 1.0, 0.9904)
    assert naive == pytest.approx(7.49e-3, rel=5e-3)


def test_scattering_adds_up_along_a_track():
    # The same number from tracks actually followed: 1 GeV pions through the
    # tracker without field; the direction at its edge (r = 1.2 m), per
    # plane, has the spread 8.76 mrad (previous test; the 2% of momentum
    # lost on the way adds 1%). 1200 tracks, two planes: 1.4% statistical error.
    rng = np.random.default_rng(27)
    angles = []
    for _ in range(1200):
        p4 = four_momentum(P["pi+"].mass, [1.0, 0.0, 0.0])
        track = propagate("pi+", "hadron", 1, p4, np.zeros(3), 0.0, 0.0, None, rng)[0]
        k = np.flatnonzero(np.hypot(track.points[:, 0], track.points[:, 1]) < 1.2)[-1]
        d = track.points[k] - track.points[k - 1]
        angles += [np.arctan2(d[1], d[0]), np.arctan2(d[2], d[0])]
    assert np.std(angles) == pytest.approx(8.76e-3, rel=0.06)


def test_tracks_always_end():
    rng = np.random.default_rng(28)
    start = np.zeros(3)
    # A four-momentum that is not a number, infinite, or zero has no speed:
    # following it used never to return. It is refused.
    for bad in ([np.nan, 1.0, 0.0, 0.0], [np.inf, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                [1.0, np.nan, 0.0, 0.0], [1.0, 5.0, 0.0, 0.0]):  # fmt: skip
        with pytest.raises(ValueError):
            propagate("mu-", "muon", -1, np.array(bad), start, 0.0, 3.8, None, rng)
    with pytest.raises(ValueError):
        p4 = four_momentum(P["mu-"].mass, [1.0, 0.0, 0.0])
        propagate("mu-", "muon", -1, p4, np.array([np.nan, 0.0, 0.0]), 0.0, 3.8, None, rng)
    # A particle curling for ever in the field (50 MeV: radius 4 cm) is
    # dropped after a bounded number of steps.
    p4 = four_momentum(P["pi+"].mass, [0.05, 0.0, 0.0])
    track = propagate("pi+", "hadron", 1, p4, start, 0.0, 3.8, None, rng, material=False)[0]
    assert track.end == "lost" and len(track.points) <= MAX_STEPS + 1
    # One at rest ends at once.
    p4 = four_momentum(P["mu-"].mass, [0.0, 0.0, 0.0])
    assert propagate("mu-", "muon", -1, p4, start, 0.0, 3.8, None, rng)[0].end == "stopped"


def test_electromagnetic_shower_profile():
    # 45 GeV electron in lead tungstate (E_c = 9.64 MeV, X0 = 0.8903 cm):
    # maximum at ln(E / E_c) - 0.5 = 7.95 radiation lengths = 7.1 cm.
    shower = em_shower(ECAL, np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0, 45.0, False)
    assert shower.depth_max == pytest.approx(0.0708, abs=0.001)
    depths, weights = shower.profile(40)
    assert weights.sum() == pytest.approx(1.0)
    assert depths[np.argmax(weights)] == pytest.approx(shower.depth_max, abs=0.01)


def test_muons_need_a_few_gev_to_cross_the_iron():
    # CMS: a muon needs about 3 to 5 GeV to reach the outer muon stations.
    rng = np.random.default_rng(10)

    def fate(pt):
        p4 = four_momentum(P["mu-"].mass, [pt, 0.0, 0.0])
        return propagate("mu-", "muon", -1, p4, np.zeros(3), 0.0, 3.8, None, rng)[0]

    assert fate(2.5).end == "stopped"
    through = fate(10.0)
    assert through.end == "exit" and len(through.hits) == 4
    assert 3.0 < through.ionisation < 5.5


def test_electrons_are_measured_through_their_radiation():
    # Electrons radiate in the tracker; the calorimeter collects the
    # electron and its photons, so the Z still comes out at its mass.
    rng = np.random.default_rng(11)
    masses = np.array([generate("Z -> e+ e-", 3.8, rng).masses[0][1] for _ in range(120)])
    assert np.median(masses) == pytest.approx(91.1876, abs=0.6)


def test_higgs_to_four_leptons():
    # H -> Z Z*, Higgs mass 125.25 GeV < 2 * 91.19: one Z must be off its mass
    # shell. The two masses follow (Djouadi 2008, Phys. Rep. 457, 1, eq. 2.31)
    #   BW(q1^2) BW(q2^2) sqrt(lam) (lam + 12 q1^2 q2^2 / M_H^4) dq1^2 dq2^2,
    #   lam = (1 - q1^2/M_H^2 - q2^2/M_H^2)^2 - 4 q1^2 q2^2 / M_H^4,
    #   BW(q^2) = 1 / ((q^2 - M_Z^2)^2 + M_Z^2 Gamma_Z^2),
    # integrated here on a grid, independently of the sampler, above 12 GeV:
    # the means of the heavier and the lighter mass.
    mh, mz, gz = 125.25, 91.1876, 2.4952
    grid = np.linspace(12.0, mh - 12.0, 1500)
    m1, m2 = np.meshgrid(grid, grid, indexing="ij")
    q1, q2 = m1**2 / mh**2, m2**2 / mh**2
    lam = np.clip((1.0 - q1 - q2) ** 2 - 4.0 * q1 * q2, 0.0, None) * (m1 + m2 < mh)
    bw = lambda m: 1.0 / ((m * m - mz * mz) ** 2 + (mz * gz) ** 2)  # noqa: E731
    weight = bw(m1) * bw(m2) * np.sqrt(lam) * (lam + 12.0 * q1 * q2) * 4.0 * m1 * m2
    heavy = (np.maximum(m1, m2) * weight).sum() / weight.sum()
    light = (np.minimum(m1, m2) * weight).sum() / weight.sum()
    assert 80.0 < heavy < 92.0 and 20.0 < light < 35.0  # near the Z mass, and far below it
    rng = np.random.default_rng(12)
    pairs = np.array([z_pair_masses(rng) for _ in range(1500)])
    assert pairs[:, 0].mean() == pytest.approx(heavy, abs=4 * pairs[:, 0].std() / np.sqrt(1500))
    assert pairs[:, 1].mean() == pytest.approx(light, abs=4 * pairs[:, 1].std() / np.sqrt(1500))
    assert np.all(pairs.sum(axis=1) < mh) and np.all(pairs[:, 1] >= 12.0)

    # In events: the two decays in a row conserve energy and momentum (each
    # lepton pair adds up to its Z, the two Z to the Higgs), and the four
    # leptons as measured give the Higgs mass within the resolution (about
    # 1 GeV; before any smearing, exactly).
    found = []
    for _ in range(60):
        event = generate("Higgs -> 4 leptons", 3.8, rng)
        higgs, first, second = (b["p4"] for b in event.bosons)
        leptons = [t.p4 for t in event.tracks if t.parent < 0]
        assert len(leptons) == 4
        np.testing.assert_allclose(leptons[0] + leptons[1], first, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(leptons[2] + leptons[3], second, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(first + second, higgs, rtol=1e-9, atol=1e-9)
        assert float(mass(higgs)) == pytest.approx(125.25, abs=1e-6)
        assert float(mass(first)) + float(mass(second)) < 125.25
        found += [m for _, m in event.masses]
    assert len(found) > 30
    assert np.median(found) == pytest.approx(125.25, abs=0.6)
    assert 0.5 * (np.percentile(found, 84) - np.percentile(found, 16)) < 3.0


def test_two_body_decay_is_back_to_back_in_the_rest_frame():
    rng = np.random.default_rng(8)
    parent = four_momentum(P["K0S"].mass, [0.0, 0.0, 0.0])
    a, b = two_body_decay(parent, P["pi+"].mass, P["pi-"].mass, rng)
    np.testing.assert_allclose(a[1:], -b[1:], atol=1e-12)
    assert np.linalg.norm(a[1:]) == pytest.approx(0.20597, abs=1e-4)


def test_histogram_bins():
    # Linear: 60 bins from 60 to 120 GeV, 91.19 falls in bin 31. Logarithmic:
    # 120 bins from 2 to 120 GeV, bin = 120 ln(m / 2) / ln(60): the J/psi
    # (3.097) in bin 12, the Z in bin 111.
    linear = Hist("m", 60.0, 120.0, 60)
    assert bin_index(linear, 91.19) == 31
    assert bin_index(linear, 59.9) == -1 and bin_index(linear, 120.0) == -1
    logarithmic = Hist("m", 2.0, 120.0, 120, log=True)
    assert bin_index(logarithmic, 3.097) == 12
    assert bin_index(logarithmic, 91.19) == 111


def test_mass_histogram_statistics():
    # 20000 masses from a Gaussian at 91 +- 3 GeV, some outside the range:
    # the counts are those of bin_index, and the median and central-68%
    # half-width read off the counts agree with the sorted values to one
    # fine bin (60 GeV / 2400 = 0.025 GeV).
    rng = np.random.default_rng(29)
    hist = Hist("m", 60.0, 120.0, 60)
    values = np.concatenate([rng.normal(91.0, 3.0, 20000), [10.0, 500.0, 120.0]])
    stats = MassHistogram(hist)
    assert stats.summary() is None
    for chunk in np.array_split(values, 7):  # added a few at a time
        stats.add(chunk)
    expected = np.bincount([bin_index(hist, m) for m in values if bin_index(hist, m) >= 0])
    np.testing.assert_array_equal(stats.counts[: len(expected)], expected)
    inside = values[(values >= 60.0) & (values < 120.0)]
    count, median, half_width = stats.summary()
    assert stats.entries == len(values) and count == len(inside)
    assert median == pytest.approx(np.median(inside), abs=0.025)
    reference = 0.5 * (np.percentile(inside, 84) - np.percentile(inside, 16))
    assert half_width == pytest.approx(reference, abs=0.025)
    # Logarithmic bins: the J/psi (3.097) in bin 12 of 120 from 2 to 120 GeV.
    logarithmic = MassHistogram(Hist("m", 2.0, 120.0, 120, log=True))
    logarithmic.add([3.097, 3.097, 91.19])
    assert logarithmic.counts[12] == 2 and logarithmic.counts[111] == 1
    assert logarithmic.quantile(0.5) == pytest.approx(3.097, rel=0.002)


# One line of Zmumu.csv as CMS published it (the file's first event), for
# the tests of the reader. Its mass, by hand from
# m^2 = 2 pt1 pt2 (cosh(eta1 - eta2) - cos(phi1 - phi2)) (muon mass neglected):
# 2 * 54.7055 * 34.2464 * (cosh(0.5561) - cos(3.0729)) = 3746.9 * 2.15629 = 8079.4,
# m = 89.886 GeV.
ZMUMU_HEADER = "Run,Event,pt1,eta1,phi1,Q1,dxy1,iso1,pt2,eta2,phi2,Q2,dxy2,iso2"
ZMUMU_LINE = "165617,74969122,54.7055,-0.4324,2.5742,1,-0.0745,0.4999,34.2464,-0.9885,-0.4987,-1,0.0712,3.4221"


def test_data_files_with_bad_lines(tmp_path, monkeypatch):
    bad_lines = [
        "1,2,nan,0.1,0.2,1,0,0,30,0.3,-2,-1,0,0",  # not a number
        "1,2,40,0.1,0.2,1,0,0,inf,0.3,-2,-1,0,0",  # infinite
        "1,2,40,0.1",  # short
        "1,2,forty,0.1,0.2,1,0,0,30,0.3,-2,-1,0,0",  # text
        "1,2,40,0.1,0.2,0,0,0,30,0.3,-2,-1,0,0",  # a charge of zero
        "1,2,0,0.1,0.2,1,0,0,30,0.3,-2,-1,0,0",  # no momentum
        "1,2,40,0.1,0.2,1,0,0,30,900,-2,-1,0,0",  # sinh(900) overflows
        "",
    ]
    file = tmp_path / "Zmumu.csv"
    cases = {
        "empty file": "",
        "header only": ZMUMU_HEADER + "\n",
        "wrong header": "a,b,c\n1,2,3\n",
        "bad lines only": "\n".join([ZMUMU_HEADER, *bad_lines]) + "\n",
    }
    for text in cases.values():
        file.write_text(text)
        momenta, charges = particle_opendata.read_csv(file)
        assert momenta.shape == (0, 2, 4) and charges.shape == (0, 2)
    # The good line among the bad ones is the one event read.
    file.write_text("\n".join([ZMUMU_HEADER, *bad_lines[:4], ZMUMU_LINE, *bad_lines[4:]]) + "\n")
    momenta, charges = particle_opendata.read_csv(file)
    assert momenta.shape == (1, 2, 4) and charges.tolist() == [[1, -1]]
    assert np.all(np.isfinite(momenta))
    assert float(mass(momenta[0, 0] + momenta[0, 1])) == pytest.approx(89.886, abs=0.005)

    # Through the module's own entry points, with the data directory pointed
    # at these files: one usable event is shown; none is an error that says
    # so (it used to be an IndexError deep inside, or a loop without end).
    monkeypatch.setattr(particle_opendata, "DATA_DIR", tmp_path)
    monkeypatch.setattr(particle_opendata, "_cache", {})
    scene = "Real data: Z -> mu+ mu-"
    rng = np.random.default_rng(30)
    event = generate(scene, 3.8, rng)
    assert [t.name for t in event.tracks] == ["mu+", "mu-"]
    assert event.masses[0][1] == pytest.approx(89.886, abs=0.005)
    assert quick_masses(scene, 5, 3.8, rng) == pytest.approx([event.masses[0][1]] * 5)
    file.write_text(cases["bad lines only"])
    monkeypatch.setattr(particle_opendata, "_cache", {})
    with pytest.raises(ValueError, match="no usable events"):
        generate(scene, 3.8, rng)
    with pytest.raises(ValueError, match="no usable events"):
        quick_masses(scene, 5, 3.8, rng)


def test_module_survives_failures_and_fills_in_the_background(monkeypatch):
    # The module's event handling without a window: a failure while an event
    # is made (in the background thread or not) must leave a message and a
    # usable module, and events added in bulk must arrive without the caller
    # generating them.
    pytest.importorskip("imgui_bundle")
    pytest.importorskip("moderngl")
    from physengine.modules import particlemodule

    module = particlemodule.ParticleModule()  # "Z -> mu+ mu-"; no GL until setup()

    def collect(until, seconds=60.0):
        deadline = time.perf_counter() + seconds
        while not until() and time.perf_counter() < deadline:
            module._collect_fill()  # what step() does each frame
            time.sleep(0.005)
        assert until()

    try:
        module.new_event()
        assert len(module.event.tracks) == 2 and not module.event.error
        seen = module.events_seen

        def fail(*args, **kwargs):
            raise RuntimeError("injected")

        with monkeypatch.context() as patch:
            patch.setattr(particlemodule, "generate", fail)
            module.new_event()  # shows the event prepared before; prepares one that fails
            module.new_event()  # must return, with the failure as a message
            assert "injected" in module.event.error and not module.event.tracks
            assert particlemodule.narrate(module.event) == []
            module.worker.join()
        module.new_event()  # the failed one prepared under the patch
        module.new_event()
        assert len(module.event.tracks) == 2 and not module.event.error
        assert module.events_seen > seen

        # 300 events for the histogram are made by another thread.
        before = module.histogram.entries
        caller = threading.get_ident()
        made_by = set()
        real_quick = particlemodule.quick_masses

        def spy(*args, **kwargs):
            made_by.add(threading.get_ident())
            return real_quick(*args, **kwargs)

        monkeypatch.setattr(particlemodule, "quick_masses", spy)
        module.request_fill(300)
        assert module.to_fill == 300
        collect(lambda: module.to_fill == 0)
        assert made_by and caller not in made_by
        # 80 to 97% of Z -> mu mu events give a mass (see test_z_peak_is_reconstructed).
        assert 0.75 * 300 < module.histogram.entries - before <= 300
        assert module.histogram.summary()[1] == pytest.approx(91.19, abs=1.0)

        # A failure while filling: reported, the rest dropped, and a later fill works.
        monkeypatch.setattr(particlemodule, "quick_masses", fail)
        module.request_fill(50)
        collect(lambda: bool(module.fill_error))
        assert "injected" in module.fill_error and module.to_fill == 0
        monkeypatch.setattr(particlemodule, "quick_masses", real_quick)
        module.request_fill(20)
        collect(lambda: module.to_fill == 0)
        assert not module.fill_error
    finally:
        module.fill_cancel.set()
        module.cancel_fill()
        for thread in (module.worker, module.filler):
            if thread is not None:
                thread.join()


@needs_data
def test_recorded_events_show_the_known_resonances():
    # CMS open data (2011): the two-muon mass spectrum has peaks at the J/psi
    # (3.0969 GeV), the Upsilon (9.4603) and the Z (91.1876). The medians of
    # the pairs around each (measured: 3.093, 9.449, 90.85; the Z's is pulled
    # down by photon radiation off the muons and by the falling background
    # under it).
    rng = np.random.default_rng(13)
    m = np.array(quick_masses("Real data: all muon pairs", 100000, 3.8, rng))
    windows = ((3.0, 3.2, 3.0969, 0.01), (9.2, 9.7, 9.4603, 0.03), (86.0, 96.0, 91.1876, 0.6))
    for low, high, known, tolerance in windows:
        peak = m[(m > low) & (m < high)]
        assert len(peak) > 1000
        assert np.median(peak) == pytest.approx(known, abs=tolerance)


@needs_data
def test_recorded_momenta_give_the_masses_cms_published():
    # Dimuon_DoubleMu.csv has CMS's own invariant mass for each pair (column
    # M, four decimals): the masses computed here from the momenta in the
    # file must agree with it, line by line. (The momenta are given to four
    # decimals too, which limits the agreement to about 1e-3 relative.)
    scene = "Real data: all muon pairs"
    momenta, charges = particle_opendata.load(scene)
    with open(particle_opendata.path(scene), newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(momenta) == 100000  # nothing skipped in the real file
    published = np.array([float(row["M"]) for row in rows])
    computed = mass(momenta[:, 0] + momenta[:, 1])
    np.testing.assert_allclose(computed, published, rtol=2e-3, atol=2e-3)
    assert [int(row["Q1"]) for row in rows[:50]] == charges[:50, 0].tolist()
    # Zmumu.csv, first line, by hand (see ZMUMU_LINE): 89.886 GeV.
    momenta, charges = particle_opendata.load("Real data: Z -> mu+ mu-")
    assert float(mass(momenta[0, 0] + momenta[0, 1])) == pytest.approx(89.886, abs=0.005)
    assert charges[0].tolist() == [1, -1]


@needs_data
def test_recorded_pairs_are_labelled_by_their_charges():
    # Zmumu.csv: counted from the file's Q1 and Q2 columns, 336 of the 10000
    # pairs (3.36%) have two muons of the same charge (186 ++, 150 --).
    # Such an event is shown as what it is and kept out of the mu+ mu-
    # histogram; the other two files have opposite charges only.
    scene = "Real data: Z -> mu+ mu-"
    momenta, charges = particle_opendata.load(scene)
    same = ~particle_opendata.opposite_sign(charges)
    assert len(charges) == 10000 and int(same.sum()) == 336
    assert int((charges[same].sum(axis=1) == 2).sum()) == 186
    for other in ("Real data: J/psi -> mu+ mu-", "Real data: all muon pairs"):
        assert particle_opendata.opposite_sign(particle_opendata.load(other)[1]).all()
    rng = np.random.default_rng(14)
    kinds = set()
    for _ in range(150):  # 150 events: no same-sign pair among them has probability 0.6%
        event = generate(scene, 3.8, rng)
        names = sorted(t.name for t in event.tracks)
        label = event.masses[0][0]
        assert event.real and len(event.tracks) == 2 and not event.mechanism
        assert label.startswith(" ".join(names))
        assert event.in_histogram == (names == ["mu+", "mu-"])
        assert ("same charge" in event.headline) == (names[0] == names[1])
        kinds.add(tuple(names))
        # The tracks carry the recorded momenta and the charge's sense of
        # rotation: a positive charge along +x in B along +z turns to -y.
        for t in event.tracks:
            if len(t.points) > 5 and abs(t.p4[3]) < 5.0 * np.hypot(t.p4[1], t.p4[2]):
                turn = t.p4[1] * (t.points[5][1] - t.points[0][1]) - t.p4[2] * (
                    t.points[5][0] - t.points[0][0]
                )
                assert np.sign(turn) == -P[t.name].charge
    assert ("mu+", "mu-") in kinds and len(kinds) > 1
    # The histogram's supply leaves the same-sign pairs out: 96.64% remain.
    n = 20000
    assert len(quick_masses(scene, n, 3.8, rng)) / n == pytest.approx(0.9664, abs=0.006)


# ----------------------------------------------------------------- custom


def _decay_entry(name, daughters, value, **more):
    return Entry("decay", name, value=value, daughters=daughters, **more)


def _children(event, index):
    return [t for t in event.tracks if t.parent == index and t.origin == "decay"]


CUSTOM_DECAYS = [
    # (parent, mode, momentum [GeV], where it decays)
    ("K0S", "pi+ pi-", 5.0, "lifetime"),
    ("K0S", "pi0 pi0", 5.0, "start"),
    ("Lambda", "p pi-", 5.0, "start"),
    ("pi0", "gamma gamma", 2.0, "lifetime"),
    ("K+", "pi+ pi+ pi-", 3.0, "start"),
    ("K-", "mu- nu", 3.0, "start"),
    ("mu-", "e- nu nu", 0.0, "start"),
    ("tau-", "pi- pi0 nu", 45.0, "lifetime"),
    ("tau+", "pi+ pi- pi+ nu", 45.0, "lifetime"),
    ("J/psi", "mu+ mu-", 10.0, "lifetime"),
    ("Z", "e+ e-", 0.0, "lifetime"),
    ("Z", "tau+ tau-", 30.0, "lifetime"),
    ("H", "gamma gamma", 0.0, "lifetime"),
    ("H", "Z Z", 20.0, "lifetime"),
    ("K0S", "gamma gamma", 1.0, "start"),  # typed by hand: not a listed mode
    # Charm and beauty: the chain also covers the D from B+ -> D0bar pi+ decaying in turn.
    ("D0", "K- pi+", 3.5, "start"),
    ("B+", "D0bar pi+", 20.0, "lifetime"),
    ("Ds+", "K+ K- pi+", 5.0, "lifetime"),
    # D* (decay at once): the D*+ with its Breit-Wigner mass, and B decays
    # whose D* daughter decays in turn (B -> D* l nu with the CLN model).
    ("D*+", "D0 pi+", 5.0, "lifetime"),
    ("D*0", "D0 gamma", 5.0, "lifetime"),
    ("Ds*+", "Ds+ gamma", 5.0, "lifetime"),
    ("B+", "D*0bar e+ nu", 20.0, "lifetime"),
    ("B0", "D*- mu+ nu", 20.0, "lifetime"),
    ("B0", "D*- Ds*+", 20.0, "lifetime"),
]


@pytest.mark.parametrize("name,mode,momentum,where", CUSTOM_DECAYS)
def test_custom_decays_give_the_chosen_mode_and_conserve_four_momentum(name, mode, momentum, where):
    # The parent's first decay must be the mode asked for, and at every decay
    # in the chain the daughters' four-momenta must add up to the parent's
    # where it decays: its own mass (the mass it was made with, for a
    # resonance) and the momentum it has left. Tolerance: 1e-8 of the energy
    # (rounding in the boosts; exact arithmetic would give zero).
    rng = np.random.default_rng(61)
    typed = not any(" ".join(d) == mode for _, d in custom.modes_of(name))
    setup = Setup([_decay_entry(name, mode, momentum, where=where, typed=typed)])
    assert not custom.problems(setup)
    decayed = 0
    for _ in range(6):
        event = custom.generate_custom(setup, 3.8, rng)
        assert event.custom and event.tracks[0].name == name and event.tracks[0].parent < 0
        if event.tracks[0].end != "decay":
            continue  # it left or showered first (possible for "lifetime")
        decayed += 1
        assert sorted(t.name for t in _children(event, 0)) == sorted(mode.split())
        for i, t in enumerate(event.tracks):
            kids = _children(event, i)
            if not kids:
                continue
            assert t.end == "decay"
            m = float(mass(t.p4)) if P[t.name].width > 0.0 else P[t.name].mass
            parent = four_momentum(m, t.p4_end[1:])
            total = sum(k.p4 for k in kids)
            np.testing.assert_allclose(total, parent, rtol=0.0, atol=1e-8 * parent[0])
            for k in kids:
                np.testing.assert_allclose(k.points[0], t.points[-1])
                if P[k.name].width == 0.0:  # on its mass shell
                    assert float(mass(k.p4)) == pytest.approx(
                        P[k.name].mass, abs=2e-6 * k.p4[0] + 1e-9
                    )
        if name == "H" and mode == "Z Z":
            # Each Z goes on to a pair of electrons or muons.
            for i, t in enumerate(event.tracks):
                if t.name == "Z":
                    pair = sorted(k.name for k in _children(event, i))
                    assert pair in (["e+", "e-"], ["mu+", "mu-"])
    assert decayed >= 4


def test_custom_random_mode_follows_the_branching_fractions():
    # "Random": K0_S -> pi+ pi- 69.20%, pi0 pi0 30.69% (PDG), the rest 0.11%.
    # The Z lists only its three charged-lepton modes (3.3632, 3.3662 and
    # 3.3696%), so a random Z decay is one of those, a third each (0.3330,
    # 0.3333, 0.3337). 20000 draws: binomial error 0.0033.
    rng = np.random.default_rng(62)
    entry = _decay_entry("K0S", "", 5.0)
    drawn = [custom._choose("K0S", entry, rng) for _ in range(20000)]
    assert np.mean([d == ("pi+", "pi-") for d in drawn]) == pytest.approx(0.6920, abs=0.012)
    assert np.mean([d == ("pi0", "pi0") for d in drawn]) == pytest.approx(0.3069, abs=0.012)
    entry = _decay_entry("Z", "", 0.0)
    drawn = [custom._choose("Z", entry, rng) for _ in range(20000)]
    for pair in (("mu+", "mu-"), ("e+", "e-"), ("tau+", "tau-")):
        assert np.mean([d == pair for d in drawn]) == pytest.approx(1.0 / 3.0, abs=0.012)
    assert custom.listed_share("Z") == pytest.approx(0.10099, abs=1e-5)
    # A chosen mode is used every time.
    entry = _decay_entry("K0S", "pi0 pi0", 5.0)
    assert {custom._choose("K0S", entry, rng) for _ in range(50)} == {("pi0", "pi0")}
    # And in whole events the daughters are those drawn.
    setup = Setup([_decay_entry("K0S", "", 2.0, where="start")])
    seen = set()
    for _ in range(40):
        event = custom.generate_custom(setup, 3.8, rng)
        seen.add(tuple(sorted(t.name for t in _children(event, 0))))
    assert seen <= {("pi+", "pi-"), ("pi0", "pi0"), ("e-", "nu", "pi+")} and len(seen) >= 2


@pytest.mark.parametrize(
    "name,mode,momentum,count,reference,tolerance",
    [
        # Known masses (PDG). Tolerance: several times width / sqrt(count),
        # the width being that of the measured peak (K0_S 2 MeV, J/psi 18 MeV,
        # Z 2.1 GeV from its natural width, Higgs 0.6 GeV).
        ("K0S", "pi+ pi-", 5.0, 150, 0.497611, 0.001),
        ("J/psi", "mu+ mu-", 10.0, 400, 3.0969, 0.006),
        ("Z", "mu+ mu-", 0.0, 400, 91.1876, 0.45),
        ("H", "Z Z", 0.0, 40, 125.25, 0.6),
    ],
)
def test_custom_parent_mass_is_reconstructed(name, mode, momentum, count, reference, tolerance):
    rng = np.random.default_rng(63)
    setup = Setup([_decay_entry(name, mode, momentum)])
    hist = custom.histogram_for(setup)
    # The histogram is centred on the parent: its line is the PDG mass.
    assert hist.peaks == ((P[name].mass, name),) and hist.low < reference < hist.high
    assert P[name].mass == pytest.approx(reference, rel=1e-6)
    masses = quick_masses("Custom", count, 3.8, rng, custom=setup)
    assert len(masses) > 0.6 * count
    histogram = MassHistogram(hist)
    histogram.add(masses)
    inside, median, half_width = histogram.summary()
    assert inside > 0.9 * len(masses)  # the range holds the peak
    assert median == pytest.approx(reference, abs=tolerance)
    assert half_width < 0.05 * reference  # within the resolution (and the Z's width)
    # The events shown give the same masses as the quick ones.
    shown = []
    for _ in range(12):
        event = custom.generate_custom(setup, 3.8, rng)
        assert event.entered == [m for _, m in event.masses]
        shown += event.entered
    assert np.median(shown) == pytest.approx(reference, abs=0.04 * reference)


def test_custom_kinked_kaon_keeps_the_d0_peak():
    # D0 -> K- pi+ with a D0 of 3.5 GeV (kaons of 2.0 GeV on average). A K-
    # that decays in flight inside the tracker is measured up to its kink
    # (custom.measured_to_kink), so the event still gives the D0 mass,
    # labelled by the chain. Reference: D0 mass 1864.84 +- 0.05 MeV (PDG
    # particle listing of the D0, "Mass m = 1864.84 +- 0.05 MeV", as for
    # HEAVY below). Resolution sigma: the
    # spread (1.4826 x median absolute deviation) of the events whose K-
    # did not decay in the tracker, about 9.5 MeV. Measured with 3000
    # events: 74% of the kinked events lie within 3 sigma of the PDG mass
    # (1.6% before the kink was measured, when the K-'s muon stood in for
    # it); the rest have a kink too close to the start (fewer than 3
    # layers) or a K- decay without a neutrino whose pi0 is not resolved.
    rng = np.random.default_rng(7)
    setup = Setup([_decay_entry("D0", "K- pi+", 3.5, where="start")])
    reference = 1.86484
    kinked, other, labels = [], [], []
    for _ in range(600):
        event = custom.generate_custom(setup, 3.8, rng, quick=True)
        kaon = next(t for t in event.tracks if t.name == "K-")
        end = kaon.points[-1]
        inside = (
            kaon.end == "decay"
            and np.hypot(end[0], end[1]) < custom.TRACKER_RADIUS
            and abs(end[2]) < custom.TRACKER_HALF_LENGTH
        )
        label, value = event.masses[0] if event.masses else (None, np.nan)
        if inside:
            kinked.append(value)
            labels.append(label)
        elif event.masses:
            other.append(value)
    other, kinked = np.array(other), np.array(kinked)
    sigma = 1.4826 * np.median(np.abs(other - np.median(other)))
    assert 0.005 < sigma < 0.02
    assert len(kinked) > 60  # about 19% of the events
    assert np.mean(np.abs(kinked - reference) < 3.0 * sigma) > 0.6
    assert labels.count("K- pi+ from the D0") > 0.6 * len(kinked)


def test_custom_histogram_choices():
    # Only fired particles: no parent, so no histogram and no masses.
    rng = np.random.default_rng(64)
    gun = Setup([Entry("fire", "K0S", value=2.0), Entry("fire", "mu-", value=20.0)])
    assert custom.histogram_for(gun) is None
    event = custom.generate_custom(gun, 3.8, rng)
    assert event.masses == [] and event.entered == []
    assert quick_masses("Custom", 5, 3.8, rng, custom=gun) == []
    # A tau's decays all have a neutrino: the visible mass, from 0 to 1.2
    # masses, and every entry below the tau's mass (1.77686 GeV) within the
    # resolution.
    tau = Setup([_decay_entry("tau-", "", 45.0)])
    hist = custom.histogram_for(tau)
    assert hist.low == 0.0 and hist.high == pytest.approx(1.2 * 1.77686, rel=0.01)
    masses = quick_masses("Custom", 60, 3.8, rng, custom=tau)
    assert masses and max(masses) < 1.77686 * 1.1
    # The histogram belongs to the first parent; other parents' masses are
    # shown but not entered.
    both = Setup([_decay_entry("J/psi", "mu+ mu-", 10.0), _decay_entry("Z", "mu+ mu-", 0.0)])
    assert custom.histogram_for(both).peaks[0][1] == "J/psi"
    event = custom.generate_custom(both, 3.8, rng)
    assert len(event.masses) == 2 and len(event.entered) == 1
    assert event.entered[0] == pytest.approx(3.0969, abs=0.2)


def test_custom_decay_lengths():
    # Mean decay length (p / m) c tau. A pi0 (m = 0.1349768 GeV, c tau =
    # 25.3 nm) fired at 100 GeV: 18.74 micrometres. A Lambda (1.115683 GeV,
    # 7.89 cm) as a parent at 2 GeV: 14.14 cm, far inside the tracker (1.2 m),
    # so nearly every one decays before it could shower. Statistical error
    # of the mean: 1 / sqrt(N) = 2.2% (2000 pi0) and 1.6% (4000 Lambda).
    rng = np.random.default_rng(65)
    # (They start 5 cm from the calorimeter, so that following the photons is short.)
    gun = Setup([Entry("fire", "pi0", value=100.0, count=50, start=(1.15, 0.0, 0.0))])
    kin = custom.kinematics(gun.entries[0], 3.8)
    assert kin.decay_length == pytest.approx(18.744e-6, rel=1e-3)
    assert kin.beta * kin.gamma == pytest.approx(100.0 / 0.1349768)
    flights = []
    for _ in range(40):
        event = custom.generate_custom(gun, 3.8, rng)
        primaries = [t for t in event.tracks if t.parent < 0]
        assert len(primaries) == 50 and all(t.end == "decay" for t in primaries)
        flights += [np.linalg.norm(t.points[-1] - t.points[0]) for t in primaries]
    assert np.mean(flights) == pytest.approx(18.744e-6, rel=0.07)
    entry = _decay_entry("Lambda", "p pi-", 2.0)
    assert custom.kinematics(entry, 3.8).decay_length == pytest.approx(0.14144, rel=1e-3)
    flights = []
    for _ in range(4000):
        event = Event()
        index = custom._place_parent(event, entry, np.array([2.0, 0.0, 0.0]), 3.8, rng)
        if event.tracks[index].end == "decay":
            flights.append(np.linalg.norm(event.tracks[index].points[-1]))
    assert len(flights) > 3990
    assert np.mean(flights) == pytest.approx(0.14144, rel=0.05)
    # Forced: decays where it is put. At rest: too.
    for entry in (
        _decay_entry("K+", "mu+ nu", 3.0, where="start"),
        _decay_entry("K+", "mu+ nu", 0.0),
    ):
        event = custom.generate_custom(Setup([entry]), 3.8, rng)
        assert event.tracks[0].end == "decay"
        np.testing.assert_allclose(event.tracks[0].points[-1], 0.0)


def test_b_plus_event_decay_length():
    # Mean decay length (p / m) c tau for a B+ of 20 GeV. Reference (PDG 2024,
    # scratchpad/pdg/b.txt line 81): m = 5279.41 MeV, c tau = 491.1 um, so
    # (20 / 5.27941) x 491.1e-6 = 1.8604e-3 m. Statistical error of the mean of
    # 4000 exponentials: 1.6%; 5% allowed.
    expected = 20.0 / 5.27941 * 491.1e-6
    assert expected == pytest.approx(1.8604e-3, rel=1e-4)
    rng = np.random.default_rng(171)
    entry = _decay_entry("B+", "D0bar pi+", 20.0)
    assert custom.kinematics(entry, 3.8).decay_length == pytest.approx(expected, rel=1e-3)
    flights = []
    for _ in range(4000):
        event = Event()
        index = custom._place_parent(event, entry, np.array([20.0, 0.0, 0.0]), 3.8, rng)
        assert event.tracks[index].end == "decay"
        flights.append(np.linalg.norm(event.tracks[index].points[-1] - event.tracks[index].points[0]))
    assert np.mean(flights) == pytest.approx(expected, rel=0.05)


def test_b_plus_mass_from_jpsi_k_products():
    # B+ -> J/psi K+, J/psi -> mu+ mu-, in events as the Custom window makes
    # them (PDG 2024, scratchpad/pdg/b.txt line 81: m(B+) = 5279.41 MeV). From
    # the true momenta the mass is exact; from the tracker-measured momenta
    # (curvature smeared, particle_events.measured) it peaks at that mass:
    # the median must lie within 3 sigma (the measured width, robust: 1.4826
    # MAD) of 5.27941 GeV.
    rng = np.random.default_rng(172)
    entry = _decay_entry("B+", "J/psi K+", 20.0, where="start")
    true_m, meas_m = [], []
    while len(meas_m) < 400:
        event = custom.generate_custom(Setup([entry]), 3.8, rng, quick=True)
        k = next(i for i, t in enumerate(event.tracks) if t.name == "J/psi")
        kids = _children(event, k)
        kaon = next(t for t in _children(event, event.tracks[k].parent) if t.name == "K+")
        products = kids + [kaon]
        if sorted(t.name for t in kids) != ["mu+", "mu-"]:
            continue
        true_m.append(float(mass(sum(t.p4 for t in products))))
        ps = [_events.measured(t, 3.8, rng) for t in products]
        if any(p is None for p in ps):
            continue
        p4 = sum(four_momentum(P[t.name].mass, p) for t, p in zip(products, ps))
        meas_m.append(float(mass(p4)))
    np.testing.assert_allclose(true_m, 5.27941, rtol=0.0, atol=1e-6)
    meas = np.array(meas_m)
    sigma = 1.4826 * np.median(np.abs(meas - np.median(meas)))
    assert 0.0 < sigma < 1.0
    assert abs(np.median(meas) - 5.27941) < 3.0 * sigma


def test_custom_kinematics_and_bending_radius():
    # R = p_T / (0.299792458 q B): 1 GeV in 1 T: 3.3356 m. A muon of 10 GeV at
    # 30 degrees to the beam has p_T = 5 GeV: 4.3890 m in 3.8 T.
    entry = Entry("fire", "mu-", value=10.0, theta=30.0, phi=40.0)
    kin = custom.kinematics(entry, 3.8)
    assert kin.pt == pytest.approx(5.0)
    assert kin.radius == pytest.approx(5.0 / (0.299792458 * 3.8)) == pytest.approx(4.3890, abs=1e-3)
    assert custom.kinematics(Entry("fire", "mu-", value=1.0), 1.0).radius == pytest.approx(
        3.3356, abs=1e-3
    )
    # E^2 = p^2 + m^2, beta = p / E, gamma = E / m (m_mu = 0.1056584 GeV).
    assert kin.energy == pytest.approx(np.hypot(10.0, 0.1056584))
    assert kin.beta == pytest.approx(10.0 / kin.energy) and kin.gamma == pytest.approx(
        kin.energy / 0.1056584
    )
    np.testing.assert_allclose(
        kin.direction,
        [0.5 * np.cos(np.radians(40.0)), 0.5 * np.sin(np.radians(40.0)), np.cos(np.radians(30.0))],
        atol=1e-12,
    )
    # eta = -ln tan(theta / 2): theta = 30 degrees is eta = 1.3170; and back.
    assert kin.eta == pytest.approx(1.3170, abs=1e-4)
    by_eta = custom.kinematics(Entry("fire", "mu-", value=10.0, angles="eta", eta=1.3170), 3.8)
    assert by_eta.theta == pytest.approx(30.0, abs=0.01)
    # The same momentum given as total or kinetic energy.
    for quantity, value in (("energy", kin.energy), ("kinetic", kin.kinetic)):
        again = custom.kinematics(
            Entry("fire", "mu-", quantity=quantity, value=value, theta=30.0), 3.8
        )
        assert again.momentum == pytest.approx(10.0)
    # Neutral, along the beam, or at rest: no radius.
    assert custom.kinematics(Entry("fire", "gamma", value=10.0), 3.8).radius is None
    assert custom.kinematics(Entry("fire", "mu-", value=10.0, theta=0.0), 3.8).radius is None
    assert custom.kinematics(Entry("fire", "mu-", value=0.0), 3.8).radius is None
    # The path that is drawn has that radius: a 2 GeV muon across a 3.8 T
    # field, R = 1.7556 m, from three points of its first 60 cm (the circle
    # through them; scattering and energy loss in the tracker change it by
    # well under the 4% allowed).
    rng = np.random.default_rng(66)
    radii = []
    for _ in range(20):
        event = custom.generate_custom(Setup([Entry("fire", "mu+", value=2.0)]), 3.8, rng)
        a, b, c = (event.tracks[0].points[k, :2] for k in (0, 10, 20))
        ab, bc, ca = (np.linalg.norm(u - v) for u, v in ((a, b), (b, c), (c, a)))
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        radii.append(ab * bc * ca / (4.0 * area))
    assert np.mean(radii) == pytest.approx(2.0 / (0.299792458 * 3.8), rel=0.04)
    assert 2.0 / (0.299792458 * 3.8) == pytest.approx(1.7556, abs=1e-4)


def test_gun_spread():
    # Momentum: Gaussian of the given relative width. Direction: two
    # independent Gaussian angles of the given width, so the angle to the
    # axis has the mean square 2 sigma^2 (for small angles).
    rng = np.random.default_rng(67)
    entry = Entry("fire", "mu-", value=20.0, spread_p=10.0, spread_angle=2.0)
    kin = custom.kinematics(entry, 3.8)
    drawn = np.array([custom._draw(entry, kin, rng) for _ in range(20000)])
    sizes = np.linalg.norm(drawn, axis=1)
    assert sizes.mean() == pytest.approx(20.0, rel=0.005) and sizes.std() == pytest.approx(
        2.0, rel=0.03
    )
    angles = np.arccos(np.clip(drawn @ np.array(kin.direction) / sizes, -1.0, 1.0))
    assert np.sqrt(np.mean(angles**2)) == pytest.approx(np.radians(2.0) * np.sqrt(2.0), rel=0.03)
    # No spread: exactly the set momentum, every time.
    entry = Entry("fire", "mu-", value=20.0, theta=60.0)
    kin = custom.kinematics(entry, 3.8)
    np.testing.assert_allclose(custom._draw(entry, kin, rng), 20.0 * np.array(kin.direction))


REFUSED = [
    # (entry, words of the message)
    (Entry("fire", "Z", value=10.0), "decays at once"),
    (Entry("fire", "gamma", value=0.0), "no mass"),
    (Entry("fire", "mu-", value=1.0e6), "at most 7000"),
    (Entry("fire", "mu-", value=-1.0), "negative"),
    (Entry("fire", "mu-", value=float("nan")), "not a number"),
    (Entry("fire", "mu-", value=float("inf")), "not a number"),
    (Entry("fire", "mu-", quantity="energy", value=0.05), "below the rest energy"),
    (Entry("fire", "quark", value=1.0), "not in the particle table"),
    (Entry("fire", "mu-", theta=200.0), "polar angle"),
    (Entry("fire", "mu-", angles="eta", eta=50.0), "pseudorapidity"),
    (Entry("fire", "mu-", start=(9.0, 0.0, 0.0)), "outside the detector"),
    (Entry("fire", "mu-", start=(0.0, float("nan"), 0.0)), "three numbers"),
    (Entry("fire", "mu-", count=0), "between 1 and"),
    (Entry("fire", "mu-", count=10**6), "between 1 and"),
    (Entry("fire", "mu-", spread_p=500.0), "momentum spread"),
    (Entry("fire", "mu-", spread_angle=-1.0), "direction spread"),
    (Entry("explode", "mu-"), "unknown action"),
    # Stable particles (here) cannot be parents.
    (Entry("decay", "p", value=5.0), "stable here"),
    (Entry("decay", "e-", value=5.0), "stable here"),
    (Entry("decay", "gamma", value=5.0), "stable here"),
    # K0_S (0.4976 GeV) -> four pions (0.549 GeV): energy is missing.
    (Entry("decay", "K0S", daughters="pi+ pi- pi0 pi0", typed=True), "kinematically forbidden"),
    (Entry("decay", "K0S", daughters="pi+ pi0", typed=True), "charge is not conserved"),
    (Entry("decay", "K0S", daughters="pi+ kaon", typed=True), "not in the particle table"),
    (Entry("decay", "K0S", daughters="pi0", typed=True), "at least two daughters"),
    (Entry("decay", "K0S", daughters="", typed=True), "at least two daughters"),
    (Entry("decay", "Z", daughters=" ".join(["gamma"] * 7), typed=True), "at most 6"),
    (Entry("decay", "H", daughters="Z gamma", typed=True), "not supported"),
    (Entry("decay", "K0S", where="later"), "unknown place"),
]


@pytest.mark.parametrize("entry,words", REFUSED, ids=[w for _, w in REFUSED])
def test_custom_setups_that_are_refused(entry, words):
    errors, _ = custom.check(entry)
    assert any(words in text for text in errors), errors
    setup = Setup([entry])
    assert any(words in text for text in custom.problems(setup))
    rng = np.random.default_rng(68)
    with pytest.raises(ValueError, match=words):
        custom.generate_custom(setup, 3.8, rng)
    before = custom.current()
    with pytest.raises(ValueError, match=words):
        custom.set_current(setup)
    assert custom.current() is before  # nothing changed


def test_custom_setup_limits_and_files():
    assert "empty" in custom.problems(Setup([]))[0]
    crowd = Setup([Entry("fire", "pi+", count=100) for _ in range(4)])
    assert any("At most 300 particles" in text for text in custom.problems(crowd))
    many = Setup([Entry("fire", "pi+") for _ in range(custom.MAX_ENTRIES + 1)])
    assert any("At most 20 entries" in text for text in custom.problems(many))
    # Every preset is valid, and survives being written as JSON and read back.
    import json

    for name in custom.PRESETS:
        setup = custom.preset(name)
        assert custom.problems(setup) == []
        assert custom.from_dict(json.loads(json.dumps(custom.to_dict(setup)))) == setup
    full = Setup(
        [
            Entry("decay", "K0S", "kinetic", 3.5, "eta", 12.0, -1.5, 77.0, (0.1, -0.2, 0.3), 7,
                  5.0, 2.5, "pi0 pi0", False, "start")
        ]
    )  # fmt: skip
    assert custom.from_dict(json.loads(json.dumps(custom.to_dict(full)))) == full
    # Malformed files are refused with ValueError, whatever is wrong.
    good = custom.to_dict(custom.preset("K0S at 5 GeV"))

    def changed(**values):
        return {"format": 1, "entries": [{**good["entries"][0], **values}]}

    bad = [
        None, [], "text", {}, {"entries": "none"}, {"entries": [1]}, {"entries": [{}]},
        {"format": 99, "entries": good["entries"]}, {"format": "1", "entries": good["entries"]},
        {"entries": good["entries"] * 21},
        changed(name="quark"), changed(name=5), changed(value="fast"), changed(value=float("nan")),
        changed(value=True), changed(count=2.5), changed(count=True), changed(count=5000),
        changed(start=[0.0, 0.0]), changed(start="origin"), changed(start=[0.0, None, 0.0]),
        changed(quantity="speed"), changed(angles="degrees"), changed(action="explode"),
        changed(daughters=["pi+", "pi-"]), changed(daughters="pi+ pi+"), changed(typed="yes"),
        changed(where=3), changed(theta=500.0), changed(eta=float("inf")),
    ]  # fmt: skip
    for data in bad:
        with pytest.raises(ValueError):
            custom.from_dict(data)
    # Missing optional values take their defaults.
    assert custom.from_dict({"entries": [{"name": "mu-"}]}) == Setup([Entry()])


def test_large_custom_setups_stay_bounded():
    # The largest setup the checks allow, of the most expensive kinds (TeV
    # electrons and photons, which radiate and convert; curling pions; tau
    # chains): an event stops taking on particles after TIME_BUDGET (1.5 s)
    # and says so. The bound asserted leaves room for a slow machine.
    rng = np.random.default_rng(69)
    setup = Setup(
        [
            Entry("fire", "e-", value=7000.0, count=100, spread_angle=30.0),
            Entry("fire", "pi+", value=0.3, count=100, spread_angle=30.0),
            Entry("decay", "tau-", value=45.0, count=100, spread_angle=30.0),
        ]
    )
    assert custom.problems(setup) == [] and custom.placed(setup) == custom.MAX_PLACED
    started = time.perf_counter()
    event = custom.generate_custom(setup, 3.8, rng)
    elapsed = time.perf_counter() - started
    assert elapsed < custom.TIME_BUDGET + 4.0
    assert event.tracks and len(event.tracks) < custom.MAX_TRACKS + 2000
    primaries = sum(1 for t in event.tracks if t.parent < 0)
    if primaries < 300:
        assert f"Only {primaries} of the 300" in event.note
    for t in event.tracks:
        assert np.all(np.isfinite(t.points)) and np.all(np.isfinite(t.times))
    # Extreme but allowed single particles all end, quickly.
    for entry in (
        Entry("fire", "mu-", value=7000.0, theta=0.0),  # along the beam
        Entry("fire", "mu-", value=7000.0, theta=180.0),
        Entry("fire", "e-", value=1.0e-9),  # taken as at rest
        Entry("fire", "pi+", value=0.0),
        Entry("fire", "p", value=0.0),
        Entry("fire", "gamma", value=1.0e-9),
        Entry("fire", "nu", value=7000.0),
        Entry("fire", "pi+", value=0.02),  # curls with a radius of 2 cm
        Entry("decay", "Z", value=7000.0, daughters="tau+ tau-"),
        Entry("decay", "H", value=7000.0, daughters="Z Z", angles="eta", eta=10.0),
        Entry("decay", "pi0", value=7000.0, where="start"),
        Entry("decay", "mu-", value=0.0),
        Entry("fire", "K0S", value=5.0, start=(0.0, 0.0, 6.4)),  # at the detector's edge
        Entry("fire", "mu-", value=5.0, start=(7.2, 0.0, 0.0)),
    ):
        setup = Setup([entry])
        assert custom.problems(setup) == [], entry
        started = time.perf_counter()
        event = custom.generate_custom(setup, 3.8, rng)
        assert time.perf_counter() - started < 3.0, entry
        assert event.tracks and all(t.end for t in event.tracks)


def test_custom_events_are_not_called_collisions():
    # Accuracy scope: a custom event is a set of particles the user placed.
    # Its texts say so, and say nothing of protons or of a production
    # mechanism; no protons are drawn for it.
    pytest.importorskip("imgui_bundle")
    pytest.importorskip("moderngl")
    from physengine.modules import particlemodule

    rng = np.random.default_rng(70)
    for name in custom.PRESETS:
        event = generate("Custom", 3.8, rng, custom=custom.preset(name))
        assert event.custom and not event.real and event.mechanism == "" and not event.bosons
        assert "not a collision" in event.headline
        assert all(t.origin == "placed" for t in event.tracks if t.parent < 0)
        log = particlemodule.narrate(event)
        assert log and "not a collision" in log[0][1] and log[0][0] == 0.0
        text = " ".join(line for _, line in log)
        assert "proton" not in text.lower() and "collide" not in text
        assert "from the collision" not in text
    # generate() without a setup uses the current one (the default preset).
    event = generate("Custom", 3.8, rng)
    assert [t.name for t in event.tracks if t.parent < 0] == ["mu-"]
    assert "Custom" in SCENES


def test_module_runs_custom_setups():
    # The module with the Custom event type, without a window: applying a
    # setup shows an event of it and restarts the histogram; events added in
    # bulk are of the setup that is running, also when it changes meanwhile;
    # a saved scenario comes back; a bad one changes nothing.
    pytest.importorskip("imgui_bundle")
    pytest.importorskip("moderngl")
    from physengine.modules import particle_editor, particlemodule

    module = particlemodule.ParticleModule()
    module.scene_name = "Custom"
    module.playback = "One at a time"

    def frames(until, seconds=60.0):
        deadline = time.perf_counter() + seconds
        while not until() and time.perf_counter() < deadline:
            module._follow_custom()  # what draw_ui does each frame
            module._collect_fill()  # what step does
            time.sleep(0.005)
        assert until()

    def settled():
        return not (module.custom_dirty or module.custom_waiting)

    try:
        module.new_event()
        assert module.event.custom and module.event.tracks[0].name == "mu-"
        assert module._histogram() is None
        module.request_fill(100)  # no histogram: nothing to add
        assert module.to_fill == 0
        # No protons among the segments (they are the ones with a negative width).
        assert np.all(module._segments()[4][:, 3] >= 0.0)
        assert module.lead_in < 1.0 and module.now == -module.lead_in

        module.apply_custom(custom.preset("J/psi at 10 GeV to mu+ mu-"))
        frames(settled)
        assert (
            module.event.tracks[0].name == "J/psi" and module.histogram.hist.peaks[0][1] == "J/psi"
        )
        assert module.histogram.entries == len(module.event.entered) <= 1
        module.request_fill(300)
        frames(lambda: module.to_fill == 0)
        assert module.histogram.entries > 250
        assert module.histogram.summary()[1] == pytest.approx(3.0969, abs=0.01)

        # Changing the setup while events are being added: the histogram
        # starts again and takes nothing made for the old setup.
        module.request_fill(4000)
        frames(lambda: module.histogram.entries > 350)
        module.apply_custom(custom.preset("Z at rest to e+ e-"))
        assert module.to_fill == 0 and module.histogram.entries == 0
        frames(settled)
        if module.filler is not None:
            module.filler.join()
        module._collect_fill()
        assert module.event.tracks[0].name == "Z"
        assert module.histogram.hist.peaks[0][1] == "Z"
        assert module.histogram.entries == len(module.event.entered) <= 1

        # An invalid setup is refused and nothing changes.
        version, running = module.version, module.custom
        with pytest.raises(ValueError, match="stable here"):
            module.apply_custom(Setup([Entry("decay", "p")]))
        assert module.version == version and module.custom is running

        # Save and load (the dictionary scenario_files writes).
        module.field = 2.0
        data = module.scenario()
        module.apply_custom(custom.preset("tau at 45 GeV"))
        module.field = 3.8
        module.load_scenario(data)
        assert module.field == 2.0 and module.custom == custom.preset("Z at rest to e+ e-")
        assert particle_editor.draft() == module.custom
        frames(settled)
        assert module.event.tracks[0].name == "Z"
        for bad in (
            None,
            {"scene_name": "Z -> mu+ mu-"},
            {"setup": {"entries": [{"name": "quark"}]}},
            {"field": 99.0, "setup": data["setup"]},
            {"field": "strong", "setup": data["setup"]},
        ):
            with pytest.raises(ValueError):
                module.load_scenario(bad)
            assert module.field == 2.0 and module.custom == custom.preset("Z at rest to e+ e-")

        # Full reset: back to the default setup.
        module.discard_edits()
        assert custom.current() == custom.preset(custom.DEFAULT_PRESET)
        assert particle_editor.draft() == custom.current()
        # The window's readout: a 50 GeV muon in 3.8 T has R = 50 / (0.2998 * 3.8) = 43.9 m.
        lines = particle_editor.readout(Entry("fire", "mu-", value=50.0), 3.8)
        assert any("43.9 m" in line for line in lines)
    finally:
        module.fill_cancel.set()
        module.cancel_fill()
        for thread in (module.worker, module.filler):
            if thread is not None:
                thread.join()
        particle_editor.forget_draft()


# ------------------------------------------------- charm and bottom mesons
#
# Reference values: PDG 2024 (S. Navas et al., Phys. Rev. D 110, 030001),
# particle listings of the D+-, D0, Ds+-, B+-, B0 and Bs. Masses [MeV] and
# c tau [micrometres]: D+ 1869.66, 309.8; D0 1864.84, 123.01; Ds+ 1968.35,
# 150.3; B+ 5279.41, 491.1; B0 5279.72, 454.8; Bs 5366.93, 455.7.
# Branching fractions quoted in the tests below; a K0 or K0bar mode is
# entered half as K0_S, half as K0_L.

HEAVY = {
    "D+": (1.86966, 309.8e-6, +1),
    "D0": (1.86484, 123.01e-6, 0),
    "Ds+": (1.96835, 150.3e-6, +1),
    "B+": (5.27941, 491.1e-6, +1),
    "B0": (5.27972, 454.8e-6, 0),
    "Bs": (5.36693, 455.7e-6, 0),
}
ANTI_HEAVY = {"D+": "D-", "D0": "D0bar", "Ds+": "Ds-", "B+": "B-", "B0": "B0bar", "Bs": "Bsbar"}
# Share of all decays the listed modes make up: the sum of the PDG fractions
# listed (D+ 38 PDG modes, D0 37, Ds+ 22, B+ 35, B0 27, Bs 7; the B's l nu
# modes counted once for e and once for mu, as the PDG's [a] footnote says).
# B+ and B0 with the D* modes (added 2026-11-23, PDG 2024, scratchpad
# pdg/b.txt lines in particle_data): B+ 0.089259 + 0.248640, B0 0.081751 +
# 0.218920.
# J/psi: mu mu 5.961% + e e 5.971%. Z: 3.3662 + 3.3632 + 3.3696%.
COVERED = {
    "D+": 0.677671,
    "D0": 0.719814,
    "Ds+": 0.352840,
    "B+": 0.337899,
    "B0": 0.300671,
    "Bs": 0.038612,
    "J/psi": 0.11932,
    "Z": 0.10099,
}
# Some of the listed fractions, as quoted by the PDG.
HEAVY_BRANCHING = {
    "D+": {("K-", "pi+", "pi+"): 0.0938, ("K0S", "e+", "nu"): 0.0872 / 2, ("mu+", "nu"): 3.74e-4},
    "D0": {("K-", "pi+"): 0.03947, ("K-", "pi+", "pi0"): 0.144, ("K-", "e+", "nu"): 0.03549},
    "Ds+": {("K+", "K-", "pi+"): 0.0537, ("tau+", "nu"): 0.0536, ("p", "nbar"): 1.22e-3},
    "B+": {
        ("J/psi", "K+"): 1.020e-3,
        ("D0bar", "e+", "nu"): 0.0221,
        ("D0bar", "mu+", "nu"): 0.0221,
        ("D*0bar", "e+", "nu"): 0.0553,  # b.txt:347
        ("D*0bar", "Ds*+"): 0.0171,  # b.txt:693-694
    },
    "B0": {
        ("J/psi", "K0S"): 8.91e-4 / 2,
        ("D-", "pi+"): 2.51e-3,
        ("D-", "tau+", "nu"): 9.9e-3,
        ("D*-", "mu+", "nu"): 0.049,  # b.txt:1804
        ("D*-", "pi+"): 2.66e-3,  # b.txt:1877
    },
    "Bs": {("Ds-", "pi+"): 2.98e-3, ("Ds-", "mu+", "nu"): 0.0231, ("J/psi", "K+", "K-"): 7.9e-4},
}


def test_heavy_mesons_have_pdg_masses_and_lifetimes_and_no_width():
    for name, (m, ctau, charge) in HEAVY.items():
        for n in (name, ANTI_HEAVY[name]):
            particle = P[n]
            assert particle.mass == pytest.approx(m, abs=1e-9)
            assert particle.ctau == pytest.approx(ctau, rel=1e-9)
            # Not resonances: they fly (c tau) and decay with their own mass.
            assert particle.width == 0.0
            assert particle.partial and not particle.rest
        anti = P[ANTI_HEAVY[name]]
        assert anti.charge == -charge == -P[name].charge
        assert len(anti.decays) == len(P[name].decays)


def test_heavy_meson_decays_conserve_four_momentum():
    # Every listed mode of every D and B meson (and antiparticle), through the
    # function events use, from a moving parent: energy and momentum conserved,
    # daughters on their mass shells, and the parent's mass reconstructed from
    # the daughters (m^2 = (sum E)^2 - |sum p|^2) equal to its PDG mass.
    rng = np.random.default_rng(91)
    for name in [*HEAVY, *ANTI_HEAVY.values()]:
        particle = P[name]
        parent = four_momentum(particle.mass, [0.4, 2.0, -1.0])
        for _, daughters in particle.decays:
            assert sum(P[d].charge for d in daughters) == particle.charge
            assert sum(P[d].mass for d in daughters) < particle.mass
            for _ in range(3):
                out = decay_momenta(name, parent, daughters, rng)
                # A daughter with a width (D*+, J/psi) has a Breit-Wigner mass,
                # within 10 widths of its PDG mass (particle_events.daughter_mass).
                masses = []
                for d, q in zip(daughters, out, strict=True):
                    m_d = float(mass(q)) if P[d].width > 0.0 else P[d].mass
                    assert abs(m_d - P[d].mass) <= 10.0 * P[d].width + 1e-12
                    masses.append(m_d)
                _check_decay(out, parent, masses)
                assert float(mass(sum(out))) == pytest.approx(particle.mass, rel=1e-9)


def test_partial_tables_draw_renormalised_pdg_fractions():
    rng = np.random.default_rng(92)
    for name, share in COVERED.items():
        names = [name, ANTI_HEAVY[name]] if name in ANTI_HEAVY else [name]
        for n in names:
            particle = P[n]
            # Share covered = sum of the listed fractions (to the 1e-6 of the sums above).
            assert particle.covered() == pytest.approx(share, abs=1e-6)
            assert custom.listed_share(n) == pytest.approx(share, abs=1e-6)
            assert n in custom.INCOMPLETE
            probabilities, modes = particle.modes()  # no ValueError for a partial table
            assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)
            for (b, _), p in zip(particle.decays, probabilities, strict=True):
                assert p == pytest.approx(b / share, rel=1e-5)
    for name, table in HEAVY_BRANCHING.items():
        listed = dict((d, b) for b, d in P[name].decays)
        for daughters, fraction in table.items():
            assert listed[daughters] == pytest.approx(fraction, rel=1e-9)
    # The draw: 100000 D0 decays; K- pi+ should come out at 3.947% / 71.9814%
    # = 5.483%, binomial error sqrt(f (1 - f) / n) = 0.07%.
    probabilities, modes = P["D0"].modes()
    n = 100000
    drawn = rng.choice(len(modes), p=probabilities, size=n)
    f = 0.03947 / 0.719814
    error = np.sqrt(f * (1 - f) / n)
    assert np.mean(drawn == modes.index(("K-", "pi+"))) == pytest.approx(f, abs=4.0 * error)
    # The Custom window says which share is covered (71.98% to 3 digits: 72%).
    warnings = custom.check(Entry("decay", "D0", value=10.0))[1]
    assert any("which are 72% of all" in w for w in warnings)
    assert any("divided by the listed share so that the fractions sum to one" in w for w in warnings)


def test_d0_k_minus_pi0_e_nu_is_the_isospin_estimate():
    # Isospin limit, K pi in I = 1/2 (K*(892)- and S-wave): Clebsch-Gordan
    # |<K- pi0|1/2,-1/2>|^2 = 1/3, |<Kbar0 pi-|1/2,-1/2>|^2 = 2/3, so
    # B(D0 -> K- pi0 e+ nu) = B(D0 -> Kbar0 pi- e+ nu) / 2. Reference:
    # PDG 2024 (S. Navas et al., PRD 110, 030001), D0 semileptonic modes,
    # Kbar0 pi- e+ nu = (1.44 +- 0.04)%, so 0.72%. The Kbar0 is entered as
    # half K0S, half K0L (0.72% each).
    listed = dict((d, b) for b, d in P["D0"].decays)
    kbar0 = listed[("K0S", "pi-", "e+", "nu")] + listed[("K0L", "pi-", "e+", "nu")]
    assert kbar0 == pytest.approx(0.0144, rel=1e-9)
    assert listed[("K-", "pi0", "e+", "nu")] == pytest.approx(0.0144 / 2, rel=1e-9)
    # And the same for the antiparticle (K+ pi0 e- nu from the D0bar).
    anti = dict((d, b) for b, d in P["D0bar"].decays)
    assert anti[("K+", "pi0", "e-", "nu")] == pytest.approx(0.0072, rel=1e-9)


def test_jpsi_daughter_decays_by_its_own_table():
    # A J/psi made in a decay (B+ -> J/psi K+, PDG 1.020e-3) decays at once by
    # its table: mu+ mu- 5.961%, e+ e- 5.971%, renormalised to 0.4996 and
    # 0.5004. 400 decays: binomial error 2.5%; 4 sigma allowed. No crash.
    rng = np.random.default_rng(93)
    probabilities, modes = P["J/psi"].modes()
    assert probabilities[modes.index(("mu+", "mu-"))] == pytest.approx(0.05961 / 0.11932, rel=1e-9)
    entry = _decay_entry("B+", "J/psi K+", 20.0, where="start")
    counts = {}
    n = 400
    for _ in range(n):
        event = custom.generate_custom(Setup([entry]), 3.8, rng, quick=True)
        assert not event.error
        jpsi = [k for k, t in enumerate(event.tracks) if t.name == "J/psi"]
        assert len(jpsi) == 1 and event.tracks[jpsi[0]].end == "decay"
        children = tuple(sorted(t.name for t in _children(event, jpsi[0])))
        counts[children] = counts.get(children, 0) + 1
        # The J/psi's own four-momentum is shared out exactly.
        total = sum(t.p4 for t in _children(event, jpsi[0]))
        np.testing.assert_allclose(total, event.tracks[jpsi[0]].p4_end, atol=1e-9)
    assert set(counts) <= {("mu+", "mu-"), ("e+", "e-")}
    assert counts[("mu+", "mu-")] / n == pytest.approx(0.4996, abs=0.1)
    # And in flight, as events make it (particle_events._produce): a J/psi
    # that is the daughter of a B decays without an error.
    for _ in range(50):
        event = Event()
        _produce(event, "J/psi", four_momentum(P["J/psi"].mass, [1.0, 3.0, 2.0]),
                 np.zeros(3), 0.0, 3.8, rng, generation=1, origin="decay")  # fmt: skip
        assert event.tracks[0].end == "decay"
        assert tuple(sorted(t.name for t in _children(event, 0))) in {("mu+", "mu-"), ("e+", "e-")}


# --- Semileptonic decays P -> P' l nu (particle_kinematics.semileptonic_decay)
from physengine.modules import particle_events as _events  # noqa: E402
from physengine.modules import particle_kinematics as _kin  # noqa: E402

LEPTONS = ("e+", "e-", "mu+", "mu-", "tau+", "tau-")


def _sl_analytic(model, M, m, ml, q2):
    # dGamma/dq^2 / (G_F^2 |V|^2 / 24 pi^3), the standard result for a 0- -> 0-
    # transition with a massive lepton (e.g. Korner & Schuler 1990, Z. Phys. C
    # 46, 93): (1 - ml^2/q^2)^2 |p'| [(1 + ml^2/2q^2) |p'|^2 f+^2
    #                                  + 3 ml^2/(8 q^2) (M^2 - m^2)^2 / M^2 f0^2].
    fp, f0 = _kin.semileptonic_form_factors(model, M, m, q2)
    p = _kin._momentum(M, np.sqrt(q2), m)
    return (1 - ml**2 / q2) ** 2 * p * (
        (1 + ml**2 / (2 * q2)) * p**2 * fp**2 + 3 * ml**2 / (8 * q2) * (M * M - m * m) ** 2 / M**2 * f0**2
    )


def test_every_heavy_p_to_p_l_nu_mode_has_a_form_factor_model():
    for name in [*HEAVY, *ANTI_HEAVY.values()]:
        for _, d in P[name].decays:
            if len(d) == 3 and d[1] in LEPTONS and d[2] == "nu":
                assert (name, d[0]) in _events.SEMILEPTONIC_MODELS, (name, d)


def test_semileptonic_matrix_element_integrates_to_the_analytic_q2_spectrum():
    # The (q^2, cos theta) density, integrated over cos theta, must equal the
    # analytic dGamma/dq^2 above up to one constant, for e, mu and tau
    # (the tau case tests the scalar form factor f0 of CLN).
    cases = [
        (_events.SEMILEPTONIC_MODELS[("B0", "D-")], P["B0"].mass, P["D-"].mass, P["tau+"].mass),
        (_events.SEMILEPTONIC_MODELS[("B0", "D-")], P["B0"].mass, P["D-"].mass, P["mu+"].mass),
        (_events.SEMILEPTONIC_MODELS[("D0", "K-")], P["D0"].mass, P["K-"].mass, P["mu+"].mass),
        (_events.SEMILEPTONIC_MODELS[("D0", "pi-")], P["D0"].mass, P["pi-"].mass, P["e+"].mass),
    ]
    for model, M, m, ml in cases:
        q2 = np.linspace(ml**2, (M - m) ** 2, 401)[1:-1]
        c = np.linspace(-1.0, 1.0, 801)
        numeric = np.trapezoid(_kin.semileptonic_density(model, M, m, ml, q2[:, None], c[None, :]), c, axis=1)
        ratio = numeric / _sl_analytic(model, M, m, ml, q2)
        np.testing.assert_allclose(ratio / ratio.mean(), 1.0, atol=1e-5)


def test_d_to_k_and_pi_e_nu_rates_match_besiii():
    # Reference: BESIII 2015, PRD 92, 072012 (arXiv:1508.07560). From their
    # two-parameter series fit, f+(0)|Vcs| = 0.7172, r1 = -2.2286 (K) and
    # f+(0)|Vcd| = 0.1435, r1 = -2.0365 (pi), the rate
    #   Gamma = G_F^2 |V f+(0)|^2 / (24 pi^3) int |p'|^3 (f+(q^2)/f+(0))^2 dq^2
    # times tau(D0) = 410.3 fs (PDG 2024) must give their measured
    # B(D0 -> K- e+ nu) = (3.505 +- 0.014 +- 0.033)% and
    # B(D0 -> pi- e+ nu) = (0.295 +- 0.004 +- 0.003)%. G_F = 1.1663788e-5 GeV^-2,
    # hbar = 6.582119569e-25 GeV s. Allowed: the total errors (0.036%, 0.005%).
    G_F, hbar = 1.1663788e-5, 6.582119569e-25
    for meson, fv, measured_bf, error in (("K-", 0.7172, 0.03505, 0.00036), ("pi-", 0.1435, 0.00295, 0.00005)):
        model = _events.SEMILEPTONIC_MODELS[("D0", meson)]
        M, m = P["D0"].mass, P[meson].mass
        q2 = np.linspace(1e-9, (M - m) ** 2, 20001)
        f = _kin.semileptonic_form_factors(model, M, m, q2)[0]
        f0 = _kin.semileptonic_form_factors(model, M, m, np.array(1e-9))[0]
        integral = np.trapezoid(_kin._momentum(M, np.sqrt(q2), m) ** 3 * (f / f0) ** 2, q2)
        bf = G_F**2 * fv**2 / (24 * np.pi**3) * integral * 410.3e-15 / hbar
        assert bf == pytest.approx(measured_bf, abs=error)


def test_b_to_d_tau_over_light_lepton_ratio_from_cln():
    # R(D) = Gamma(B -> D tau nu) / Gamma(B -> D l nu) with the CLN form factors
    # (rho^2 = 1.129, HFLAV spring 2021; S1/V1 from CLN Table A.1). Reference:
    # the CLN-based SM value R(D) = 0.296 +- 0.016 (Fajfer, Kamenik & Nisandzic
    # 2012, PRD 85, 094025). The model gives 0.302.
    model = _events.SEMILEPTONIC_MODELS[("B0", "D-")]
    M, m = P["B0"].mass, P["D-"].mass

    def rate(ml):
        q2 = np.linspace(ml**2, (M - m) ** 2, 20001)[1:-1]
        return np.trapezoid(_sl_analytic(model, M, m, ml, q2), q2)

    assert rate(P["tau+"].mass) / rate(P["e+"].mass) == pytest.approx(0.296, abs=0.016)


def _w_frame_angle(parent, lepton, neutrino):
    # cos of the charged lepton's angle, in the l nu rest frame, to the parent's direction there.
    w = lepton + neutrino
    inverse = np.array([w[0], *(-w[1:])])
    l_w = _kin.boost(lepton, inverse)[1:]
    p_w = _kin.boost(parent, inverse)[1:]
    return l_w @ p_w / np.linalg.norm(l_w) / np.linalg.norm(p_w)


def test_semileptonic_events_follow_q2_spectrum_and_sin2_lepton_angle():
    rng = np.random.default_rng(94)
    # D0 -> K- e+ nu from a moving D0, through decay_momenta (what events use).
    # (1) Four-momentum conserved. (2) The electron's angle in the e nu frame
    # follows sin^2(theta) (W helicity 0, m_e -> 0): <cos^2> = 1/5, not the 1/3
    # of phase space; n = 6000: error of <cos^2> about sqrt(Var)/sqrt(n) with
    # Var(cos^2) = 1/7 - 1/25 for sin^2: 0.0041; 4 sigma allowed.
    # (3) q^2 histogram matches the analytic spectrum (chi^2 per bin < 2).
    parent = four_momentum(P["D0"].mass, [0.5, -1.0, 3.0])
    model = _events.SEMILEPTONIC_MODELS[("D0", "K-")]
    M, m, ml = P["D0"].mass, P["K-"].mass, P["e+"].mass
    n = 6000
    q2, cos2 = np.empty(n), np.empty(n)
    for i in range(n):
        out = decay_momenta("D0", parent, ("K-", "e+", "nu"), rng)
        if i < 50:
            _check_decay(out, parent, [m, ml, 0.0])
        q2[i] = float(mass(out[1] + out[2]) ** 2)
        cos2[i] = _w_frame_angle(parent, out[1], out[2]) ** 2
    assert np.mean(cos2) == pytest.approx(0.2, abs=4 * np.sqrt((1 / 7 - 1 / 25) / n))
    edges = np.linspace(ml**2, (M - m) ** 2, 21)
    counts = np.histogram(q2, edges)[0]
    fine = np.linspace(ml**2, (M - m) ** 2, 20001)[1:-1]
    density = _sl_analytic(model, M, m, ml, fine)
    cumulative = np.concatenate([[0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(fine))])
    expected = np.diff(np.interp(edges, fine, cumulative, left=0.0, right=cumulative[-1]))
    expected *= n / expected.sum()
    keep = expected > 5
    chi2 = np.sum((counts[keep] - expected[keep]) ** 2 / expected[keep])
    assert chi2 / keep.sum() < 2.0
    # B0 -> D- tau+ nu: the tau's mass enters (scalar form factor); same check.
    parent = four_momentum(P["B0"].mass, [0.0, 2.0, 5.0])
    model = _events.SEMILEPTONIC_MODELS[("B0", "D-")]
    M, m, ml = P["B0"].mass, P["D-"].mass, P["tau+"].mass
    q2 = np.array([float(mass(sum(decay_momenta("B0", parent, ("D-", "tau+", "nu"), rng)[1:])) ** 2) for _ in range(n)])
    edges = np.linspace(ml**2, (M - m) ** 2, 21)
    counts = np.histogram(q2, edges)[0]
    fine = np.linspace(ml**2, (M - m) ** 2, 20001)[1:-1]
    density = _sl_analytic(model, M, m, ml, fine)
    cumulative = np.concatenate([[0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(fine))])
    expected = np.diff(np.interp(edges, fine, cumulative, left=0.0, right=cumulative[-1]))
    expected *= n / expected.sum()
    keep = expected > 5
    assert np.sum((counts[keep] - expected[keep]) ** 2 / expected[keep]) / keep.sum() < 2.0


# --- B -> J/psi K: the J/psi is longitudinally polarised
def _helicity_cos(jpsi, lepton, b):
    # cos of the lepton's angle in the J/psi rest frame to the J/psi's flight
    # direction in the B frame (= minus the B's direction in the J/psi frame).
    inverse = np.array([jpsi[0], *(-jpsi[1:])])
    l_rest = _kin.boost(lepton, inverse)[1:]
    axis = -_kin.boost(b, inverse)[1:]
    return l_rest @ axis / np.linalg.norm(l_rest) / np.linalg.norm(axis)


def test_jpsi_from_b_to_jpsi_k_has_sin2_muon_distribution():
    # Spin 0 -> spin 1 + spin 0: the J/psi has helicity 0 (angular momentum
    # along the decay axis), so l+ l- go as sin^2(theta) + (4 m_l^2/M^2) cos^2(theta)
    # in the helicity frame (helicity formalism, Jacob & Wick 1959; e.g. the
    # sin^2 law used in LHCb/CMS B -> J/psi K angular analyses). Reference
    # values: <cos^2 theta> = (4/15 + 2 eps/5) / (4/3 + 2 eps/3) with
    # eps = 4 m_mu^2 / m_J/psi^2 = 0.004656: 0.20093 (isotropic would be 1/3).
    # Var(cos^2) for sin^2 = 1/7 - 1/25; n = 4000: 4 sigma = 0.0115.
    rng = np.random.default_rng(95)
    m_mu, m_psi = P["mu+"].mass, P["J/psi"].mass
    eps = 4 * m_mu**2 / m_psi**2
    reference = (4 / 15 + 2 * eps / 5) / (4 / 3 + 2 * eps / 3)
    tolerance = 4 * np.sqrt((1 / 7 - 1 / 25) / 4000)
    b = four_momentum(P["B+"].mass, [1.0, -2.0, 6.0])
    jpsi, kaon = _kin.two_body_decay(b, m_psi, P["K+"].mass, rng)
    c2 = []
    for _ in range(4000):
        out = decay_momenta("J/psi", jpsi, ("mu+", "mu-"), rng, polarised_by=b)
        _check_decay(out, jpsi, [m_mu, m_mu])
        c2.append(_helicity_cos(jpsi, out[0], b) ** 2)
    assert np.mean(c2) == pytest.approx(reference, abs=tolerance)
    # The shape, not only the mean: in |cos| bins of 0.2, sin^2 gives fractions
    # proportional to the integral of (1 - c^2) over each bin.
    edges = np.linspace(0.0, 1.0, 6)
    counts = np.histogram(np.sqrt(c2), edges)[0]
    expected = np.diff(edges - edges**3 / 3) / (2 / 3) * len(c2)
    assert np.sum((counts - expected) ** 2 / expected) < 20.0  # chi^2, 5 bins (p ~ 1e-3)
    # Without the B (a J/psi from elsewhere) the decay stays isotropic: <cos^2> = 1/3.
    c2 = [_helicity_cos(jpsi, decay_momenta("J/psi", jpsi, ("mu+", "mu-"), rng)[0], b) ** 2 for _ in range(4000)]
    assert np.mean(c2) == pytest.approx(1 / 3, abs=4 * np.sqrt((1 / 5 - 1 / 9) / 4000))


def test_jpsi_polarisation_in_custom_b_events():
    # End to end: B+ -> J/psi K+ placed in the Custom window, as events make it.
    # The muons from the J/psi follow sin^2 (<cos^2> = 0.2009, see above;
    # n = 600: 4 sigma = 0.030). A J/psi from B+ -> J/psi K+ pi+ pi- is not
    # recognised as two-body (recoil is a K pi pi mass, not a kaon's).
    rng = np.random.default_rng(96)
    entry = _decay_entry("B+", "J/psi K+", 20.0, where="start")
    c2 = []
    while len(c2) < 600:
        event = custom.generate_custom(Setup([entry]), 3.8, rng, quick=True)
        k = next(i for i, t in enumerate(event.tracks) if t.name == "J/psi")
        b_track = event.tracks[event.tracks[k].parent]
        b = four_momentum(P["B+"].mass, b_track.p4_end[1:])
        kids = _children(event, k)
        lepton = next(t for t in kids if t.name in ("mu+", "e+"))
        c2.append(_helicity_cos(event.tracks[k].p4, lepton.p4, b) ** 2)
    assert np.mean(c2) == pytest.approx(0.2009, abs=4 * np.sqrt((1 / 7 - 1 / 25) / 600))
    b = four_momentum(P["B+"].mass, [0.0, 0.0, 4.0])
    out = decay_momenta("B+", b, ("J/psi", "K+", "pi+", "pi-"), rng)
    event = Event()
    event.tracks.append(type("T", (), {"name": "B+", "p4_end": b})())
    assert _events._b_to_jpsi_k(event, "J/psi", out[0], 0) is None
    out = decay_momenta("B+", b, ("J/psi", "K+"), rng)
    assert _events._b_to_jpsi_k(event, "J/psi", out[0], 0) is not None


# --- Flavour oscillation of B0, Bs, D0 (particle_events.decay_flavour)
from physengine.modules.particle_data import MIXING  # noqa: E402
from physengine.modules.particle_tracks import C as _C  # noqa: E402


def test_mixing_parameters_give_the_pdg_chi():
    # chi = (x^2 + y^2) / (2 (1 + x^2)) (|q/p| = 1). References, PDG 2024:
    # chi_d = 0.1860 +- 0.0011 (b.txt:1466), chi_s = 0.499318 +- 0.000005
    # (bs.txt:32). D0: x = 0.00409, y = 0.00697 give 3.27e-5 (no PDG chi_D;
    # the reference is the formula with charm.txt:462, 464).
    for name, chi, tol in (("B0", 0.1860, 0.0011), ("Bs", 0.499318, 0.00001), ("D0", 3.2657e-5, 1e-8)):
        x, y = MIXING[name]
        assert (x * x + y * y) / (2 * (1 + x * x)) == pytest.approx(chi, abs=tol)
        # And the time-dependent P(flip | t) averaged over exp(-t / tau) gives
        # the same chi (to O(y^2 / x^2) for the Bs, where the exact time
        # distribution would carry cosh(y u)): numerical integral in u = t / tau.
        u = np.linspace(0.0, 60.0, 600001)
        tau = P[name].ctau / _C
        flip = np.array([_events.flip_probability(name, t) for t in u[::100] * tau])
        integral = np.trapezoid(np.exp(-u[::100]) * flip, u[::100])
        assert integral == pytest.approx(chi, rel=2e-3)
    assert _events.flip_probability("B+", 1e-3) == 0.0 and _events.flip_probability("K0S", 1e-3) == 0.0


def test_flipped_flavour_fraction_matches_chi():
    # Proper times drawn as events draw them (exponential, mean c tau / c);
    # fraction decaying as the antiparticle vs chi_d = 0.1860 and
    # chi_s = 0.499318 (PDG 2024, b.txt:1466, bs.txt:32). n = 200000:
    # binomial 4 sigma = 0.0035 (B0), 0.0045 (Bs).
    rng = np.random.default_rng(97)
    n = 200000
    for name, chi in (("B0", 0.1860), ("Bs", 0.499318), ("B0bar", 0.1860), ("Bsbar", 0.499318)):
        times = rng.exponential(P[name].ctau / _C, n)
        flipped = sum(_events.decay_flavour(name, t, rng) != name for t in times)
        assert flipped / n == pytest.approx(chi, abs=4 * np.sqrt(chi * (1 - chi) / n))
    # D0: chi_D = 3.3e-5, so of 200000 D0 about 6.5 flip (Poisson; < 25 allowed).
    times = rng.exponential(P["D0"].ctau / _C, n)
    assert sum(_events.decay_flavour("D0", t, rng) == "D0bar" for t in times) < 25


def test_b0_events_flavour_specific_modes_flip_and_cp_modes_do_not():
    # B0 made in events (_produce), its decay mode read off its daughters.
    # Flavour-specific modes (in only one of the two tables, e.g. D- l+ nu vs
    # D+ l- nu): the share in B0bar's table is chi_d = 0.1860 (PDG 2024,
    # b.txt:1466); n = 1500 B0: 4 sigma = 0.040. CP eigenstates (J/psi K0S,
    # J/psi K0L) are in both tables with the same fraction: unaffected.
    rng = np.random.default_rng(98)
    own = {tuple(sorted(d)) for _, d in P["B0"].decays}
    anti = {tuple(sorted(d)) for _, d in P["B0bar"].decays}
    flipped = specific = cp = 0
    n = 1500
    for _ in range(n):
        event = Event()
        _produce(event, "B0", four_momentum(P["B0"].mass, [0.0, 0.0, 1.0]), np.zeros(3), 0.0, 0.0, rng,
                 quick=True)  # fmt: skip
        if event.tracks[0].end != "decay":
            continue
        mode = tuple(sorted(t.name for t in _children(event, 0)))
        if mode in own and mode in anti:
            cp += 1
        elif mode in anti:
            flipped += 1
            specific += 1
        else:
            assert mode in own
            specific += 1
    assert flipped / specific == pytest.approx(0.1860, abs=4 * np.sqrt(0.186 * 0.814 / specific))
    # A self-conjugate mode (a CP eigenstate or a C-symmetric final state:
    # J/psi K0S, J/psi K0L, Ds+ Ds-, K+ K-, pi+ pi- pi0, ...) has the same
    # probability in both tables, so a flip leaves its rate unchanged. (Modes
    # like D0 -> K0S K- pi+ are neither: the flip swaps them with K0S K+ pi-.)
    assert cp + specific > 0.9 * n
    from physengine.modules.particle_data import _ANTI as anti_names

    for name in ("B0", "Bs", "D0"):
        a = dict(zip(map(lambda d: tuple(sorted(d)), P[name].modes()[1]), P[name].modes()[0], strict=True))
        b = dict(zip(map(lambda d: tuple(sorted(d)), P[ANTI_HEAVY[name]].modes()[1]), P[ANTI_HEAVY[name]].modes()[0], strict=True))
        both = {m for m in a if tuple(sorted(anti_names.get(d, d) for d in m)) == m}
        assert both  # e.g. J/psi K0S, Ds+ Ds-, K+ K-
        for mode in both:
            assert a[mode] == pytest.approx(b[mode], rel=1e-12), (name, mode)


# ----------------------------------------------------------------- D* mesons (added 2026-11-23)

# PDG 2024 (S. Navas et al., PRD 110, 030001), scratchpad pdg/ text files:
# D*(2010)+ mass 2010.26 MeV (charm.txt:1342), width 83.4 keV (charm.txt:1345),
# D0 pi+ 67.7%, D+ pi0 30.7%, D+ gamma 1.6% (charm.txt:1351-1353);
# D*(2007)0 mass 2006.85 MeV (charm.txt:1324), width only < 2.1 MeV (charm.txt:1326),
# D0 pi0 64.7%, D0 gamma 35.3% (charm.txt:1332-1333);
# Ds*+ mass 2112.2 MeV (ds.txt:444), width only < 1.9 MeV (ds.txt:447),
# Ds+ gamma 93.6%, Ds+ pi0 5.77% (ds.txt:454-456).
D_STAR = {
    "D*+": (
        2.01026, 83.4e-6, +1, {("D0", "pi+"): 0.677, ("D+", "pi0"): 0.307, ("D+", "gamma"): 0.016}
    ),
    "D*0": (2.00685, 0.0, 0, {("D0", "pi0"): 0.647, ("D0", "gamma"): 0.353}),
    "Ds*+": (2.1122, 0.0, +1, {("Ds+", "gamma"): 0.936, ("Ds+", "pi0"): 0.0577}),
}


def test_d_star_mesons_have_pdg_masses_widths_and_fractions():
    for name, (m, width, charge, table) in D_STAR.items():
        p = P[name]
        assert (p.mass, p.width, p.charge, p.ctau) == (m, width, charge, 0.0)
        assert not p.partial
        assert dict((d, b) for b, d in p.decays) == table
        probabilities, modes = p.modes()  # complete: no ValueError
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)
        anti = P[{"D*+": "D*-", "D*0": "D*0bar", "Ds*+": "Ds*-"}[name]]
        assert anti.charge == -charge and len(anti.decays) == len(p.decays)
    # Ds*+: the 0.63% not listed is the stand-in Ds+ e+ e- (measured (6.7 +- 1.6)e-3,
    # ds.txt:457-458).
    assert P["Ds*+"].rest == ("Ds+", "e+", "e-")
    assert P["Ds*-"].rest == ("Ds-", "e-", "e+")
    assert P["D*-"].decays[0] == (0.677, ("D0bar", "pi-"))


def test_d_star_plus_mass_from_d0_pi_follows_its_breit_wigner():
    # D*- made in B0 -> D*- pi+ and decayed to D0bar pi- through the functions
    # events use. Reference: m0 = 2010.26 MeV, Gamma = 83.4 keV (PDG 2024,
    # charm.txt:1342, 1345). The D0bar pi- mass is the D*'s; its median must be
    # m0 (Cauchy median error pi Gamma / (4 sqrt(n)) = 1.0 keV for n = 4000;
    # allowed 5 keV), half its interquartile range Gamma/2 x 0.951 = 39.7 keV
    # (the line shape cut at +-10 widths: tan(arctan(20) / 2) = 0.951), allowed
    # +-15%, and every mass within 10 widths.
    rng = np.random.default_rng(131)
    b0 = four_momentum(P["B0"].mass, [1.0, -2.0, 15.0])
    masses = []
    for _ in range(4000):
        d_star, _pion = decay_momenta("B0", b0, ("D*-", "pi+"), rng)
        out = decay_momenta("D*-", d_star, ("D0bar", "pi-"), rng)
        np.testing.assert_allclose(sum(out), d_star, atol=1e-9)
        masses.append(float(mass(sum(out))))
    masses = np.array(masses)
    m0, width = 2.01026, 83.4e-6
    assert np.all(np.abs(masses - m0) <= 10.0 * width + 1e-12)
    assert np.median(masses) == pytest.approx(m0, abs=5e-6)
    q25, q75 = np.percentile(masses, [25, 75])
    assert 0.5 * (q75 - q25) == pytest.approx(0.951 * width / 2, rel=0.15)


def test_b_to_d_star_l_nu_rate_from_cln_matches_the_measured_fraction():
    # B(B0 -> D*- e+ nu) = tau_B0 / hbar x integral of
    #   dGamma/dq^2 = G_F^2 (eta_EW F(1)|V_cb|)^2 p q^2 / (96 pi^3 M^2) (1 - m_l^2/q^2)^2
    #                 [(H+^2 + H-^2 + H0^2)(1 + m_l^2/2q^2) + 3 m_l^2/(2 q^2) Ht^2]
    # (Korner & Schuler 1990, Z. Phys. C 46, 93), H from CLN (Caprini, Lellouch,
    # Neubert 1998, NPB 530, 153, eqs. (36), (38)) with the HFLAV spring 2021
    # B -> D* CLN averages rho^2 = 1.121, R1(1) = 1.269, R2(1) = 0.853,
    # eta_EW F(1)|V_cb| = 35.00e-3 (hflav-eos.web.cern.ch,
    # semi/spring21/html/ExclusiveVcb/exclBtoDstar.html). G_F = 1.1663788e-5
    # GeV^-2, hbar = 6.582119569e-25 GeV s (PDG constants), tau_B0 = 1.517 ps
    # (PDG 2024). Reference: PDG 2024 B0 -> D*(2010)- l+ nu = (4.90 +- 0.12)%
    # (b.txt:1804); this gives 4.855%: required within the PDG's error.
    model = _events.SEMILEPTONIC_MODELS[("B0", "D*-")]
    assert model == ("CLN-V", 1.121, 1.269, 0.853)
    assert _events.SEMILEPTONIC_MODELS[("B+", "D*0bar")] == model
    assert _events.SEMILEPTONIC_MODELS[("B0bar", "D*+")] == model
    M, m, ml = P["B0"].mass, P["D*-"].mass, P["e+"].mass
    q2 = np.linspace(ml**2, (M - m) ** 2, 100001)[1:-1]
    hp, hm, h0, ht = _kin.vector_helicity_amplitudes(model, M, m, q2)
    p = _kin._momentum(M, np.sqrt(q2), m)
    eps = ml**2 / q2
    rate = (
        1.1663788e-5**2 * 35.00e-3**2 * p * q2 / (96 * np.pi**3 * M**2) * (1 - eps) ** 2
        * ((hp**2 + hm**2 + h0**2) * (1 + eps / 2) + 1.5 * eps * ht**2)
    )  # fmt: skip
    fraction = np.trapezoid(rate, q2) * 1.517e-12 / 6.582119569e-25
    assert fraction == pytest.approx(0.0490, abs=0.0012)
    # b -> c is left-handed: |H-| > |H+| away from zero recoil (they are equal at w = 1).
    assert np.all(hm[:-10] > hp[:-10])


def test_b_to_d_star_l_nu_density_integrates_to_the_q2_spectrum():
    # The (q^2, cos theta) density of the sampler, integrated over cos theta,
    # must have the shape of the dGamma/dq^2 above (Korner & Schuler 1990), for
    # e and mu and B0 -> D*- and B+ -> D*0bar (masses PDG 2024).
    for parent, meson in (("B0", "D*-"), ("B+", "D*0bar")):
        model = _events.SEMILEPTONIC_MODELS[(parent, meson)]
        M, m = P[parent].mass, P[meson].mass
        for lepton in ("e+", "mu+"):
            ml = P[lepton].mass
            q2 = np.linspace(ml**2, (M - m) ** 2, 41)[1:-1]
            c = np.linspace(-1.0, 1.0, 801)
            density = _kin.semileptonic_density(model, M, m, ml, q2[:, None], c[None, :])
            numeric = np.trapezoid(density, c, axis=1)
            hp, hm, h0, ht = _kin.vector_helicity_amplitudes(model, M, m, q2)
            eps = ml**2 / q2
            spectrum = (
                _kin._momentum(M, np.sqrt(q2), m) * q2 * (1 - eps) ** 2
                * ((hp**2 + hm**2 + h0**2) * (1 + eps / 2) + 1.5 * eps * ht**2)
            )  # fmt: skip
            ratio = numeric / spectrum
            np.testing.assert_allclose(ratio / ratio.mean(), 1.0, atol=1e-5)


def test_b_to_d_star_l_nu_events_have_a_hard_lepton():
    # V - A in b -> c: the term H-^2 (1 + cos theta_W)^2 dominates, so the
    # charged lepton goes mostly along the W (theta_W: its angle in the l nu
    # frame to the W's direction in the B frame). Checked: the events' mean
    # cos theta_W, for e+ from B0 -> D*- e+ nu, equals the density's own mean
    # (computed on a grid) within 0.03 (statistical error 0.55 / sqrt(4000)
    # = 0.009), and that mean is positive.
    rng = np.random.default_rng(133)
    model = _events.SEMILEPTONIC_MODELS[("B0", "D*-")]
    M, m, ml = P["B0"].mass, P["D*-"].mass, P["e+"].mass
    b0 = four_momentum(M, [0.0, 0.0, 0.0])
    cosines = []
    for _ in range(4000):
        meson, lepton, nu = decay_momenta("B0", b0, ("D*-", "e+", "nu"), rng)
        w = lepton + nu
        to_w = np.array([w[0], *(-w[1:])])
        lep = boost(lepton, to_w)[1:]
        axis = w[1:] / np.linalg.norm(w[1:])
        cosines.append(float(lep @ axis / np.linalg.norm(lep)))
    q2 = np.linspace(ml**2, (M - m) ** 2, 401)[1:-1]
    c = np.linspace(-1.0, 1.0, 401)
    density = _kin.semileptonic_density(model, M, m, ml, q2[:, None], c[None, :])
    expected = -np.sum(density * c[None, :]) / np.sum(density)  # cos theta_W = -cos theta
    assert expected > 0.1
    assert np.mean(cosines) == pytest.approx(expected, abs=0.03)


# --- Isobar model of three-body D decays (particle_events.isobar_decay).
# Reference values, PDG 2024 (S. Navas et al., Phys. Rev. D 110, 030001), as
# text in the session scratchpad pdg/ files (the line of each):
# - phi(1020): m = 1019.461 MeV, Gamma = 4.249 MeV (light.txt:351-352).
# - K*(892)+- hadroproduced: m = 891.67 MeV, Gamma = 51.4 MeV (strange.txt:494, 497).
# - K*(892)0: m = 895.55 MeV, Gamma = 47.3 MeV (strange.txt:496, 499).
# - Ds+ -> K+ K- pi+ 5.37% (ds.txt:161); phi pi+, phi -> K+ K- 2.21%
#   (ds.txt:163); K+ K*0bar, K*0bar -> K- pi+ 2.58% (ds.txt:166-167).
# - D0 -> K- pi+ pi0 14.4% (charm.txt:791); K- rho+ 11.2% (charm.txt:792);
#   K*- pi+, K*- -> K- pi0 2.31% (charm.txt:794-796); K*0bar pi0, K*0bar ->
#   K- pi+ 1.95% (charm.txt:797-799). These add up to 15.46% > 14.4%: without
#   interference the module renormalises them to their sum (particle_data.ISOBARS).
# - D+ -> K- pi+ pi+ 9.38% (charm.txt:183); K*0bar pi+, K*0bar -> K- pi+
#   1.04% (charm.txt:190-193).
# Helicity angle of a spin-0 -> vector + spin-0 decay, vector -> two spin-0:
# cos^2(theta) (d^1_00 = cos theta, Jacob & Wick 1959), so <cos^2> = 3/5.
_ISOBAR_MODES = {
    "Ds+": ("K+", "K-", "pi+"),
    "D0": ("K-", "pi+", "pi0"),
    "D+": ("K-", "pi+", "pi+"),
}


def _peak(values, low, high, bins):
    """Centre of the fullest bin of a histogram of `values` over [low, high)."""
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    k = int(np.argmax(counts))
    return 0.5 * (edges[k] + edges[k + 1])


def test_ds_to_kkpi_kk_mass_peaks_at_the_phi():
    # (a) All Ds+ -> K+ K- pi+ decays (the path events use): the K+ K- mass
    # histogram (1 MeV bins) peaks within the phi width of 1019.461 MeV.
    rng = np.random.default_rng(301)
    parent = four_momentum(P["Ds+"].mass, [0.4, 2.0, -1.0])
    mode = _ISOBAR_MODES["Ds+"]
    kk = [float(mass(sum(decay_momenta("Ds+", parent, mode, rng)[:2]))) for _ in range(6000)]
    assert abs(_peak(kk, 0.98, 1.06, 80) - 1.019461) < 0.004249


def test_d0_to_kpipi0_kpi0_mass_peaks_at_the_charged_kstar():
    # (b) All D0 -> K- pi+ pi0 decays: the K- pi0 mass (the K*(892)- pi+
    # channel; 20 MeV bins over 0.6 to 1.2 GeV, under it the reflection of
    # the rho+ and K*0bar channels) peaks within the K*+- width of 891.67 MeV.
    # And the events drawn through the K*- alone have a median K- pi0 mass
    # within half a width of it.
    rng = np.random.default_rng(302)
    parent = four_momentum(P["D0"].mass, [0.4, 2.0, -1.0])
    all_masses, kstar = [], []
    for _ in range(12000):
        out, resonance = _events.isobar_decay("D0", parent, _ISOBAR_MODES["D0"], rng)
        m = float(mass(out[0] + out[2]))
        all_masses.append(m)
        if resonance == "K*-":
            kstar.append(m)
    assert abs(_peak(all_masses, 0.6, 1.2, 30) - 0.89167) < 0.0514
    assert abs(float(np.median(kstar)) - 0.89167) < 0.5 * 0.0514


def test_isobar_decays_conserve_four_momentum():
    # (c) As test_heavy_meson_decays_conserve_four_momentum: every isobar mode
    # and its conjugate, through every sub-channel and phase space, from a
    # moving parent: energy and momentum conserved, daughters on their mass
    # shells, and the parent's PDG mass from the sum.
    rng = np.random.default_rng(303)
    modes = dict(_ISOBAR_MODES)
    modes.update({"Ds-": ("K-", "K+", "pi-"), "D0bar": ("K+", "pi-", "pi0")})
    modes["D-"] = ("K+", "pi-", "pi-")
    for name, daughters in modes.items():
        parent = four_momentum(P[name].mass, [0.4, 2.0, -1.0])
        seen = set()
        for _ in range(400):
            out, resonance = _events.isobar_decay(name, parent, daughters, rng)
            seen.add(resonance)
            _check_decay(out, parent, [P[d].mass for d in daughters])
            assert float(mass(sum(out))) == pytest.approx(P[name].mass, rel=1e-9)
        expected = {r for r, _ in _events.isobar_probabilities(name, daughters)}
        if name not in ("D0", "D0bar"):
            expected.add(None)  # phase space takes the rest (none for the D0: renormalised)
        assert seen == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Ds+", {"phi": 2.21 / 5.37, "K*0bar": 2.58 / 5.37}),
        ("D0", {"rho+": 11.2 / 15.46, "K*-": 2.31 / 15.46, "K*0bar": 1.95 / 15.46}),
        ("D+", {"K*0bar": 1.04 / 9.38}),
    ],
)
def test_isobar_channel_fractions_match_the_pdg(name, expected):
    # (d) The share of decays drawn through each resonance matches the PDG
    # sub-channel fraction over the mode's (D0: over the sum of the three,
    # see above) within 3 sigma, sigma = sqrt(f (1 - f) / n).
    rng = np.random.default_rng(304)
    parent = four_momentum(P[name].mass, [0.0, 0.0, 0.0])
    n = 8000
    drawn = [_events.isobar_decay(name, parent, _ISOBAR_MODES[name], rng)[1] for _ in range(n)]
    for resonance, f in expected.items():
        share = drawn.count(resonance) / n
        assert abs(share - f) < 3.0 * np.sqrt(f * (1.0 - f) / n), resonance


def test_isobar_vector_decays_with_cos2_helicity_angle():
    # The spin structure: through the phi, the K+'s angle in the phi rest
    # frame to the phi's flight direction has <cos^2> = 3/5 (cos^2 law), not
    # 1/3 (isotropic) or 1/5 (sin^2). 4000 decays: sigma(<cos^2>) = 0.3 / sqrt(n
    # in the phi channel) < 0.008.
    rng = np.random.default_rng(305)
    parent = four_momentum(P["Ds+"].mass, [0.4, 2.0, -1.0])
    c2 = []
    for _ in range(4000):
        out, resonance = _events.isobar_decay("Ds+", parent, _ISOBAR_MODES["Ds+"], rng)
        if resonance == "phi":
            c2.append(_helicity_cos(out[0] + out[1], out[0], parent) ** 2)
    assert np.mean(c2) == pytest.approx(0.6, abs=0.03)


def test_isobar_order_of_daughters_does_not_matter():
    # A decay typed in another order (the Custom window) goes through the
    # same sub-channels; for D+ -> K- pi+ pi+ either pi+ comes from the K*0bar.
    rng = np.random.default_rng(306)
    assert _events.isobar_probabilities("Ds+", ("pi+", "K-", "K+")) == _events.isobar_probabilities(
        "Ds+", ("K+", "K-", "pi+")
    )
    parent = four_momentum(P["D+"].mass, [0.0, 0.0, 0.0])
    which = []
    for _ in range(3000):
        out, resonance = _events.isobar_decay("D+", parent, ("K-", "pi+", "pi+"), rng)
        if resonance == "K*0bar":
            m1, m2 = float(mass(out[0] + out[1])), float(mass(out[0] + out[2]))
            which.append(abs(m1 - 0.89555) < abs(m2 - 0.89555))
    assert 0.35 < np.mean(which) < 0.65


def test_face_each_other_separates_origin_starts():
    # Two entries at the origin are put 1.0 m apart on the beam axis (z = -0.5, +0.5)
    # and aimed at each other: direction . unit(start_other - start_own) = 1.
    pytest.importorskip("imgui_bundle")
    pytest.importorskip("moderngl")
    from physengine.modules import particle_editor

    a, b = Entry(name="mu-"), Entry(name="mu+")
    particle_editor.face_each_other(a, b)
    sa, sb = np.array(a.start, float), np.array(b.start, float)
    assert not np.allclose(sa, sb)
    assert abs(np.linalg.norm(sb - sa) - 1.0) < 1e-9
    for own, other, entry in ((sa, sb, a), (sb, sa, b)):
        u = (other - own) / np.linalg.norm(other - own)
        assert abs(float(np.dot(custom._direction(entry)[0], u)) - 1.0) < 1e-9


def test_custom_window_sections_and_controls_by_kind(monkeypatch):
    # Layout checks without a window: a fake imgui records the section headers
    # the Custom event window draws and runs its build functions for a fire
    # entry and a decay entry. The look of the window is not tested here.
    pytest.importorskip("imgui_bundle")
    from physengine.modules import particle_custom as custom
    from physengine.modules import particle_editor as editor

    assert editor.SECTIONS == (
        ("What", True), ("Where", False), ("How many", False), ("Face each other", False),
    )
    fire, decay = editor.controls_for("fire"), editor.controls_for("decay")
    assert "decay mode" not in fire and "daughters" not in fire
    assert {"decay mode", "daughters", "where it decays"} <= set(decay)
    assert set(fire) <= set(decay)

    headers = []

    class Fake:
        Cond_ = SliderFlags_ = WindowFlags_ = type("E", (), {"__getattr__": lambda s, n: 0})()

        def collapsing_header(self, title, *a):
            headers.append(title)
            return True

        def combo(self, label, index, items, *a):
            return False, index

        def input_text(self, label, text, *a):
            return False, text

        def slider_float(self, label, v, *a):
            return False, v

        slider_int = slider_float

        def drag_float3(self, label, v, *a):
            return False, v

        def __getattr__(self, name):
            return lambda *a, **k: (False if name in ("radio_button", "button") else (False, 0))

    fake = Fake()
    monkeypatch.setattr(editor, "imgui", fake)
    window = editor.CustomWindow(type("M", (), {"field": 2.0})())
    setup = custom.preset(custom.PRESETS[0])
    setup.entries.append(editor.Entry("fire", "mu-", value=10.0))
    setup.entries.append(editor.Entry("decay", "K0S", value=5.0, daughters="pi+ pi-"))
    for entry in setup.entries[-2:]:
        headers.clear()
        window._entry_controls(entry, setup)
        assert headers == [t for t, _ in editor.SECTIONS]


# ------------------------------------------------- custom collisions (pair -> Z)
#
# Reference values (PDG 2024, Z boson listing, as in particle_data.py):
# M_Z = 91.1876 GeV, Gamma_Z = 2.4952 GeV. s = (p1 + p2)^2 (PDG 2024,
# "Kinematics" review); head on with energies E1, E2 and negligible masses,
# sqrt(s) = 2 sqrt(E1 E2). Formation weight: the non-relativistic
# Breit-Wigner of PDG 2024, "Resonances" review, scaled to 1 at the peak:
# w = (G/2)^2 / ((sqrt(s) - M)^2 + (G/2)^2), so w(M) = 1, w(M +- G/2) = 1/2,
# w(M + G) = 1/5. For sqrt(s) uniform in M -+ 10 G, the formed pairs follow
# the Breit-Wigner truncated to that window: the fraction within G/2 of the
# peak is atan(1) / atan(20) = 0.51643 (1/2 for the untruncated line shape).

M_Z, G_Z = 91.1876, 2.4952


def _colliding_pair(e_a, e_b, names=("e-", "e+"), collide=True, spread_p=0.0):
    """Two fired leptons 1 m apart on the beam axis, aimed at each other
    (z = -0.5 m moving +z, z = +0.5 m moving -z), given by total energy."""
    a = Entry("fire", names[0], quantity="energy", value=e_a, theta=0.0,
              start=(0.0, 0.0, -0.5), spread_p=spread_p)  # fmt: skip
    b = Entry("fire", names[1], quantity="energy", value=e_b, theta=180.0,
              start=(0.0, 0.0, 0.5), spread_p=spread_p)  # fmt: skip
    setup = Setup([a, b])
    if collide:
        custom.pair_up(setup, 0, 1)
    return setup


def _fired(entry):
    kin = custom.kinematics(entry, 3.8)
    return four_momentum(P[entry.name].mass, kin.momentum * np.array(kin.direction))


def test_breit_wigner_formation_weight():
    from physengine.modules.particle_kinematics import breit_wigner_formation

    assert breit_wigner_formation(M_Z, M_Z, G_Z) == pytest.approx(1.0)
    assert breit_wigner_formation(M_Z + G_Z / 2, M_Z, G_Z) == pytest.approx(0.5)
    assert breit_wigner_formation(M_Z - G_Z / 2, M_Z, G_Z) == pytest.approx(0.5)
    assert breit_wigner_formation(M_Z + G_Z, M_Z, G_Z) == pytest.approx(0.2)


@pytest.mark.parametrize("names", [("e-", "e+"), ("mu-", "mu+")])
def test_collision_at_the_z_peak_forms_a_z_and_its_mass_peaks_there(names):
    # (a) Both at E = M_Z / 2 head on: sqrt(s) = 91.1876 GeV (exactly, as
    # 2 E for equal energies); w = 1, so every event forms a Z. Its mass is
    # sqrt(s); the mass measured from e+ e- / mu+ mu- daughters peaks at M_Z
    # within the width (detector resolution about 1 to 2%).
    setup = _colliding_pair(M_Z / 2, M_Z / 2, names)
    assert custom.problems(setup) == []
    assert custom.nominal_sqrt_s(*setup.entries) == pytest.approx(M_Z, abs=1e-9)
    rng = np.random.default_rng(81)
    measured = []
    for _ in range(120):
        event = custom.generate_custom(setup, 3.8, rng)
        (record,) = event.collisions
        assert record["formed"] and record["within"] and record["probability"] == 1.0
        assert record["sqrt_s"] == pytest.approx(M_Z, abs=1e-9)
        z = event.tracks[record["index"]]
        assert z.name == "Z" and z.parent == -1
        assert float(mass(z.p4)) == pytest.approx(M_Z, abs=1e-9)
        # The two fired particles are absorbed: no track of them.
        assert [t.name for t in event.tracks if t.parent < 0] == ["Z"]
        assert "not a collision" not in event.headline and "forms a Z" in event.headline
        measured += [m for label, m in event.masses if "visible" not in label]
    assert len(measured) > 40
    assert abs(float(np.median(measured)) - M_Z) < G_Z


def test_formed_collisions_follow_the_breit_wigner():
    # (b) sqrt(s) uniform over M -+ 10 G (head-on pairs with E = sqrt(s) / 2):
    # the formed ones follow the truncated Breit-Wigner, fraction within G/2
    # of the peak atan(1) / atan(20) = 0.51643, checked to 3 sigma (binomial).
    # And at sqrt(s) = M + G/2 the formed fraction is w = 1/2.
    rng = np.random.default_rng(82)
    m_e = P["e-"].mass
    formed = []
    for root_s in rng.uniform(M_Z - 10 * G_Z, M_Z + 10 * G_Z, 40000):
        e = root_s / 2
        p = np.sqrt(e * e - m_e * m_e)
        record = custom.collide(np.array([e, 0, 0, p]), np.array([e, 0, 0, -p]), "Z", rng)
        assert record["within"]
        if record["formed"]:
            formed.append(record["sqrt_s"])
    formed = np.array(formed)
    expected = np.arctan(1.0) / np.arctan(20.0)
    assert expected == pytest.approx(0.51643, abs=1e-5)
    share = float(np.mean(np.abs(formed - M_Z) < G_Z / 2))
    sigma = np.sqrt(expected * (1 - expected) / len(formed))
    assert abs(share - expected) < 3 * sigma
    # Half maximum, through the whole generation (momentum fixed, so sqrt(s) too).
    setup = _colliding_pair((M_Z + G_Z / 2) / 2, (M_Z + G_Z / 2) / 2)
    n, hits = 600, 0
    for _ in range(n):
        event = custom.generate_custom(setup, 0.0, rng)
        hits += event.collisions[0]["formed"]
    assert abs(hits / n - 0.5) < 3 * np.sqrt(0.25 / n)


def test_collision_conserves_four_momentum_and_meets_halfway():
    # (c) Unequal energies (60 GeV and M^2 / 240 GeV, sqrt(s) = M up to the
    # electron mass): the Z's four-momentum is the sum of the two fired ones,
    # to 1e-8 of the energy, and so is the sum of its daughters'. Both move at
    # beta = 1 to 1e-10, so they meet midway, z = 0, after 0.5 m / c.
    setup = _colliding_pair(60.0, M_Z**2 / 240.0)
    total = _fired(setup.entries[0]) + _fired(setup.entries[1])
    rng = np.random.default_rng(83)
    for _ in range(20):
        event = custom.generate_custom(setup, 3.8, rng)
        (record,) = event.collisions
        if not record["formed"]:
            continue
        z = event.tracks[record["index"]]
        assert np.max(np.abs(z.p4 - total)) < 1e-8 * total[0]
        assert float(mass(z.p4)) == pytest.approx(record["sqrt_s"], abs=1e-9)
        kids = [t.p4 for t in event.tracks if t.parent == record["index"]]
        assert np.max(np.abs(np.sum(kids, axis=0) - total)) < 1e-8 * total[0]
        assert np.allclose(record["point"], (0.0, 0.0, 0.0), atol=1e-6)
        assert record["time"] == pytest.approx(0.5 / _C, rel=1e-6)


def _tracks(event):
    return [(t.name, t.end, t.p4.tolist(), t.points.tolist()) for t in event.tracks]


@pytest.mark.parametrize("energy", [5.0, 3.0969 / 2])
def test_pair_far_from_the_z_passes_through(energy):
    # (d) sqrt(s) = 10 GeV, and at the J/psi mass 3.0969 GeV (no J/psi
    # formation is simulated): outside M_Z -+ 10 G, nothing is formed, and the
    # event is the one without the option for the same seed.
    on, off = _colliding_pair(energy, energy), _colliding_pair(energy, energy, collide=False)
    for seed in range(5):
        a = custom.generate_custom(on, 3.8, np.random.default_rng(seed))
        b = custom.generate_custom(off, 3.8, np.random.default_rng(seed))
        (record,) = a.collisions
        assert not record["within"] and not record["formed"] and record["probability"] == 0.0
        assert record["sqrt_s"] == pytest.approx(2 * energy, rel=1e-6)
        assert "they pass through" in custom.collision_caption(record)
        assert _tracks(a) == _tracks(b) and not b.collisions
        assert sorted(t.name for t in a.tracks if t.parent < 0) == ["e+", "e-"]


def test_collide_is_off_by_default_and_only_offered_for_pairs_that_form_a_z():
    # (e) Default off: Entry().collide is "", presets have no collisions, and
    # a faced e+ e- pair at the Z peak without the option makes no Z.
    assert Entry().collide == ""
    rng = np.random.default_rng(84)
    for name in custom.PRESETS:
        assert not custom.collision_groups(custom.preset(name))
    event = custom.generate_custom(_colliding_pair(M_Z / 2, M_Z / 2, collide=False), 3.8, rng)
    assert not event.collisions and "Z" not in {t.name for t in event.tracks}
    a, b = _colliding_pair(45.0, 45.0).entries
    assert custom.collision_problem(a, b) == ""
    for names in (("e-", "mu+"), ("e-", "e-"), ("pi-", "pi+")):
        x, y = _colliding_pair(45.0, 45.0, names, collide=False).entries
        assert "no resonance" in custom.collision_problem(x, y)
    off_axis = _colliding_pair(45.0, 45.0, collide=False)
    off_axis.entries[1].start = (0.3, 0.0, 0.5)
    assert "do not meet" in custom.collision_problem(*off_axis.entries)
    spread = _colliding_pair(45.0, 45.0, collide=False)
    spread.entries[0].spread_angle = 1.0
    assert "miss" in custom.collision_problem(*spread.entries)
    # A pair set to collide that cannot is refused; a lone one too.
    lone = _colliding_pair(45.0, 45.0)
    lone.entries[1].collide = ""
    assert any("no other entry" in t for t in custom.problems(lone))
    off_axis.entries[0].collide = off_axis.entries[1].collide = "c1"
    assert any("cannot collide" in t for t in custom.problems(off_axis))
    # Same start point: the paths meet there at once.
    same = _colliding_pair(M_Z / 2, M_Z / 2)
    same.entries[0].start = same.entries[1].start = (0.1, 0.2, 0.3)
    record = custom.generate_custom(same, 3.8, rng).collisions[0]
    assert record["formed"] and record["time"] == 0.0
    assert record["point"] == pytest.approx((0.1, 0.2, 0.3))
    # Saved and read back.
    setup = _colliding_pair(45.0, 45.0)
    assert custom.from_dict(custom.to_dict(setup)) == setup
    custom.unpair(setup, 1)
    assert not custom.collision_groups(setup)


def test_custom_window_collide_controls(monkeypatch):
    # The Collide checkbox section draws without a window (fake imgui) for a
    # pair that can collide, one set to collide, and one that cannot.
    pytest.importorskip("imgui_bundle")
    from physengine.modules import particle_editor as editor

    calls = []

    class Fake:
        def checkbox(self, label, value):
            calls.append(label)
            return False, value

        def text(self, s, *a):
            calls.append(s)

        text_disabled = text

        def __getattr__(self, name):
            return lambda *a, **k: (False, 0)

    monkeypatch.setattr(editor, "imgui", Fake())
    setup = _colliding_pair(M_Z / 2, M_Z / 2)
    event = custom.generate_custom(setup, 3.8, np.random.default_rng(85))
    window = editor.CustomWindow(type("M", (), {"field": 3.8, "event": event, "custom": setup})())
    monkeypatch.setattr(editor, "_selected", 0)
    monkeypatch.setattr(editor, "_partner", 1)
    window._collide(setup.entries[0], setup)
    assert "Collide##pair" in calls
    assert any(c.startswith("Forms a Z at sqrt(s) = 91.1876") for c in calls)
    calls.clear()
    window.m.event = None
    window._collide(setup.entries[0], setup)
    assert any("no event drawn yet" in c for c in calls)
    calls.clear()
    other = _colliding_pair(45.0, 45.0, ("mu-", "e+"), collide=False)
    window._collide(other.entries[0], other)
    assert "Collide##pair" not in calls and any("not offered" in c for c in calls)


def _fake_collide_window(monkeypatch, tick):
    """CustomWindow with a fake imgui whose Collide checkbox returns `tick`
    (changed, value) and records the text drawn."""
    pytest.importorskip("imgui_bundle")
    from physengine.modules import particle_editor as editor

    calls = []

    class Fake:
        def checkbox(self, label, value):
            return tick(value)

        def text(self, s, *a):
            calls.append(s)

        text_disabled = text

        def __getattr__(self, name):
            return lambda *a, **k: (False, 0)

    monkeypatch.setattr(editor, "imgui", Fake())
    monkeypatch.setattr(editor, "_selected", 0)
    monkeypatch.setattr(editor, "_partner", 1)
    return editor, calls


def test_ticking_collide_sets_half_z_momenta_once(monkeypatch):
    # Reference: PDG 2024 M_Z = 91.1876 GeV, so each beam is M_Z / 2 = 45.5938
    # GeV and sqrt(s) = 91.19 GeV (muon mass 0.1057 GeV shifts it by 2e-4 GeV).
    setup = _colliding_pair(20.0, 30.0, ("mu+", "mu-"), collide=False)
    editor, calls = _fake_collide_window(monkeypatch, lambda v: (True, not v))
    m = type("M", (), {"field": 3.8, "event": None, "custom": setup})()
    window = editor.CustomWindow(m)
    window._collide(setup.entries[0], setup)  # tick on
    assert custom.collision_partner(setup, 0) == 1
    for e in setup.entries:
        assert (e.quantity, e.value) == ("momentum", pytest.approx(45.5938, abs=1e-4))
    assert custom.nominal_sqrt_s(*setup.entries) == pytest.approx(91.19, abs=0.01)
    assert any(c.startswith("sqrt(s) = 91.19 GeV. Z window (91.19 +- 10 widths): inside") for c in calls)
    setup.entries[0].value = 40.0  # user edits afterwards; ticking off changes nothing
    window._collide(setup.entries[0], setup)  # tick off
    assert custom.collision_partner(setup, 0) is None
    assert setup.entries[0].value == 40.0 and setup.entries[1].value == pytest.approx(45.5938, abs=1e-4)
    # already on: no change on a non-ticking frame
    custom.pair_up(setup, 0, 1)
    editor2, _ = _fake_collide_window(monkeypatch, lambda v: (False, v))
    window._collide(setup.entries[0], setup)
    assert setup.entries[0].value == 40.0


def test_formation_share_caption_breit_wigner():
    # Gamma_Z = 2.4952 GeV, M_Z = 91.1876 GeV (particle_data.py, PDG 2024);
    # w = (G/2)^2 / ((s - M)^2 + (G/2)^2): 1 at the peak, ~0.01965 at 100 GeV.
    from physengine.modules import particle_editor as editor

    assert editor.formation_share_caption(91.1876).endswith("100.0 %")
    w = (G_Z / 2) ** 2 / ((100.0 - M_Z) ** 2 + (G_Z / 2) ** 2)
    assert w == pytest.approx(0.01965, abs=1e-5)
    assert editor.formation_share_caption(100.0).endswith(f"{100 * w:.1f} %")
    assert "2.0 %" in editor.formation_share_caption(100.0)
    assert editor.window_caption(100.0).endswith("inside.")  # 3.5 widths away
    assert editor.window_caption(130.0).endswith("outside.")  # 38.8 GeV > 10 widths = 24.95 GeV
