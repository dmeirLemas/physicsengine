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
"""

import csv
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from physengine.modules import particle_opendata
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
    histogram_masses,
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


# ------------------------------------------------- helpers for B decays


def _b_to_jpsi_k_event(momentum, rng):
    """A B+ of `momentum` GeV along x that decayed to J/psi K+ at the origin, the
    J/psi then followed as events follow it (_produce). The B+ is the first
    track (name and final four-momentum only), the J/psi the second. Returns
    the event and the B+ and K+ four-momenta."""
    b = four_momentum(P["B+"].mass, [momentum, 0.0, 0.0])
    jpsi, kaon = decay_momenta("B+", b, ("J/psi", "K+"), rng)
    event = Event()
    event.tracks.append(SimpleNamespace(name="B+", p4_end=b, parent=-1, origin="collision"))
    _produce(event, "J/psi", jpsi, np.zeros(3), 0.0, 3.8, rng, parent=0, generation=1,
             origin="decay", quick=True)  # fmt: skip
    return event, b, kaon


def _children(event, index):
    return [t for t in event.tracks if t.parent == index and t.origin == "decay"]


def test_b_plus_decay_length():
    # Mean decay length (p / m) c tau for a B+ of 20 GeV. Reference (PDG 2024):
    # m = 5279.41 MeV, c tau = 491.1 um, so (20 / 5.27941) x 491.1e-6 =
    # 1.8604e-3 m. Statistical error of the mean of 4000 exponentials: 1.6%;
    # 5% allowed. The decay time drawn is a lab time, so the flight is v t.
    expected = 20.0 / 5.27941 * 491.1e-6
    assert expected == pytest.approx(1.8604e-3, rel=1e-4)
    rng = np.random.default_rng(171)
    b = four_momentum(P["B+"].mass, [20.0, 0.0, 0.0])
    flights = [_decay_time(P["B+"], b, rng) * C * 20.0 / b[0] for _ in range(4000)]
    assert np.mean(flights) == pytest.approx(expected, rel=0.05)


def test_b_plus_mass_from_jpsi_k_products():
    # B+ -> J/psi K+, J/psi -> mu+ mu- (PDG 2024: m(B+) = 5279.41 MeV). From the
    # true momenta the mass is exact; from the tracker-measured momenta
    # (curvature smeared, particle_events.measured) it peaks at that mass:
    # the median must lie within 3 sigma (the measured width, robust: 1.4826
    # MAD) of 5.27941 GeV.
    rng = np.random.default_rng(172)
    true_m, meas_m = [], []
    while len(meas_m) < 400:
        event, _, kaon_p4 = _b_to_jpsi_k_event(20.0, rng)
        kids = _children(event, 1)
        if sorted(t.name for t in kids) != ["mu+", "mu-"]:
            continue
        kaon = Event()
        _produce(kaon, "K+", kaon_p4, np.zeros(3), 0.0, 3.8, rng, origin="decay")
        products = kids + [kaon.tracks[0]]
        true_m.append(float(mass(sum(t.p4 for t in products))))
        ps = [_events.measured(t, 3.8, rng) for t in products]
        if any(p is None for p in ps):
            continue
        p4 = sum(
            four_momentum(P[t.name].mass, p) for t, p in zip(products, ps, strict=True)
        )
        meas_m.append(float(mass(p4)))
    np.testing.assert_allclose(true_m, 5.27941, rtol=0.0, atol=1e-6)
    meas = np.array(meas_m)
    sigma = 1.4826 * np.median(np.abs(meas - np.median(meas)))
    assert 0.0 < sigma < 1.0
    assert abs(np.median(meas) - 5.27941) < 3.0 * sigma


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
            assert particle.partial
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
    counts = {}
    n = 400
    for _ in range(n):
        event, _, _ = _b_to_jpsi_k_event(20.0, rng)
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


def test_jpsi_polarisation_in_b_events():
    # End to end: B+ -> J/psi K+ followed as events follow it. The muons from
    # the J/psi follow sin^2 (<cos^2> = 0.2009, see above; n = 600: 4 sigma =
    # 0.030). A J/psi from B+ -> J/psi K+ pi+ pi- is not recognised as
    # two-body (recoil is a K pi pi mass, not a kaon's).
    rng = np.random.default_rng(96)
    c2 = []
    while len(c2) < 600:
        event, b, _ = _b_to_jpsi_k_event(20.0, rng)
        lepton = next(t for t in _children(event, 1) if t.name in ("mu+", "e+"))
        c2.append(_helicity_cos(event.tracks[1].p4, lepton.p4, b) ** 2)
    assert np.mean(c2) == pytest.approx(0.2009, abs=4 * np.sqrt((1 / 7 - 1 / 25) / 600))
    b = four_momentum(P["B+"].mass, [0.0, 0.0, 4.0])
    out = decay_momenta("B+", b, ("J/psi", "K+", "pi+", "pi-"), rng)
    event = Event()
    event.tracks.append(SimpleNamespace(name="B+", p4_end=b))
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
    # A decay typed in another order goes through the
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


# Z mass and width, PDG 2024, Z boson listing.
M_Z, G_Z = 91.1876, 2.4952


def test_breit_wigner_formation_weight():
    from physengine.modules.particle_kinematics import breit_wigner_formation

    assert breit_wigner_formation(M_Z, M_Z, G_Z) == pytest.approx(1.0)
    assert breit_wigner_formation(M_Z + G_Z / 2, M_Z, G_Z) == pytest.approx(0.5)
    assert breit_wigner_formation(M_Z - G_Z / 2, M_Z, G_Z) == pytest.approx(0.5)
    assert breit_wigner_formation(M_Z + G_Z, M_Z, G_Z) == pytest.approx(0.2)


def _tracks(event):
    return [(t.name, t.end, t.p4.tolist(), t.points.tolist()) for t in event.tracks]


# ------------------------------------- scenes with forced chains, H -> gamma gamma, e+ e- scan
#
# Reference masses, as in particle_data.py: B+ 5279.41 MeV and D0 1864.84
# MeV (PDG 2024, S. Navas et al., Phys. Rev. D 110, 030001), Higgs 125.25 GeV
# (PDG 2022), Z 91.1876 GeV with width 2.4952 GeV (PDG 2022).

NEW_SCENES = {
    "B+ -> J/psi K+": 5.27941,
    "D*+ -> D0 pi+": 1.86484,
    "Higgs -> 2 photons": 125.25,
}


def _decayed_into(event, index):
    """Sorted names and summed four-momentum of the decay products of track `index`."""
    kids = _children(event, index)
    return sorted(t.name for t in kids), sum(t.p4 for t in kids)


def _at_decay(track):
    """The four-momentum a track decays with, as _produce takes it: its momentum at
    the end, with its own mass (or, for a resonance, the mass it was made with)."""
    m = float(mass(track.p4)) if P[track.name].width > 0.0 else P[track.name].mass
    return four_momentum(m, track.p4_end[1:])


def test_forced_chains_conserve_four_momentum_at_every_forced_decay():
    # B+ -> J/psi K+, J/psi -> mu+ mu-; D*+ -> D0 pi+, D0 -> K- pi+; H -> gamma gamma:
    # the products of each forced decay add up to the parent's four-momentum
    # where it decays (energy and momentum conservation, exact to rounding).
    rng = np.random.default_rng(401)
    for _ in range(30):
        event = generate("B+ -> J/psi K+", 3.8, rng)
        b = event.tracks[0]
        assert b.name == "B+" and b.end == "decay"
        names, total = _decayed_into(event, 0)
        assert names == ["J/psi", "K+"]
        np.testing.assert_allclose(total, _at_decay(b), rtol=1e-9, atol=1e-9)
        j = next(i for i, t in enumerate(event.tracks) if t.name == "J/psi")
        names, total = _decayed_into(event, j)
        assert names == ["mu+", "mu-"]
        np.testing.assert_allclose(total, _at_decay(event.tracks[j]), rtol=1e-9, atol=1e-9)

        event = generate("D*+ -> D0 pi+", 3.8, rng)
        d_star = event.tracks[0]
        assert d_star.name == "D*+"
        # Its mass from its Breit-Wigner (width 83.4 keV, within 10 widths).
        assert abs(float(mass(d_star.p4)) - 2.01026) <= 10.0 * 83.4e-6 + 1e-9
        names, total = _decayed_into(event, 0)
        assert names == ["D0", "pi+"]
        np.testing.assert_allclose(total, _at_decay(d_star), rtol=1e-9, atol=1e-9)
        k = next(i for i, t in enumerate(event.tracks) if t.name == "D0")
        names, total = _decayed_into(event, k)
        assert names == ["K-", "pi+"]
        np.testing.assert_allclose(total, _at_decay(event.tracks[k]), rtol=1e-9, atol=1e-9)

        event = generate("Higgs -> 2 photons", 3.8, rng)
        (higgs,) = event.bosons
        photons = [t.p4 for t in event.tracks if t.parent < 0]
        assert [t.name for t in event.tracks if t.parent < 0] == ["gamma", "gamma"]
        np.testing.assert_allclose(photons[0] + photons[1], higgs["p4"], rtol=1e-9, atol=1e-9)
        assert float(mass(higgs["p4"])) == pytest.approx(125.25, abs=1e-6)


@pytest.mark.parametrize("scene", list(NEW_SCENES))
def test_new_scenes_reconstruct_their_mass(scene):
    # Measured mass (tracker model for mu+ mu- K+ and K- pi+, calorimeter
    # model for gamma gamma): the median within the measured width (half the
    # central 68%) of the reference mass, that width small (under 2% of the
    # mass), and most events give a mass (the others: a track too close to
    # the beam line, or a kaon or pion that decayed before the tracker
    # could measure it).
    rng = np.random.default_rng(402)
    reference = NEW_SCENES[scene]
    found = quick_masses(scene, 300, 3.8, rng)
    assert len(found) > 0.8 * 300
    half_width = 0.5 * (np.percentile(found, 84) - np.percentile(found, 16))
    assert 0.0 < half_width < 0.02 * reference
    assert abs(np.median(found) - reference) < half_width
    hist = SCENES[scene][1]
    assert hist.low < reference < hist.high and hist.peaks[0][0] == pytest.approx(reference)


def test_electron_positron_scan_traces_the_breit_wigner():
    # sqrt(s) uniform in M -+ 2 Gamma; a Z forms with probability
    # w = (Gamma/2)^2 / ((sqrt(s) - M)^2 + (Gamma/2)^2). Integrating w over the
    # range: fraction formed = (1 / 4 Gamma) (Gamma / 2) 2 atan(4) = atan(4) / 4
    # = 0.33146; of those, the fraction within Gamma/2 of the peak is
    # atan(1) / atan(4) = 0.59238. Each within 4 binomial sigma. A formed Z has
    # the pair's four-momentum: at rest, mass sqrt(s). An event that formed
    # nothing enters nothing in the histogram.
    scene = "e+ e- -> Z (collider scan)"
    rng = np.random.default_rng(403)
    n = 20000
    roots, formed = [], 0
    for _ in range(n):
        event = generate(scene, 3.8, rng, quick=True)
        record = event.formation
        root_s = record["sqrt_s"]
        assert abs(root_s - M_Z) <= 2.0 * G_Z + 1e-9
        assert event.beams == ("e-", "e+")
        assert event.beam_energy * 1000.0 == pytest.approx(root_s / 2.0, rel=1e-12)
        if record["formed"]:
            formed += 1
            (z,) = event.bosons
            assert float(mass(z["p4"])) == pytest.approx(root_s, abs=1e-9)
            np.testing.assert_allclose(z["p4"][1:], 0.0, atol=1e-9)
            assert event.masses == [("sqrt(s)", root_s)]
            assert histogram_masses(event) == [root_s]
            roots.append(root_s)
        else:
            assert not event.bosons and histogram_masses(event) == []
            assert "No Z was formed" in event.note
    p = np.arctan(4.0) / 4.0
    assert p == pytest.approx(0.33146, abs=1e-5)
    assert formed / n == pytest.approx(p, abs=4.0 * np.sqrt(p * (1.0 - p) / n))
    q = np.arctan(1.0) / np.arctan(4.0)
    assert q == pytest.approx(0.59238, abs=1e-5)
    near = np.mean(np.abs(np.array(roots) - M_Z) < G_Z / 2.0)
    assert near == pytest.approx(q, abs=4.0 * np.sqrt(q * (1.0 - q) / formed))
    # quick_masses: only the collisions that formed a Z (about a third).
    masses = quick_masses(scene, 3000, 3.8, rng)
    assert len(masses) / 3000 == pytest.approx(p, abs=4.0 * np.sqrt(p * (1.0 - p) / 3000))
    # Full events: the Z's decay products add up to it (four-momentum conserved).
    for _ in range(200):
        event = generate(scene, 3.8, rng)
        if event.formation["formed"]:
            leptons = [t.p4 for t in event.tracks if t.parent < 0]
            assert len(leptons) == 2
            np.testing.assert_allclose(sum(leptons), event.bosons[0]["p4"], rtol=1e-9, atol=1e-9)
        else:
            # The two fly on along the beam line, unbent, and leave.
            assert [t.name for t in event.tracks] == ["e-", "e+"]
            for t in event.tracks:
                assert t.end == "exit" and np.allclose(t.points[:, :2], 0.0)


@pytest.mark.parametrize("scene", list(SCENES))
def test_every_scene_generates(scene):
    # 20 events of every scene, as shown and as made for the histogram.
    if scene.startswith("Real data") and not particle_opendata.available(scene):
        pytest.skip("CMS open data not downloaded")
    rng = np.random.default_rng(404)
    for quick in (False, True):
        for _ in range(20):
            event = generate(scene, 3.8, rng, quick=quick)
            assert not event.error
            assert all(np.isfinite(m) for _, m in event.masses)
