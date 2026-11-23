"""Relativistic kinematics for the particle-physics module: four-momenta, boosts, decays.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
4). Review the physics.

Units: c = 1, energies, momenta and masses in GeV. A four-momentum is an
array (E, px, py, pz); arrays of them have that as their last axis.

- Invariant mass: m^2 = E^2 - |p|^2 (of one particle or of a sum).
- Boost: a particle with four-momentum p* in the rest frame of a parent
  moving with velocity beta = p_parent / E_parent has, in the lab,
      E = gamma (E* + beta . p*),
      p = p* + [(gamma - 1) (beta . p*) / beta^2 + gamma E*] beta.
- Two-body decay M -> m1 m2: in the parent's rest frame the daughters are
  back to back with momentum
      p* = sqrt((M^2 - (m1 + m2)^2) (M^2 - (m1 - m2)^2)) / (2 M)
  (e.g. PDG, Kinematics review, eq. 49.17).
- Decays to three or more bodies, phase space only (no matrix element):
  n-body phase space factorises into a chain of two-body decays,
  M -> (1..n-1) n, (1..n-1) -> (1..n-2) (n-1), ..., each isotropic in its
  own rest frame, with the masses M_i of the intermediate systems
  distributed with weight prod_i p*(M_i; M_{i-1}, m_i) (James 1968, CERN
  68-15, "GENBOD"; PDG eq. 49.20 for n = 3). The M_i are drawn uniformly
  (ordered) between their thresholds and accepted with that weight.
  Real decays are also weighted by their matrix elements. That is done only
  for the cases below (leptonic decays, and the tau's decays through the
  rho and the a1); every other decay to three or more bodies used by the
  module is plain phase space, which is an approximation.
- Leptonic decays l -> l' nubar_l' nu_l (muon and tau decay). The weak
  (V - A) interaction gives, summed over spins and with the daughters'
  masses neglected in the matrix element,
      |M|^2 proportional to (P . k1) (q . k2),
  P the parent, q the charged daughter, k1 the antineutrino paired with it
  and k2 the neutrino of the parent's flavour (e.g. Griffiths 2008,
  Introduction to Elementary Particles, sec. 9.2 and 10.2). Events are drawn
  from phase space and accepted with that weight (leptonic_decay). The
  charged lepton's energy then follows the Michel spectrum
  x^2 (3 - 2 x), x = 2 E / M, with mean 7 M / 20 (phase space alone would
  give M / 3).
- Hadronic tau decays through a resonance, tau -> (hadrons) nu. With s the
  squared mass of the hadrons, the weak current gives (Tsai 1971, Phys. Rev.
  D 4, 2821)
      dGamma / ds  proportional to  (1 - s / M^2)^2 (1 + 2 s / M^2) v(s),
  v(s) the spectral function of the hadronic state. Two models of v:
  * pi pi0 (the rho(770), tau_decay_via_rho): v = beta^3 |F|^2 with
    beta = 2 p* / sqrt(s) the pions' velocity in their rest frame and a
    Breit-Wigner form factor with the width of a decay to two pions in a
    P wave,
        |F|^2 = 1 / ((s - m0^2)^2 + m0^2 Gamma(s)^2),
        Gamma(s) = Gamma0 (m0 / sqrt(s)) (p*(s) / p*(m0^2))^3
    (e.g. Kuehn & Santamaria 1990, Z. Phys. C 48, 445, with the rho alone:
    no rho', no Gounaris-Sakurai correction).
  * three pions (the a1(1260) -> rho pi, tau_decay_via_a1): a Breit-Wigner
    of constant width for the a1 and one for the rho, times three-body
    phase space,
        density in (m_a1, m_rho)  proportional to
        (1 - s / M^2)^2 (1 + 2 s / M^2) BW_a1(s) BW_rho(m_rho^2) p*(m_a1; m_rho, m_pi) p*(m_rho; m_pi, m_pi),
        BW(q) = 1 / ((q - m0^2)^2 + m0^2 Gamma0^2), s = m_a1^2.
    This is a simple model: the a1's width is taken constant, and the
    interference between the two ways of pairing the like pions is left out.
  In both, every two-body step is isotropic in its rest frame: the angular
  correlations that the tau's and the resonances' spins give are left out.
"""

from functools import lru_cache

import numpy as np


def mass(p):
    """Invariant mass of four-momenta p (sign kept: negative for space-like)."""
    p = np.asarray(p, dtype=float)
    m2 = p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, axis=-1)
    return np.sign(m2) * np.sqrt(np.abs(m2))


def four_momentum(m, momentum):
    """(E, px, py, pz) of a particle of mass m with three-momentum `momentum`."""
    momentum = np.asarray(momentum, dtype=float)
    return np.array([np.sqrt(m * m + momentum @ momentum), *momentum])


def boost(p_rest, parent):
    """Four-momentum in the lab of a particle with `p_rest` in the rest frame of `parent`."""
    beta = parent[1:] / parent[0]
    b2 = beta @ beta
    if b2 == 0.0:
        return np.array(p_rest, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = beta @ p_rest[1:]
    energy = gamma * (p_rest[0] + bp)
    momentum = p_rest[1:] + ((gamma - 1.0) * bp / b2 + gamma * p_rest[0]) * beta
    return np.array([energy, *momentum])


def two_body_momentum(M, m1, m2):
    """Momentum of each daughter of M -> m1 m2 in the parent's rest frame."""
    if M < m1 + m2:
        raise ValueError(f"decay not allowed: {M} < {m1} + {m2}")
    return np.sqrt((M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)) / (2.0 * M)


def isotropic_direction(rng):
    cos_theta = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def two_body_decay(parent, m1, m2, rng, direction=None):
    """Four-momenta in the lab of the daughters of `parent` -> m1 m2.

    Isotropic in the parent's rest frame unless `direction` (a unit vector
    there, for daughter 1) is given.
    """
    M = float(mass(parent))
    p = two_body_momentum(M, m1, m2)
    n = isotropic_direction(rng) if direction is None else np.asarray(direction, dtype=float)
    first = four_momentum(m1, p * n)
    second = four_momentum(m2, -p * n)
    return boost(first, parent), boost(second, parent)


def phase_space_decay(parent, masses, rng):
    """Four-momenta in the lab of the daughters of `parent` -> masses[0], masses[1], ...
    distributed uniformly in phase space (see the module docstring)."""
    masses = [float(m) for m in masses]
    n = len(masses)
    if n < 2:
        raise ValueError("a decay needs at least two daughters")
    if n == 2:
        return list(two_body_decay(parent, masses[0], masses[1], rng))
    M = float(mass(parent))
    free = M - sum(masses)  # kinetic energy to share out
    if free < 0.0:
        raise ValueError(f"decay not allowed: {M} < {sum(masses)}")
    partial = np.cumsum(masses)  # partial[i] = m_0 + ... + m_i
    # Largest possible weight: every split at its largest momentum.
    bound = 1.0
    for i in range(1, n):
        bound *= two_body_momentum(partial[i] + free, partial[i - 1], masses[i])
    while True:
        # Masses of the systems of the first i + 1 daughters: thresholds plus
        # ordered random shares of the free energy (M for all n of them).
        shares = np.concatenate([[0.0], np.sort(rng.uniform(0.0, 1.0, n - 2)), [1.0]])
        systems = partial + shares * free
        weight = 1.0
        for i in range(1, n):
            weight *= two_body_momentum(systems[i], systems[i - 1], masses[i])
        if rng.uniform(0.0, bound) < weight:
            break
    # Peel the daughters off one at a time, last first.
    out = [None] * n
    system = np.array(parent, dtype=float)
    for i in range(n - 1, 0, -1):
        system, out[i] = two_body_decay(system, systems[i - 1], masses[i], rng)
    out[0] = system
    return out


def breit_wigner_mass(m0, width, rng, low=None, high=None):
    """A mass from the (non-relativistic) Breit-Wigner line shape of a resonance,
    density proportional to 1 / ((m - m0)^2 + width^2 / 4), within [low, high]
    (default m0 -+ 10 widths)."""
    low = m0 - 10.0 * width if low is None else low
    high = m0 + 10.0 * width if high is None else high
    if width <= 0.0:
        return m0
    # Inverse of the cumulative distribution, restricted to the range.
    a = np.arctan(2.0 * (low - m0) / width)
    b = np.arctan(2.0 * (high - m0) / width)
    return m0 + 0.5 * width * np.tan(rng.uniform(a, b))


def breit_wigner_formation(sqrt_s, m0, width):
    """Probability, relative to the peak, that a pair of invariant mass sqrt_s
    forms a resonance of mass m0 and width `width` (GeV):
        w(sqrt_s) = (width / 2)^2 / ((sqrt_s - m0)^2 + (width / 2)^2),
    the same non-relativistic Breit-Wigner as breit_wigner_mass, scaled to 1
    at sqrt_s = m0 (and 1/2 at m0 -+ width / 2: the full width at half
    maximum is the width). PDG 2024, review "Resonances" (Breit-Wigner
    line shape); the formation cross section of e+ e- -> Z follows this
    shape near the peak (PDG 2024, Z boson review), with the absolute scale
    left out here (see particle_custom.py, "Collisions")."""
    if width <= 0.0:
        return 1.0 if sqrt_s == m0 else 0.0
    half = 0.5 * width
    return half * half / ((sqrt_s - m0) ** 2 + half * half)


def dot(a, b):
    """Minkowski product a . b = E_a E_b - p_a . p_b."""
    return a[0] * b[0] - a[1:] @ b[1:]


def leptonic_decay(parent, m_lepton, rng):
    """Four-momenta in the lab of (charged lepton, antineutrino, neutrino) from the
    leptonic decay of `parent` (see the module docstring)."""
    M = float(mass(parent))
    # (P . k1) <= M^2 / 2 and (q . k2) <= M^2 / 2: the weight is at most M^4 / 4.
    bound = 0.25 * M**4
    while True:
        lepton, k1, k2 = phase_space_decay(parent, [m_lepton, 0.0, 0.0], rng)
        if rng.uniform(0.0, bound) < dot(parent, k1) * dot(lepton, k2):
            return [lepton, k1, k2]


def _momentum(M, m1, m2):
    """two_body_momentum for arrays, zero where the decay is not allowed."""
    value = (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)
    return np.sqrt(np.maximum(value, 0.0)) / (2.0 * np.maximum(M, 1e-300)) * (M > m1 + m2)


def _tau_factor(m, m_tau):
    """(1 - s / M^2)^2 (1 + 2 s / M^2) for hadrons of mass m in tau -> hadrons nu."""
    x = (m / m_tau) ** 2
    return (1.0 - x) ** 2 * (1.0 + 2.0 * x) * (x < 1.0)


def rho_mass_density(m, m_tau, m1, m2, m0, width):
    """Density (not normalised) of the pi pi mass m in tau -> pi pi nu (module docstring)."""
    p = _momentum(m, m1, m2)
    running = width * (m0 / np.maximum(m, 1e-300)) * (p / two_body_momentum(m0, m1, m2)) ** 3
    form = 1.0 / ((m * m - m0 * m0) ** 2 + (m0 * running) ** 2)
    beta = 2.0 * p / np.maximum(m, 1e-300)
    return 2.0 * m * _tau_factor(m, m_tau) * beta**3 * form


def a1_mass_density(m_a, m_r, m_tau, m1, m2, m3, rho, a1):
    """Density (not normalised) of the three-pion mass m_a and the mass m_r of the
    pair (1, 2) in tau -> a1 nu, a1 -> rho pi, rho -> pi pi (module docstring)."""
    bw_a = 1.0 / ((m_a * m_a - a1[0] ** 2) ** 2 + (a1[0] * a1[1]) ** 2)
    bw_r = 1.0 / ((m_r * m_r - rho[0] ** 2) ** 2 + (rho[0] * rho[1]) ** 2)
    return (
        _tau_factor(m_a, m_tau) * bw_a * bw_r * _momentum(m_a, m_r, m3) * _momentum(m_r, m1, m2)
    )


@lru_cache(maxsize=32)
def _rho_bound(m_tau, m1, m2, m0, width):
    grid = np.linspace(m1 + m2, m_tau, 4001)[1:-1]
    return 1.05 * float(rho_mass_density(grid, m_tau, m1, m2, m0, width).max())


@lru_cache(maxsize=32)
def _a1_bound(m_tau, m1, m2, m3, rho, a1):
    m_a = np.linspace(m1 + m2 + m3, m_tau, 501)[1:-1, None]
    m_r = np.linspace(m1 + m2, m_tau - m3, 501)[None, 1:-1]
    return 1.2 * float(a1_mass_density(m_a, m_r, m_tau, m1, m2, m3, rho, a1).max())


def tau_decay_via_rho(parent, m1, m2, rho, rng):
    """Four-momenta in the lab of (pion 1, pion 2, neutrino) from tau -> rho nu,
    rho -> pi pi, with the rho's line shape (see the module docstring).
    `rho` is (mass, width)."""
    m_tau = float(mass(parent))
    bound = _rho_bound(m_tau, m1, m2, *rho)
    while True:
        m = rng.uniform(m1 + m2, m_tau, 64)
        ok = np.flatnonzero(
            rng.uniform(0.0, bound, 64) < rho_mass_density(m, m_tau, m1, m2, *rho)
        )
        if len(ok):
            break
    pair, neutrino = two_body_decay(parent, float(m[ok[0]]), 0.0, rng)
    first, second = two_body_decay(pair, m1, m2, rng)
    return [first, second, neutrino]


def tau_decay_via_a1(parent, m1, m2, m3, rho, a1, rng):
    """Four-momenta in the lab of (pion 1, pion 2, pion 3, neutrino) from
    tau -> a1 nu, a1 -> rho pion 3, rho -> pion 1 pion 2 (see the module
    docstring). `rho` and `a1` are (mass, width)."""
    m_tau = float(mass(parent))
    bound = _a1_bound(m_tau, m1, m2, m3, tuple(rho), tuple(a1))
    while True:
        m_a = rng.uniform(m1 + m2 + m3, m_tau, 256)
        m_r = rng.uniform(m1 + m2, m_tau - m3, 256)
        density = a1_mass_density(m_a, m_r, m_tau, m1, m2, m3, rho, a1)
        ok = np.flatnonzero(rng.uniform(0.0, bound, 256) < density)
        if len(ok):
            break
    hadrons, neutrino = two_body_decay(parent, float(m_a[ok[0]]), 0.0, rng)
    pair, third = two_body_decay(hadrons, float(m_r[ok[0]]), m3, rng)
    first, second = two_body_decay(pair, m1, m2, rng)
    return [first, second, third, neutrino]


# --- Semileptonic decays of a pseudoscalar meson to a pseudoscalar, P -> P' l nu
# (B -> D l nu, D -> K l nu, D -> pi l nu, Ds -> K l nu, Bs -> Ds l nu).
#
# Matrix element. The hadronic current is
#     <P'(p')| V^mu |P(p)> = f+(q^2) (p + p')^mu + f-(q^2) q^mu,   q = p - p',
#     f-(q^2) = (f0(q^2) - f+(q^2)) (M^2 - m^2) / q^2,
# with f+ the vector and f0 the scalar form factor (f0(0) = f+(0)). The
# lepton current l-bar gamma_mu (1 - gamma5) nu, summed over spins, gives the
# tensor 8 [k^mu l^nu + k^nu l^mu - g^munu (k . l) -+ i eps^munu.. k l], k the
# neutrino and l the charged lepton; the eps term drops out against the
# symmetric H^mu H^nu (real form factors), so for l+ and for l- alike
#     |M|^2 proportional to 2 (H . k)(H . l) - H^2 (k . l),   H = f+ (p + p') + f- q.
# For a massless lepton the f- terms cancel exactly. Integrated over the
# lepton angle this gives the standard
#     dGamma/dq^2 = G_F^2 |V|^2 / (24 pi^3) (1 - m_l^2/q^2)^2 |p'| [ (1 + m_l^2 / (2 q^2)) |p'|^2 |f+|^2
#                   + 3 m_l^2 / (8 q^2) (M^2 - m^2)^2 / M^2 |f0|^2 ]
# (e.g. Korner & Schuler 1990, Z. Phys. C 46, 93), |p'| the daughter meson's
# momentum in the parent's rest frame; tests check the sampled spectrum
# against it. For m_l = 0 the lepton's angle theta in the l nu rest frame
# (from the direction of the mesons there) follows sin^2(theta): the W is
# longitudinal (helicity 0) in a 0- -> 0- transition.
#
# Form-factor models (`model` tuples, only the shape matters: the absolute
# rate is set by the branching fraction):
# - ("CLN", rho2): heavy-to-heavy (B -> D, Bs -> Ds). Caprini, Lellouch &
#   Neubert 1998, Nucl. Phys. B 530, 153 (hep-ph/9712417), eq. (28):
#       V1(w) / V1(1) = 1 - 8 rho2 z + (51 rho2 - 10) z^2 - (252 rho2 - 84) z^3,
#       z = (sqrt(w + 1) - sqrt(2)) / (sqrt(w + 1) + sqrt(2)),  w = (M^2 + m^2 - q^2) / (2 M m),
#   and their Table A.1 (w0 = 1) for the scalar form factor
#       S1(w) / V1(w) = 1.0036 [1 - 0.0068 (w - 1) + 0.0017 (w - 1)^2 - 0.0013 (w - 1)^3].
#   With r = m / M: f+ = (1 + r) / (2 sqrt(r)) V1, f0 = sqrt(r) (w + 1) / (1 + r) S1
#   (from their definitions (A.1), (A.3) of h+-, V1, S1).
# - ("z2", r1, m_pole): the two-parameter series expansion (BGL-type z
#   expansion) as fitted by BESIII 2015, Phys. Rev. D 92, 072012
#   (arXiv:1508.07560), eqs. (II.6)-(II.9), (II.12):
#       f+(t) proportional to (1 + r1 z(t, t0)) / (P(t) Phi(t, t0)),
#       z(t, t0) = (sqrt(t+ - t) - sqrt(t+ - t0)) / (sqrt(t+ - t) + sqrt(t+ - t0)),
#       t+- = (M +- m)^2, t0 = t+ (1 - sqrt(1 - t- / t+)),
#       P(t) = z(t, m_pole^2) (D -> K, the Ds* below threshold), P = 1 (D -> pi),
#       Phi(t, t0) proportional to ((t+ - t) / (t+ - t0))^(1/4) (sqrt(t+ - t) + sqrt(t+))^-5
#                  (sqrt(t+ - t) + sqrt(t+ - t0)) (sqrt(t+ - t) + sqrt(t+ - t-))^(3/2) (t+ - t)^(3/4)
#   (the constant 1 / sqrt(24 pi chi_V) dropped). BESIII fits f+ only: no
#   f0 is sourced for these decays, so f0 = f+ is taken (f- = 0). That only
#   matters for the muon modes (terms of order m_mu^2 / q^2): an
#   assumption, not a measurement.
# - ("pole", m_pole): single pole f+ = 1 / (1 - q^2 / m_pole^2) with the
#   nominal vector-meson mass (Koerner-Schuler / BSW form, BESIII 2015
#   eq. (II.3)); f0 = f+ as above. A model, with no fitted parameter.
def _z_map(t, t_plus, t0):
    a, b = np.sqrt(np.maximum(t_plus - t, 0.0)), np.sqrt(t_plus - t0)
    return (a - b) / (a + b)


def semileptonic_form_factors(model, M, m, q2):
    """(f+, f0) at q2 (arrays) for P (mass M) -> P' (mass m) l nu; arbitrary common scale."""
    q2 = np.asarray(q2, dtype=float)
    kind = model[0]
    if kind == "CLN":
        rho2 = model[1]
        w = (M * M + m * m - q2) / (2.0 * M * m)
        s = np.sqrt(w + 1.0)
        z = (s - np.sqrt(2.0)) / (s + np.sqrt(2.0))
        v1 = 1.0 - 8.0 * rho2 * z + (51.0 * rho2 - 10.0) * z**2 - (252.0 * rho2 - 84.0) * z**3
        x = w - 1.0
        s1 = 1.0036 * (1.0 - 0.0068 * x + 0.0017 * x**2 - 0.0013 * x**3) * v1
        r = m / M
        return (1.0 + r) / (2.0 * np.sqrt(r)) * v1, np.sqrt(r) * (w + 1.0) / (1.0 + r) * s1
    if kind == "z2":
        r1, m_pole = model[1], model[2]
        t_plus, t_minus = (M + m) ** 2, (M - m) ** 2
        t0 = t_plus * (1.0 - np.sqrt(1.0 - t_minus / t_plus))
        a = np.sqrt(np.maximum(t_plus - q2, 0.0))
        phi = (
            ((t_plus - q2) / (t_plus - t0)) ** 0.25
            * (a + np.sqrt(t_plus)) ** -5
            * (a + np.sqrt(t_plus - t0))
            * (a + np.sqrt(t_plus - t_minus)) ** 1.5
            * (t_plus - q2) ** 0.75
        )
        pole = _z_map(q2, t_plus, m_pole**2) if m_pole is not None else 1.0
        f = (1.0 + r1 * _z_map(q2, t_plus, t0)) / (pole * phi)
        return f, f
    if kind == "pole":
        f = 1.0 / (1.0 - q2 / model[1] ** 2)
        return f, f
    raise ValueError(f"unknown form-factor model {model!r}")


def semileptonic_density(model, M, m, m_lepton, q2, cos_theta):
    """Density (not normalised) of (q^2, cos theta) in P -> P' l nu: |M|^2 times
    three-body phase space |p'| (1 - m_l^2 / q^2). theta: the charged lepton's
    angle, in the l nu rest frame, to the direction of the mesons there."""
    if model[0] == "CLN-V":
        return vector_semileptonic_density(model, M, m, m_lepton, q2, cos_theta)
    q2, cos_theta = np.broadcast_arrays(np.asarray(q2, float), np.asarray(cos_theta, float))
    ok = (q2 > m_lepton**2) & (np.sqrt(np.maximum(q2, 0.0)) < M - m)
    q2 = np.where(ok, q2, 0.5 * ((M - m) ** 2 + m_lepton**2))
    root = np.sqrt(q2)
    f_plus, f_zero = semileptonic_form_factors(model, M, m, q2)
    f_minus = (f_zero - f_plus) * (M * M - m * m) / q2
    # Four-vectors in the l nu rest frame, the mesons along +z.
    e_parent = (M * M + q2 - m * m) / (2.0 * root)
    e_meson = (M * M - q2 - m * m) / (2.0 * root)
    p_z = np.sqrt(np.maximum(e_meson**2 - m * m, 0.0))  # same for parent and meson
    h0 = f_plus * (e_parent + e_meson) + f_minus * root
    hz = 2.0 * f_plus * p_z
    e_l = (q2 + m_lepton**2) / (2.0 * root)
    p_l = (q2 - m_lepton**2) / (2.0 * root)
    # lepton l = (e_l, p_l sin, 0, p_l cos); neutrino k = (p_l, -p_l sin, 0, -p_l cos)
    h_l = h0 * e_l - hz * p_l * cos_theta
    h_k = h0 * p_l + hz * p_l * cos_theta
    k_l = p_l * e_l + p_l * p_l
    weight = 2.0 * h_k * h_l - (h0 * h0 - hz * hz) * k_l
    phase = _momentum(M, root, m) * (1.0 - m_lepton**2 / q2)
    return np.where(ok, np.maximum(weight, 0.0) * phase, 0.0)


# --- Semileptonic decays of a pseudoscalar to a vector meson, P -> V l nu
# (B -> D* l nu). Added 2026-11-23.
#
# Form factors: ("CLN-V", rho2, R1(1), R2(1)), Caprini, Lellouch & Neubert
# 1998 (above), eq. (38) for h_A1 and eq. (36) for the ratios:
#     h_A1(w) / h_A1(1) = 1 - 8 rho2 z + (53 rho2 - 15) z^2 - (231 rho2 - 91) z^3,
#     R1(w) = R1(1) - 0.12 (w - 1) + 0.05 (w - 1)^2,   (CLN: R1(1) = 1.27)
#     R2(w) = R2(1) + 0.11 (w - 1) - 0.06 (w - 1)^2,   (CLN: R2(1) = 0.80)
# z as for B -> D. With r' = 2 sqrt(M m) / (M + m) the usual form factors are
#     A1 = r' (w + 1) / 2 h_A1,  V = R1 h_A1 / r',  A2 = R2 h_A1 / r',  A0 = R0 h_A1 / r'
# (from CLN's (A.2) and the definitions of R1 = h_V / h_A1, R2 = (h_A3 + r h_A2) / h_A1).
# A0 only enters with the lepton mass. CLN give no R0: the heavy-quark limit
# (h_A1 = h_A3, h_A2 = 0, CLN appendix) gives R0 = 1, which is taken here.
# That is an approximation, fine for e and mu (terms of order m_l^2 / q^2)
# and not for the tau, so no D* tau nu mode is listed.
#
# Helicity amplitudes of the W* (Korner & Schuler 1990, Z. Phys. C 46, 93),
# p the D*'s momentum in the B frame, for b -> c (H_- the larger):
#     H_+- = (M + m) A1 -+ 2 M p V / (M + m),
#     H_0  = [(M^2 - m^2 - q^2)(M + m) A1 - 4 M^2 p^2 A2 / (M + m)] / (2 m sqrt(q^2)),
#     H_t  = 2 M p A0 / sqrt(q^2),
# summed over the D*'s helicities (its decay is taken isotropic):
#     dGamma / dq^2 dcos(theta_W)  proportional to  p q^2 (1 - m_l^2/q^2)^2 x
#         [ (1 + c)^2 H_-^2 + (1 - c)^2 H_+^2 + 2 (1 - c^2) H_0^2
#           + m_l^2/q^2 ((1 - c^2)(H_+^2 + H_-^2) + 2 (H_0 c - H_t)^2) ],
# c = cos(theta_W), theta_W the charged lepton's angle in the l nu frame to the
# W's direction in the B frame. A lepton (helicity -1/2 for l-, +1/2 for l+)
# and antineutrino in the angular-momentum state -1 (+1) along the lepton
# give d^1_{lambda,-1}: H_- (1 + c) for b -> c l- nu-bar; the charge
# conjugate (B+, B0: l+) is the mirror image, the same distribution with
# H_+ <-> H_- relabelled, so one formula serves both. The sign of the
# H_0 H_t interference is convention-bound and only matters at order m_l^2 / q^2.
# Integrated over c:
#     dGamma / dq^2 = G_F^2 |V_cb|^2 p q^2 / (96 pi^3 M^2) (1 - m_l^2/q^2)^2
#                     [ (H_+^2 + H_-^2 + H_0^2)(1 + m_l^2/(2 q^2)) + 3 m_l^2/(2 q^2) H_t^2 ],
# with the form factors normalised to h_A1(1) = 1 and F(1)|V_cb| in front.
def vector_helicity_amplitudes(model, M, m, q2):
    """(H_+, H_-, H_0, H_t) at q2 (arrays) for P (mass M) -> V (mass m) l nu, the
    form factors in units of h_A1(1) (see the comments above)."""
    q2 = np.asarray(q2, dtype=float)
    rho2, r1_one, r2_one = model[1], model[2], model[3]
    w = (M * M + m * m - q2) / (2.0 * M * m)
    s = np.sqrt(w + 1.0)
    z = (s - np.sqrt(2.0)) / (s + np.sqrt(2.0))
    h_a1 = 1.0 - 8.0 * rho2 * z + (53.0 * rho2 - 15.0) * z**2 - (231.0 * rho2 - 91.0) * z**3
    x = w - 1.0
    r1 = r1_one - 0.12 * x + 0.05 * x**2
    r2 = r2_one + 0.11 * x - 0.06 * x**2
    rp = 2.0 * np.sqrt(M * m) / (M + m)
    a1 = rp * (w + 1.0) / 2.0 * h_a1
    v, a2, a0 = r1 * h_a1 / rp, r2 * h_a1 / rp, h_a1 / rp  # R0 = 1 (heavy-quark limit)
    p = _momentum(M, np.sqrt(np.maximum(q2, 0.0)), m)
    root = np.sqrt(np.maximum(q2, 1e-300))
    h_plus = (M + m) * a1 - 2.0 * M * p * v / (M + m)
    h_minus = (M + m) * a1 + 2.0 * M * p * v / (M + m)
    h_zero = ((M * M - m * m - q2) * (M + m) * a1 - 4.0 * M * M * p * p * a2 / (M + m)) / (
        2.0 * m * root
    )
    h_time = 2.0 * M * p * a0 / root
    return h_plus, h_minus, h_zero, h_time


def vector_semileptonic_density(model, M, m, m_lepton, q2, cos_theta):
    """Density (not normalised) of (q^2, cos theta) in P -> V l nu, theta as in
    semileptonic_density (to the mesons' direction, so cos theta_W = -cos theta)."""
    q2, cos_theta = np.broadcast_arrays(np.asarray(q2, float), np.asarray(cos_theta, float))
    ok = (q2 > m_lepton**2) & (np.sqrt(np.maximum(q2, 0.0)) < M - m)
    q2 = np.where(ok, q2, 0.5 * ((M - m) ** 2 + m_lepton**2))
    hp, hm, h0, ht = vector_helicity_amplitudes(model, M, m, q2)
    c = -cos_theta
    eps = m_lepton**2 / q2
    angular = (
        (1.0 + c) ** 2 * hm**2
        + (1.0 - c) ** 2 * hp**2
        + 2.0 * (1.0 - c * c) * h0**2
        + eps * ((1.0 - c * c) * (hp**2 + hm**2) + 2.0 * (h0 * c - ht) ** 2)
    )
    weight = _momentum(M, np.sqrt(q2), m) * q2 * (1.0 - eps) ** 2 * angular
    return np.where(ok, weight, 0.0)


@lru_cache(maxsize=64)
def _semileptonic_bound(model, M, m, m_lepton):
    q2 = np.linspace(m_lepton**2, (M - m) ** 2, 1202)[1:-1, None]
    cos_theta = np.linspace(-1.0, 1.0, 81)[None, :]
    return 1.1 * float(semileptonic_density(model, M, m, m_lepton, q2, cos_theta).max())


def semileptonic_decay(parent, m_meson, m_lepton, model, rng):
    """Four-momenta in the lab of (meson, charged lepton, neutrino) from the
    semileptonic decay P -> P' l nu of `parent` with the form-factor `model`
    (see the comments above)."""
    M = float(mass(parent))
    # A D* daughter has a mass drawn from its line shape: the bound (10% above
    # the maximum) is taken at its mass rounded to 0.1 MeV, so that it is cached.
    m_bound = round(m_meson, 4) if model[0] == "CLN-V" else m_meson
    bound = _semileptonic_bound(model, M, m_bound, m_lepton)
    low, high = m_lepton**2, (M - m_meson) ** 2
    while True:
        q2 = rng.uniform(low, high, 256)
        cos_theta = rng.uniform(-1.0, 1.0, 256)
        density = semileptonic_density(model, M, m_meson, m_lepton, q2, cos_theta)
        ok = np.flatnonzero(rng.uniform(0.0, bound, 256) < density)
        if len(ok):
            break
    q2, cos_theta = float(q2[ok[0]]), float(cos_theta[ok[0]])
    # The l nu system and the meson, back to back and isotropic in the parent's frame.
    w, meson = two_body_decay(parent, np.sqrt(q2), m_meson, rng)
    # The mesons' direction in the l nu rest frame (that of the parent there).
    axis = boost(np.asarray(parent, dtype=float), np.array([w[0], *(-w[1:])]))[1:]
    direction = _direction_about(axis, cos_theta, rng)
    p = two_body_momentum(np.sqrt(q2), m_lepton, 0.0)
    lepton = boost(four_momentum(m_lepton, p * direction), w)
    neutrino = boost(np.array([p, *(-p * direction)]), w)
    return [meson, lepton, neutrino]


def _direction_about(axis, cos_theta, rng):
    """Unit vector at polar angle theta to `axis`, uniform in azimuth about it."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    return cos_theta * axis + sin_theta * (np.cos(phi) * u + np.sin(phi) * v)


# --- A longitudinally polarised vector meson to a lepton pair (B -> J/psi K,
# J/psi -> l+ l-). In a 0- -> 1- 0- decay angular momentum leaves the vector
# meson only helicity 0 along its flight direction in the parent's frame
# (both spins along that line must add to the parent's zero; the spin-0
# kaon has none). Its decay to l+ l- through the vector (electromagnetic)
# current has helicity amplitudes M d^1_{0,+-1}(theta) for opposite lepton
# helicities and 2 m_l d^1_{0,0}(theta) for equal ones, so
#     dGamma / dcos(theta)  proportional to  sin^2(theta) + (4 m_l^2 / M^2) cos^2(theta),
# theta the angle of l+ in the vector meson's rest frame to its flight
# direction (the helicity axis); for m_l -> 0 this is sin^2(theta). (E.g.
# the helicity formalism of Jacob & Wick 1959, Ann. Phys. 7, 404; the
# transverse case gives 1 + cos^2(theta).) Integrated: (4/3)(1 + 2 m_l^2 / M^2),
# the usual factor of a vector's decay to a lepton pair.
def longitudinal_vector_decay(parent, m_lepton, grandparent, rng):
    """Four-momenta in the lab of (l+, l-) from a vector meson `parent` of helicity 0
    along its direction in the rest frame of `grandparent` (see the comments above)."""
    M = float(mass(parent))
    eps = 4.0 * m_lepton * m_lepton / (M * M)
    while True:
        c = rng.uniform(-1.0, 1.0)
        if rng.uniform(0.0, max(1.0, eps)) < (1.0 - c * c) + eps * c * c:
            break
    # Helicity axis in the parent's rest frame: opposite to the grandparent's motion there.
    inverse = np.array([parent[0], *(-np.asarray(parent[1:], dtype=float))])
    axis = -boost(np.asarray(grandparent, dtype=float), inverse)[1:]
    return list(two_body_decay(parent, m_lepton, m_lepton, rng, direction=_direction_about(axis, c, rng)))


# --- A three-body decay through a vector resonance (isobar model), P -> V P3,
# V -> P1 P2, every P of spin 0 (D -> K* pi, D -> rho K, Ds -> phi pi). As in
# longitudinal_vector_decay, angular momentum leaves the vector only helicity
# 0 along its flight direction in the parent's frame. Its decay to two spin-0
# particles then has the amplitude d^1_{0,0}(theta) = cos(theta) (Jacob & Wick
# 1959, Ann. Phys. 7, 404; d^1_{00} = cos(theta) in the PDG's table of d
# functions, review "Clebsch-Gordan coefficients, spherical harmonics, and d
# functions"), so
#     dGamma / dcos(theta)  proportional to  cos^2(theta),
# theta the helicity angle: that of daughter 1 in the vector's rest frame to
# the vector's flight direction. (sin^2(theta) is the decay of a helicity-0
# vector to a fermion pair, as in longitudinal_vector_decay; a helicity-0
# vector cannot give pseudoscalars at 90 degrees.) cos^2 is drawn by inverting
# its cumulative distribution (1 + c^3) / 2: c = cbrt(2 u - 1).
# The vector's mass is drawn from breit_wigner_mass (constant width), cut to
# the range the decay allows (and to 10 widths of its mass). Stated
# approximations: no mass-dependent width, no Blatt-Weisskopf barrier
# factors, and the mass is not weighted by the two-body momenta of the two
# steps (phase space over the resonance's mass), which matters only in the
# tails of the wide K* and rho.
def vector_isobar_decay(parent, m1, m2, m3, resonance, rng):
    """Four-momenta in the lab of (1, 2, 3) from `parent` -> V 3, V -> 1 2, with V a
    vector resonance (mass, width) = `resonance` and 1, 2, 3 of spin 0 (see the
    comments above). Also returns the mass V was given."""
    M = float(mass(parent))
    m0, width = resonance
    low, high = max(m1 + m2, m0 - 10.0 * width), min(M - m3, m0 + 10.0 * width)
    if low >= high:
        raise ValueError(f"resonance {m0} GeV out of reach in a decay of {M} GeV")
    m_v = float(breit_wigner_mass(m0, width, rng, low, high))
    vector, spectator = two_body_decay(parent, m_v, m3, rng)
    c = float(np.cbrt(rng.uniform(-1.0, 1.0)))
    # Helicity axis in the vector's rest frame: opposite to the parent's motion there.
    inverse = np.array([vector[0], *(-vector[1:])])
    axis = -boost(np.asarray(parent, dtype=float), inverse)[1:]
    first, second = two_body_decay(vector, m1, m2, rng, direction=_direction_about(axis, c, rng))
    return [first, second, spectator], m_v
