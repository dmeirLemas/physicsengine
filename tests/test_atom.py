"""Tests of the atom module's physics (orbitals, nuclear data, decays, reactions).

Units: MeV, fm, seconds for nuclei; eV, nm and Bohr radii for electrons.

Reference values and their sources:

- Constants: CODATA 2018 (Tiesinga et al., Rev. Mod. Phys. 93, 025010, 2021).
- Hydrogen-like wave functions, <r> = a0 (3 n^2 - l (l + 1)) / (2 Z), the
  number of radial nodes n - l - 1: Bransden and Joachain, "Physics of Atoms
  and Molecules", 2nd ed., section 3.2 and table 3.2 (or Griffiths, ch. 4).
- Lines: Lyman-alpha 121.567 nm and H-alpha 656.461 nm (vacuum, fine-structure
  weighted), H-alpha 656.28 nm in air: NIST Atomic Spectra Database. Bohr
  theory with the proton's reduced mass gives 121.5684 nm and 656.4696 nm
  (R_H = R_inf / (1 + m_e / m_p) = 10967758.34 m^-1); the remaining 0.0014
  nm and 0.009 nm are fine structure and the Lamb shift, not modelled.
- Slater's rules (Slater, Phys. Rev. 36, 57, 1930): the textbook examples
  N 2p Z_eff = 3.90, Zn 3d 8.85, Zn 4s 4.35 (e.g. Housecroft and Sharpe,
  "Inorganic Chemistry", box 1.6).
- Ground-state configurations: NIST ASD (Cr [Ar] 3d5 4s1, Cu [Ar] 3d10 4s1,
  Pd [Kr] 4d10, Gd [Xe] 4f7 5d1 6s2).
- Masses: AME2020 (Wang et al., Chin. Phys. C 45, 030003). Binding energies
  He-4 28.2957 MeV, Fe-56 492.260 MeV, U-235 1783.870 MeV (AME2020 B/A times
  A). Q-values: U-238 alpha 4269.9 keV, C-14 beta- 156.476 keV, n 782.347
  keV, H-3 18.591 keV, K-40 beta- 1310.9 and EC 1504.4 keV, C-11 EC 1982.0
  keV (so positrons up to 960.0 keV), He-5 -> n 735 keV, Li-5 -> p 1965 keV.
- Half-lives: NUBASE2020 (Kondev et al., Chin. Phys. C 45, 030001): C-14
  5.70 ky, U-238 4.463 Gy. (4.468 Gy, often quoted, is the pre-2020
  evaluation; NUBASE2020 lists 4.463 Gy 3.)
- D + T -> He-4 + n: Q = 17.589 MeV, the neutron takes 14.05 MeV and the
  alpha 3.54 MeV (two-body kinematics; e.g. Krane, "Introductory Nuclear
  Physics", ch. 14). pp chain I: 26.73 MeV in all (Bahcall).
- U-238 alpha line: 4198 keV (ENSDF, the ground-state branch, 79%).
- Mean beta energies (ENSDF / LNHB tables): C-14 49.47 keV, P-32 694.9 keV
  (both allowed-shaped, endpoints 156.476 and 1710.66 keV).
- Gamow energy of p + p: E_G = 2 mu c^2 (pi alpha)^2 = 493 keV (Clayton,
  "Principles of Stellar Evolution and Nucleosynthesis", ch. 4); the Gamow
  peak at 15 MK (kT = 1.2926 keV) is at 5.9 keV.
- Fission of U-235 by thermal neutrons: prompt nu_bar 2.41 (total 2.43 with
  delayed neutrons), energy release about 202.5 MeV in all including the
  fragments' beta decays and the antineutrinos (James 1971; Madland, Nucl.
  Phys. A 772, 113, 2006), pre-neutron total fragment kinetic energy 170.5
  MeV, mean prompt-neutron energy 3a/2 + a^2 b / 4 = 2.03 MeV for the Watt
  parameters a = 0.988 MeV, b = 2.249 /MeV (ENDF/B / MCNP manual).
- First-flight escape probability from a uniform isotropic source in a
  sphere of radius R mean free paths: P = 3 / (8 R^3) (2 R^2 - 1 + (1 + 2R)
  exp(-2R)) (Case, de Hoffmann, Placzek 1953); P(1) = 0.5277.
"""

import math

import numpy as np
import pytest

from physengine.modules import atom_data as D
from physengine.modules import atom_decay as K
from physengine.modules import atom_nucleus as N
from physengine.modules import atom_orbitals as O
from physengine.modules import atom_reactions as R

YEAR = D.YEAR


@pytest.fixture(scope="module")
def table():
    """The files when present, otherwise the built-in excerpt (copied from them)."""
    return D.Table()


@pytest.fixture(scope="module")
def files():
    if not D.available():
        pytest.skip("AME2020 / NUBASE2020 files not in data/atom/")
    t = D.Table(use_files=True)
    assert t.from_files, t.error
    return t


def integrate(y, x):
    return float(np.trapezoid(y, x))


# ======================================================================= constants


def test_derived_constants_are_consistent():
    # a0 = hbar c / (alpha m_e c^2), Ry = alpha^2 m_e c^2 / 2, e^2 = alpha hbar c (CODATA 2018).
    a0_fm = D.HBARC / (D.ALPHA * D.M_ELECTRON)
    assert a0_fm == pytest.approx(D.BOHR_RADIUS_FM, rel=1e-9)
    assert 0.5 * D.ALPHA**2 * D.M_ELECTRON * 1e6 == pytest.approx(D.RYDBERG_EV, rel=1e-9)
    assert pytest.approx(1.439964548, rel=1e-9) == D.E2
    # Mass excesses: neutron and H-1 from the masses in MeV (CODATA m_n, m_p + m_e).
    assert D.M_NEUTRON - D.U_MEV == pytest.approx(D.EXCESS_N, abs=2e-6)
    assert D.M_PROTON + D.M_ELECTRON - 13.6e-6 - D.U_MEV == pytest.approx(D.EXCESS_H1, abs=2e-6)


# ======================================================================= orbitals


@pytest.mark.parametrize("Z", [1.0, 3.0])
def test_radial_functions_are_orthonormal(Z):
    # Integral of R_nl R_n'l r^2 dr = delta_nn' (exact for the hydrogen-like solutions).
    r = O.radial_grid(6, Z, points=40001)
    for l in range(0, 5):
        ns = range(l + 1, 7)
        for n in ns:
            for m in ns:
                value = integrate(O.radial(n, l, r, Z) * O.radial(m, l, r, Z) * r * r, r)
                assert value == pytest.approx(1.0 if n == m else 0.0, abs=1e-7), (n, m, l)


def _sphere_grid(points=64):
    x, w = np.polynomial.legendre.leggauss(points)  # exact in cos theta for these degrees
    phi = np.linspace(0.0, 2.0 * np.pi, 2 * points, endpoint=False)
    c, p = np.meshgrid(x, phi, indexing="ij")
    weight = np.outer(w, np.full(len(phi), 2.0 * np.pi / len(phi)))
    return c, p, weight


@pytest.mark.parametrize("real", [False, True])
def test_spherical_harmonics_are_orthonormal(real):
    # Integral over the sphere of conj(Y_lm) Y_l'm' = delta_ll' delta_mm'.
    c, p, w = _sphere_grid()
    f = O.real_harmonic if real else O.spherical_harmonic
    states = [(l, m) for l in range(4) for m in range(-l, l + 1)]
    values = [f(l, m, c, p) for l, m in states]
    gram = np.array([[np.sum(np.conj(a) * b * w) for b in values] for a in values])
    assert np.allclose(gram, np.eye(len(states)), atol=1e-12)


def test_real_harmonics_point_the_named_way():
    # p_x is largest along +x, p_y along +y, d_xy positive in the first quadrant.
    assert O.real_harmonic(1, 1, 0.0, 0.0) > 0.0
    assert O.real_harmonic(1, -1, 0.0, 0.5 * np.pi) > 0.0
    assert O.real_harmonic(2, -2, 0.0, 0.25 * np.pi) > 0.0
    # |Y_10|^2 = 3 cos^2 / 4 pi.
    assert abs(O.spherical_harmonic(1, 0, 1.0, 0.0)) ** 2 == pytest.approx(3.0 / (4.0 * np.pi))


@pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 2), (5, 1), (6, 4)])
def test_mean_radius_and_nodes(n, l):
    Z = 2.0
    r = O.radial_grid(n, Z, points=40001)
    density = (O.radial(n, l, r, Z) * r) ** 2
    expected = (3.0 * n * n - l * (l + 1)) / (2.0 * Z)  # Bransden and Joachain
    assert integrate(density * r, r) == pytest.approx(expected, rel=1e-6)
    assert O.mean_radius(n, l, Z) == pytest.approx(expected)
    # Radial nodes: n - l - 1 sign changes of R_nl for r > 0.
    values = O.radial(n, l, r[1:], Z)
    values = values[np.abs(values) > 1e-12 * np.abs(values).max()]
    assert int(np.sum(np.diff(np.sign(values)) != 0)) == n - l - 1


def test_sampled_cloud_statistics():
    rng = np.random.default_rng(2)
    count = 60000
    # 3d (m = 1, complex): <r> = (27 - 6) / 2 = 10.5 a0; Var r = <r^2> - <r>^2 with
    # <r^2> = n^2 (5 n^2 + 1 - 3 l (l + 1)) / 2 = 9 (45 + 1 - 18) / 2 = 126.
    xyz, phase = O.sample_orbital(3, 2, 1, count=count, rng=rng)
    r = np.linalg.norm(xyz, axis=1)
    sigma = math.sqrt(126.0 - 10.5**2)
    assert abs(r.mean() - 10.5) < 4.0 * sigma / math.sqrt(count)
    # |psi|^2 does not depend on phi: <x> = <y> = 0; the phase is m phi, plus pi
    # where the real factor R_32 Theta_21 is negative (Theta_21 ~ -sin cos, Condon-Shortley).
    assert abs(xyz[:, 0].mean()) < 4.0 * r.std() / math.sqrt(count)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    sign = np.sign(O.theta_part(2, 1, xyz[:, 2] / r) * O.radial(3, 2, r))
    assert np.allclose(np.exp(1j * phase), sign * np.exp(1j * phi), atol=1e-9)
    # 2p_z: <cos^2 theta> = 3/5 (variance of cos^2 under 3 c^2 / 2 is 12/175).
    xyz, _ = O.sample_orbital(2, 1, 0, count=count, rng=rng)
    c2 = (xyz[:, 2] / np.linalg.norm(xyz, axis=1)) ** 2
    assert abs(c2.mean() - 0.6) < 4.0 * math.sqrt(12.0 / 175.0 / count)
    # Real 3d_xy: |psi|^2 ~ sin^2(2 phi): <cos 4 phi> = -1/2.
    xyz, phase = O.sample_orbital(3, 2, -2, count=count, rng=rng, real=True)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    assert abs(np.cos(4.0 * phi).mean() + 0.5) < 4.0 * math.sqrt(0.125 / count)
    assert set(np.unique(phase)) <= {0.0, np.pi}
    # A whole shell: radii from r^2 R^2 (2s: <r> = 6 a0 for Z = 1).
    r = np.linalg.norm(O.sample_shell(2, 0, 1.0, count, rng=rng), axis=1)
    assert abs(r.mean() - 6.0) < 4.0 * math.sqrt(42.0 - 36.0) / math.sqrt(count)


def test_energy_levels():
    # E_n = -Ry Z^2 / n^2 (infinite nuclear mass): H 13.605693 eV, He+ 54.42277 eV.
    assert O.energy(1) == pytest.approx(-13.605693122994)
    assert O.energy(1, Z=2) == pytest.approx(-54.42277249)
    assert O.energy(3) == pytest.approx(-13.605693122994 / 9)


def test_hydrogen_lines():
    proton = D.M_PROTON
    _, lyman = O.transition(2, 1, 1, proton)
    _, balmer = O.transition(3, 2, 1, proton)
    # Bohr theory with the reduced mass (R_H = 10967758.34 /m): 121.5684 and 656.4696 nm.
    assert lyman == pytest.approx(121.5684, abs=2e-4)
    assert balmer == pytest.approx(656.4696, abs=1e-3)
    # NIST (fine-structure weighted): 121.5670 and 656.461. The difference is fine
    # structure + Lamb shift (left out), under 1.2e-5 relative.
    assert lyman == pytest.approx(121.5670, rel=1.5e-5)
    assert balmer == pytest.approx(656.461, rel=1.5e-5)
    # Without the reduced mass the line is 0.36 nm off: the factor is needed.
    _, infinite = O.transition(3, 2)
    assert infinite == pytest.approx(656.112, abs=1e-3)
    # The module's own nuclear mass for H-1 (A u - m_e, ignores the 7.3 MeV
    # mass excess): within 0.003 nm of the proton-mass line.
    _, approx = O.transition(3, 2, 1, O.nuclear_mass_for(1, 1))
    assert abs(approx - balmer) < 3e-3
    # Air wavelength of H-alpha (NIST): 656.279 nm.
    assert O.air_wavelength(656.461) == pytest.approx(656.279, abs=2e-3)


def test_configurations_and_slater():
    for Z in range(1, O.Z_MAX_CONFIGURATION + 1):
        assert sum(O.configuration(Z).values()) == Z
    assert O.configuration_text(24) == "[Ar] 3d5 4s1"
    assert O.configuration_text(29) == "[Ar] 3d10 4s1"
    assert O.configuration_text(46) == "[Kr] 4d10"
    assert O.configuration_text(64) == "[Xe] 4f7 5d1 6s2"
    assert O.configuration_text(26) == "[Ar] 3d6 4s2"
    assert O.slater_charge(7, 2, 1) == pytest.approx(3.90)
    assert O.slater_charge(30, 3, 2) == pytest.approx(8.85)
    assert O.slater_charge(30, 4, 0) == pytest.approx(4.35)
    assert O.slater_charge(2, 1, 0) == pytest.approx(1.70)  # He 1s: 2 - 0.30


# ===================================================================== nuclear data


def test_label_parsing():
    assert D.parse_label("U-238") == (92, 238, "")
    assert D.parse_label("238U") == (92, 238, "")
    assert D.parse_label("Pa-234m") == (91, 234, "m")
    assert D.parse_label("234Pam") == (91, 234, "m")
    assert D.parse_label("152Sm") == (62, 152, "")  # not sulfur-152m
    assert D.parse_label("241Am") == (95, 241, "")
    assert D.parse_label("n") == (0, 1, "")
    assert D.parse_label("Xx-3") is None


@pytest.mark.parametrize(
    "Z,A,expected,tol",
    [(2, 4, 28.2957, 1e-4), (26, 56, 492.260, 1e-3), (92, 235, 1783.87, 1e-2)],  # AME2020
)
def test_binding_energies(table, Z, A, expected, tol):
    # Tolerance: half the last digit of the quoted reference.
    assert table.binding(Z, A) == pytest.approx(expected, abs=tol)
    # The formula is fitted to A >= 20 with 3.2 MeV rms: within 3 rms there.
    if A >= 20:
        assert abs(float(D.semf_binding(Z, A)) - expected) < 3 * 3.2


def test_excerpt_matches_the_files(files):
    # The built-in rows are said to be copied from the two files.
    excerpt = D.Table(use_files=False)
    for key, n in excerpt.nuclides.items():
        f = files.get(*key)
        assert f is not None, key
        assert f.mass_excess == pytest.approx(n.mass_excess, abs=1e-9), key
        assert f.half_life == n.half_life or f.half_life == pytest.approx(n.half_life), key
        assert [m for m in f.modes if m[0] in dict(n.modes)] == list(n.modes), key


def test_semf_fit_is_reproduced(files):
    # Linear least squares of the five terms on the measured B with A >= 20 (the
    # fit atom_data says it made): the same coefficients and 3.2 MeV rms.
    rows, target = [], []
    for (Z, A), (_, estimated) in files.masses.items():
        if estimated or A < 20 or Z == 0:
            continue
        N_ = A - Z
        pairing = ((Z % 2 == 0) & (N_ % 2 == 0)) - ((Z % 2 == 1) & (N_ % 2 == 1))
        rows.append([A, -(A ** (2 / 3)), -Z * (Z - 1) / A ** (1 / 3), -((A - 2 * Z) ** 2) / A,
                     pairing / math.sqrt(A)])  # fmt: skip
        target.append(files.binding(Z, A))
    coeff, *_ = np.linalg.lstsq(np.array(rows), np.array(target), rcond=None)
    code = [D.SEMF_VOLUME, D.SEMF_SURFACE, D.SEMF_COULOMB, D.SEMF_ASYMMETRY, D.SEMF_PAIRING]
    assert np.allclose(coeff, code, rtol=5e-3)
    residual = np.array(target) - np.array(rows) @ np.array(code)
    assert math.sqrt(np.mean(residual**2)) == pytest.approx(3.2, abs=0.2)


def test_valley_of_stability(table):
    # The lightest isobars: Fe-56, Ru-100, Sm-150, Pb-208 (AME2020).
    for A, Z in ((56, 26), (100, 44), (150, 62), (208, 82)):
        assert N.valley_z(A) == Z


def test_nuclear_radius_and_barrier():
    assert float(N.radius(27)) == pytest.approx(3.6)  # 1.2 fm * 3
    # D + T touching: 1.44 MeV fm / (1.2 (2^1/3 + 3^1/3) fm) = 0.444 MeV.
    assert float(N.coulomb_barrier(1, 2, 1, 3)) == pytest.approx(0.4441, abs=1e-3)


def test_half_lives(table):
    assert table.get(6, 14).half_life / YEAR == pytest.approx(5700.0)
    assert table.get(92, 238).half_life / YEAR == pytest.approx(4.463e9)
    assert table.get(0, 1).half_life == pytest.approx(609.8)
    assert table.get(26, 56).stable


@pytest.mark.parametrize(
    "nuclide,mode,expected,tol",
    [
        ("U-238", "alpha", 4.2699, 2e-3),
        ("C-14", "beta-", 0.156476, 2e-6),
        ("n", "beta-", 0.782347, 2e-6),
        ("H-3", "beta-", 0.018591, 2e-6),
        ("K-40", "beta-", 1.3109, 3e-4),
        ("K-40", "EC", 1.5044, 3e-4),  # CAPTURE_ONLY: NUBASE's B+ shown as capture
        ("C-11", "beta+", 1.9820 - 2 * D.M_ELECTRON, 1e-3),
        ("Be-7", "EC", 0.86189, 1e-4),
    ],
)
def test_q_values(table, nuclide, mode, expected, tol):
    Z, A, iso = D.parse_label(nuclide)
    found = [b for b in K.branches(table, Z, A, iso) if b.mode == mode]
    assert found, K.branches(table, Z, A, iso)
    assert found[0].q == pytest.approx(expected, abs=tol)


def test_nucleon_emission_q(files):
    # He-5 -> He-4 + n: 0.735 MeV; Li-5 -> He-4 + p: 1.965 MeV (AME2020).
    (b,) = K.branches(files, 2, 5)
    assert (b.mode, b.daughter) == ("n", (2, 4, "")) and b.q == pytest.approx(0.735, abs=2e-3)
    (b,) = K.branches(files, 3, 5)
    assert (b.mode, b.daughter) == ("p", (2, 4, "")) and b.q == pytest.approx(1.965, abs=2e-3)


# ===================================================================== decay kinematics


def _check_conserved(event_mass, products):
    total = sum(p4 for *_, p4 in products)
    assert total[0] == pytest.approx(event_mass, rel=1e-14, abs=1e-9)
    assert np.allclose(total[1:], 0.0, atol=1e-9)
    for _, mass, p4 in products:
        assert p4[0] ** 2 - p4[1:] @ p4[1:] == pytest.approx(mass**2, rel=1e-9, abs=1e-9)


DECAYS = ["U-238", "C-14", "n", "H-3", "C-11", "Be-7", "Tc-99m", "Ba-137m", "K-40", "Cs-137", "Bi-212"]


@pytest.mark.parametrize("nuclide", DECAYS)
def test_decay_conserves_four_momentum(table, nuclide):
    rng = np.random.default_rng(3)
    Z, A, iso = D.parse_label(nuclide)
    for b in K.branches(table, Z, A, iso):
        if b.mode == "SF":
            continue
        for _ in range(30):
            event = K.decay(table, Z, A, iso, rng=rng, branch=b)
            _check_conserved(event.parent_mass, [(p.name, p.mass, p.p4) for p in event.products])
            released = sum(p.kinetic for p in event.products)
            assert released == pytest.approx(b.q, rel=1e-8, abs=1e-9)


def test_nucleon_emission_conserves(files):
    rng = np.random.default_rng(4)
    for Z, A in ((2, 5), (3, 5)):
        (b,) = K.branches(files, Z, A)
        event = K.decay(files, Z, A, rng=rng, branch=b)
        _check_conserved(event.parent_mass, [(p.name, p.mass, p.p4) for p in event.products])
        assert sum(p.kinetic for p in event.products) == pytest.approx(b.q, rel=1e-9)


def test_alpha_line_with_recoil(table):
    # U-238 -> Th-234: T_alpha = 4198 keV (ENSDF); non-relativistic Q M_d / (M_d + M_a) = 4.1974.
    event = K.decay(table, 92, 238, branch=K.branches(table, 92, 238)[0])
    alpha = event.products[0]
    assert alpha.kinetic == pytest.approx(4.198, abs=2e-3)


def test_beta_spectrum_means():
    # ENSDF / LNHB mean energies of two allowed beta decays.
    assert K.beta_mean_energy(0.156476, 7) == pytest.approx(0.04947, rel=0.01)
    assert K.beta_mean_energy(1.71066, 16) == pytest.approx(0.6949, rel=0.01)
    # Sampling reproduces the mean; no energy above the endpoint.
    rng = np.random.default_rng(5)
    T = K.sample_beta_energy(rng, 1.71066, 16, size=100000)
    assert T.max() <= 1.71066
    assert T.mean() == pytest.approx(K.beta_mean_energy(1.71066, 16), abs=4 * T.std() / 316.0)
    # The Coulomb factor raises the electron spectrum and lowers the positron
    # spectrum at low energy: positrons have the higher mean.
    assert K.beta_mean_energy(1.0, 30, positron=True) > K.beta_mean_energy(1.0, 0)
    assert K.beta_mean_energy(1.0, 30) < K.beta_mean_energy(1.0, 0)


def test_beta_electron_energy_range(table):
    # Electron kinetic energy spans 0 .. Q (less a recoil of order Q^2 / 2M).
    rng = np.random.default_rng(6)
    q = K.branches(table, 0, 1)[0].q
    energies = [K.decay(table, 0, 1, rng=rng).products[0].kinetic for _ in range(3000)]
    assert 0.0 <= min(energies) and max(energies) <= q
    assert max(energies) > 0.9 * q


# ===================================================================== decay statistics


def _single(half_life):
    return [K.Species((0, 0, ""), "X", half_life, [(1, 1.0, "beta-")]),
            K.Species((0, 0, "x"), "Y", np.inf, [])]  # fmt: skip


def test_monte_carlo_decay_law():
    # N(t) = N0 exp(-lambda t): binomial error sqrt(N0 p (1 - p)), p = exp(-lambda t),
    # also at times many half-lives long and with steps much longer than one.
    n0, half = 200000, 3.0
    sample = K.DecaySample(_single(half), n0, rng=np.random.default_rng(7))
    for t in (0.1, 1.0, 3.0, 10.0, 30.0, 60.0):
        p = math.exp(-K.LN2 * t / half)
        left = sample.populations(t)[0]
        assert abs(left - n0 * p) <= 4.0 * math.sqrt(n0 * p * (1 - p)) + 1.0
    assert sample.populations(1e6 * half)[0] == 0
    # Decays counted over steps of 20 half-lives add up to all of them.
    edges = np.arange(0.0, 400.0, 60.0)
    total = sum(len(sample.decays_between(a, b)[0]) for a, b in zip(edges[:-1], edges[1:]))
    assert total == n0 - sample.populations(edges[-1])[0]


def test_bateman_two_members_and_equal_rates():
    # N2 = N0 l1 / (l2 - l1) (exp(-l1 t) - exp(-l2 t)); for l1 = l2 = l: N0 l t exp(-l t).
    s = [K.Species(None, "a", 2.0, [(1, 1.0, "x")]), K.Species(None, "b", 5.0, [(2, 1.0, "x")]),
         K.Species(None, "c", np.inf, [])]  # fmt: skip
    t = np.array([0.0, 1.0, 4.0, 20.0])
    l1, l2 = K.LN2 / 2.0, K.LN2 / 5.0
    n = K.bateman(s, t)
    assert np.allclose(n[1], l1 / (l2 - l1) * (np.exp(-l1 * t) - np.exp(-l2 * t)), atol=1e-12)
    assert np.allclose(n.sum(axis=0), 1.0, atol=1e-12)
    s[1].half_life = 2.0
    n = K.bateman(s, t)
    assert np.allclose(n[1], l1 * t * np.exp(-l1 * t), atol=1e-6)


def test_branched_chain_monte_carlo_against_bateman():
    # a -> b (70%) or c (30%); b -> d; c -> d; d stable. Half-lives 1, 3, 0.5.
    s = [K.Species(None, "a", 1.0, [(1, 0.7, "x"), (2, 0.3, "y")]),
         K.Species(None, "b", 3.0, [(3, 1.0, "x")]),
         K.Species(None, "c", 0.5, [(3, 1.0, "x")]),
         K.Species(None, "d", np.inf, [])]  # fmt: skip
    n0 = 100000
    sample = K.DecaySample(s, n0, rng=np.random.default_rng(8))
    for t in (0.5, 2.0, 6.0, 20.0):
        expected = K.bateman(s, t, n0)[:, 0]
        got = sample.populations(t)
        p = expected / n0
        assert np.all(np.abs(got - expected) <= 4.0 * np.sqrt(n0 * p * (1 - p)) + 1.0), (t, got, expected)


def test_uranium_series(table):
    s = K.build_chain(table, 92, 238)
    names = [x.name for x in s]
    assert names[0] == "U-238" and names[-1] == "Pb-206"
    assert [x.name for x in s if x.half_life == np.inf] == ["Pb-206"]
    # Bateman keeps the total and reaches secular equilibrium: after 1e7 years
    # (> 30 half-lives of Th-230), Ra-226 decays as often as U-238 (activity ratio 1).
    t = 1e7 * YEAR
    n = K.bateman(s, [1e3 * YEAR, t, 4.463e9 * YEAR])
    assert np.allclose(n.sum(axis=0), 1.0, atol=1e-9)
    act = K.activity(s, n[:, 1])
    ra = names.index("Ra-226")
    assert act[ra] / act[0] == pytest.approx(1.0, rel=1e-3)
    assert n[0, 2] == pytest.approx(0.5, rel=1e-9)  # one half-life
    # The Monte Carlo sample, at a time spanning microsecond and gigayear half-lives.
    sample = K.DecaySample(s, 20000, rng=np.random.default_rng(9))
    left = sample.populations(4.463e9 * YEAR)
    assert abs(left[0] - 10000) < 4 * math.sqrt(5000)
    assert left[0] + left[-1] > 20000 - 10  # nearly nothing in between


# ============================================================================ fusion


def test_fusion_q_values(table):
    assert R.reaction_q(table, R.REACTIONS["D + T -> He-4 + n"]) == pytest.approx(17.5893, abs=2e-4)
    assert R.reaction_q(table, R.REACTIONS["D + D -> He-3 + n"]) == pytest.approx(3.2689, abs=2e-4)
    assert R.reaction_q(table, R.REACTIONS["D + D -> T + p"]) == pytest.approx(4.0327, abs=2e-4)
    assert R.reaction_q(table, R.REACTIONS["D + He-3 -> He-4 + p"]) == pytest.approx(18.3531, abs=2e-4)
    assert R.pp_chain_summary(table)["total"] == pytest.approx(26.731, abs=2e-3)


def test_dt_kinematics(table):
    event = R.react(table, R.REACTIONS["D + T -> He-4 + n"], np.random.default_rng(10))
    energies = dict(event.kinetic_energies())
    # Relativistic two-body: T_n = Q (Q + 2 m_alpha) / 2M, T_alpha = Q (Q + 2 m_n) / 2M,
    # M = m_alpha + m_n + Q: 14.028 and 3.561 MeV (hand calculation with CODATA m_n and
    # AME2020 m_alpha = 3727.379 MeV). The textbook 14.05 / 3.54 is the non-relativistic
    # split Q m_alpha / (m_alpha + m_n); relativity moves 0.02 MeV to the alpha.
    q, m_a, m_n = 17.5893, 3727.3794, 939.5654
    M = m_a + m_n + q
    assert energies["n"] == pytest.approx(q * (q + 2 * m_a) / (2 * M), abs=2e-3)
    assert energies["He-4"] == pytest.approx(q * (q + 2 * m_n) / (2 * M), abs=2e-3)
    assert energies["n"] == pytest.approx(14.028, abs=2e-3)
    assert energies["n"] == pytest.approx(14.05, abs=0.03)
    assert energies["He-4"] == pytest.approx(3.54, abs=0.03)


@pytest.mark.parametrize("name", list(R.REACTIONS))
def test_reactions_conserve_four_momentum(table, name):
    rng = np.random.default_rng(11)
    reaction = R.REACTIONS[name]
    for _ in range(50):
        event = R.react(table, reaction, rng)
        _check_conserved(event.total_mass, event.products)
        assert sum(e for _, e in event.kinetic_energies()) == pytest.approx(event.q, rel=1e-7)


def test_gamow_factor_by_hand():
    # p + p: mu = m_p / 2 (bare protons), E_G = 2 mu (pi alpha)^2 = 0.49310 MeV.
    mu = D.M_PROTON / 2.0
    e_g = R.gamow_energy(1, 1, mu)
    assert e_g == pytest.approx(2 * mu * (math.pi / 137.035999084) ** 2, rel=1e-10)
    assert e_g * 1e3 == pytest.approx(493.1, abs=0.1)
    # At E = 1 keV: exp(-sqrt(493.10 / 1)) = exp(-22.206) = 2.27e-10.
    assert R.tunnelling(1e-3, e_g) == pytest.approx(math.exp(-math.sqrt(e_g / 1e-3)), rel=1e-12)
    assert R.tunnelling(1e-3, e_g) == pytest.approx(2.27e-10, rel=0.01)
    # Same as exp(-2 pi eta), eta = Z1 Z2 alpha / v, v = sqrt(2 E / mu) (non-relativistic).
    v = math.sqrt(2 * 1e-3 / mu)
    assert R.tunnelling(1e-3, e_g) == pytest.approx(math.exp(-2 * math.pi * D.ALPHA / v), rel=1e-12)
    # Gamow peak in the Sun's core, 15 MK: kT = 1.2926 keV, E0 = 5.9 keV; it is the maximum.
    kT = R.thermal_energy_kev(1.5e7)
    assert kT == pytest.approx(1.2926, abs=1e-4)
    e0, width = R.gamow_peak(e_g * 1e3, kT)
    assert e0 == pytest.approx(5.905, abs=0.01)
    E = np.linspace(1.0, 30.0, 200001)
    assert E[np.argmax(R.gamow_window(E, e_g * 1e3, kT))] == pytest.approx(e0, abs=1e-3)
    assert width == pytest.approx(4.0 * math.sqrt(e0 * kT / 3.0))


# =========================================================================== fission


def test_fission_u235():
    t = D.table()
    rng = np.random.default_rng(12)
    events = [R.fission(t, "U-235", rng) for _ in range(3000)]
    for e in events[:300]:
        (zl, al, ml, pl), (zh, ah, mh, ph) = e.fragments
        assert zl + zh == 92 and al + ah + len(e.neutrons) == 236
        products = [("l", ml, pl), ("h", mh, ph)] + [("n", e.neutron_mass, p) for p in e.neutrons]
        products += [("g", 0.0, p) for p in e.gammas]
        _check_conserved(e.initial_mass, products)
        assert e.fragment_energy + e.neutron_energy + e.gamma_energy == pytest.approx(e.q, rel=1e-9)
    nu = np.array([len(e.neutrons) for e in events])
    # The distribution's mean is set to nu_bar exactly; the sample within 4 sigma.
    p = R.multiplicity_distribution(2.43)
    assert p @ np.arange(len(p)) == pytest.approx(2.43, abs=1e-9)
    assert abs(nu.mean() - 2.43) < 4 * nu.std() / math.sqrt(len(nu))
    # Total energy (prompt + fragments' beta decays, antineutrinos included): ~202.5 MeV.
    total = np.mean([e.q + e.later for e in events])
    assert total == pytest.approx(202.5, abs=3.0)
    # Prompt-neutron energies: Watt mean 3a/2 + a^2 b / 4 = 2.031 MeV.
    energies = np.concatenate([[p[0] - e.neutron_mass for p in e.neutrons] for e in events])
    assert abs(energies.mean() - 2.031) < 4 * energies.std() / math.sqrt(len(energies))
    # Fragment kinetic energy (post-neutron) against the measured pre-neutron 170.5 MeV:
    # within 3 %. (The model gives 166.3 here; see the review.)
    assert np.mean([e.fragment_energy for e in events]) == pytest.approx(170.5, rel=0.03)
    # Mass yield peaks: light ~96.5, heavy ~139.5 before neutron emission.
    heavy = np.array([e.primary_masses[1] for e in events])
    assert heavy.mean() == pytest.approx(139.5, abs=0.5)


def test_watt_sampler_mean():
    rng = np.random.default_rng(13)
    a, b = 0.988, 2.249
    e = R.sample_watt(rng, a, b, size=200000)
    assert e.mean() == pytest.approx(1.5 * a + 0.25 * a * a * b, abs=4 * e.std() / math.sqrt(len(e)))


def test_escape_probability_against_transport():
    # P(1 mfp) = 3/8 (2 - 1 + 3 e^-2) = 0.52725; the module's straight-line transport agrees.
    assert float(R.escape_probability(1.0)) == pytest.approx(0.375 * (1 + 3 * math.exp(-2)))
    assert float(R.escape_probability(1e-3)) == pytest.approx(1 - 0.75e-3, rel=1e-5)
    chain = R.ChainReaction(1.0, 0.0, [1.0], rng=np.random.default_rng(14))
    chain.inject(20000)
    chain.step(1e3)
    p = float(R.escape_probability(1.0))
    assert chain.count == 0
    assert abs(chain.leaks / 20000 - p) < 4 * math.sqrt(p * (1 - p) / 20000)
