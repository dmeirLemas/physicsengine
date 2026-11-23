"""Electron clouds: hydrogen-like orbitals, their energies and transitions, and shell filling.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, atom
module). Review the physics.

Units: lengths in Bohr radii a0 (5.29177e-11 m), energies in eV, wavelengths
in nm (in vacuum). Angles: theta from the z axis, phi around it.

Exact (non-relativistic Schroedinger equation, one electron, point nucleus):

    psi_nlm(r, theta, phi) = R_nl(r) Y_lm(theta, phi)
    R_nl(r) = sqrt((2Z/n)^3 (n-l-1)! / (2n (n+l)!)) exp(-rho/2) rho^l L^(2l+1)_(n-l-1)(rho),
    rho = 2 Z r / n,

with L the associated Laguerre polynomials and Y_lm the spherical harmonics
(Condon-Shortley phase), and E_n = -Ry (mu / m_e) Z^2 / n^2, where
mu / m_e = 1 / (1 + m_e / M_nucleus) is the reduced-mass factor. Not
included: fine structure (spin-orbit, relativistic mass; 4.5e-5 eV in
hydrogen's n = 2), the Lamb shift, hyperfine structure.

An approximation, for atoms with more than one electron: the ground-state
configuration is the Madelung (aufbau) order with the known exceptions
listed in EXCEPTIONS, and each subshell is drawn as a hydrogen-like orbital
with the effective charge of Slater's rules (J. C. Slater, Phys. Rev. 36,
57, 1930). This illustrates shell structure; it is not a Hartree-Fock or
any other self-consistent calculation, and its radii and energies are rough
(tens of percent).
"""

# ruff: noqa: E741  (l is the orbital quantum number throughout)
from math import factorial

import numpy as np

from physengine.modules.atom_data import HC_EV_NM, M_ELECTRON, RYDBERG_EV, U_MEV

L_LETTERS = "spdfghik"
SERIES = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett", 5: "Pfund", 6: "Humphreys"}
N_MAX = 7  # largest principal quantum number offered

# Names of the real orbitals, by l and m (m > 0: cos(m phi), m < 0: sin(|m| phi)).
REAL_NAMES = {
    (0, 0): "s",
    (1, 0): "p_z", (1, 1): "p_x", (1, -1): "p_y",
    (2, 0): "d_z2", (2, 1): "d_xz", (2, -1): "d_yz", (2, 2): "d_x2-y2", (2, -2): "d_xy",
    (3, 0): "f_z3", (3, 1): "f_xz2", (3, -1): "f_yz2", (3, 2): "f_z(x2-y2)", (3, -2): "f_xyz",
    (3, 3): "f_x(x2-3y2)", (3, -3): "f_y(3x2-y2)",
}  # fmt: skip


# ------------------------------------------------------------- special functions


def laguerre(k, a, x):
    """Associated Laguerre polynomial L_k^(a)(x), by the three-term recurrence
    (j + 1) L_(j+1) = (2j + 1 + a - x) L_j - (j + a) L_(j-1)."""
    x = np.asarray(x, dtype=float)
    previous = np.ones_like(x)
    if k == 0:
        return previous
    current = 1.0 + a - x
    for j in range(1, k):
        previous, current = current, ((2 * j + 1 + a - x) * current - (j + a) * previous) / (j + 1)
    return current


def legendre(l, m, x):
    """Associated Legendre function P_l^m(x) for 0 <= m <= l, with the
    Condon-Shortley phase (-1)^m, by upward recurrence in l."""
    x = np.asarray(x, dtype=float)
    pmm = np.ones_like(x)
    if m > 0:
        root = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        for j in range(1, m + 1):
            pmm = -pmm * (2 * j - 1) * root
    if l == m:
        return pmm
    pm1 = x * (2 * m + 1) * pmm
    for j in range(m + 2, l + 1):
        pmm, pm1 = pm1, (x * (2 * j - 1) * pm1 - (j + m - 1) * pmm) / (j - m)
    return pm1


def radial(n, l, r, Z=1.0):
    """R_nl(r) for nuclear charge Z, normalised so that the integral of
    R^2 r^2 dr is 1. r in a0."""
    if not (n >= 1 and 0 <= l < n):
        raise ValueError(f"no orbital with n = {n}, l = {l}")
    rho = 2.0 * Z * np.asarray(r, dtype=float) / n
    norm = np.sqrt((2.0 * Z / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l)))
    return norm * np.exp(-0.5 * rho) * rho**l * laguerre(n - l - 1, 2 * l + 1, rho)


def theta_part(l, m, cos_theta):
    """Theta_lm: the theta-dependent factor of Y_lm, normalised so that
    2 pi times the integral of Theta^2 d(cos theta) is 1 (m may be negative)."""
    am = abs(m)
    norm = np.sqrt((2 * l + 1) / (4.0 * np.pi) * factorial(l - am) / factorial(l + am))
    value = norm * legendre(l, am, cos_theta)
    if m < 0 and am % 2 == 1:
        value = -value  # Y_l,-m = (-1)^m conj(Y_lm)
    return value


def spherical_harmonic(l, m, cos_theta, phi):
    """Complex Y_lm(theta, phi)."""
    return theta_part(l, m, cos_theta) * np.exp(1j * m * np.asarray(phi, dtype=float))


def real_harmonic(l, m, cos_theta, phi):
    """Real spherical harmonic: m > 0 goes as cos(m phi), m < 0 as sin(|m| phi).

    S_l0 = Y_l0; S_lm = sqrt(2) (-1)^m Re Y_l|m| (m > 0) or sqrt(2) (-1)^m Im Y_l|m| (m < 0).
    """
    am = abs(m)
    if m == 0:
        return theta_part(l, 0, cos_theta) * np.ones_like(np.asarray(phi, dtype=float))
    base = np.sqrt(2.0) * (-1) ** am * theta_part(l, am, cos_theta)
    return base * (np.cos(am * phi) if m > 0 else np.sin(am * phi))


def psi(n, l, m, r, cos_theta, phi, Z=1.0, real=False):
    """The wave function psi_nlm at spherical coordinates (r [a0], cos theta, phi)."""
    angular = real_harmonic if real else spherical_harmonic
    return radial(n, l, r, Z) * angular(l, m, cos_theta, phi)


def orbital_name(n, l, m, real=False):
    if real:
        return f"{n}{REAL_NAMES.get((l, m), L_LETTERS[l])}"
    return f"{n}{L_LETTERS[l]}, m = {m:+d}" if l > 0 else f"{n}s"


# ----------------------------------------------------------------- expectations


def mean_radius(n, l, Z=1.0):
    """<r> = a0 (3 n^2 - l (l + 1)) / (2 Z), in a0."""
    return (3.0 * n * n - l * (l + 1)) / (2.0 * Z)


def most_probable_radius_circular(n, Z=1.0):
    """Where r^2 R^2 peaks for the nodeless orbital l = n - 1: n^2 / Z (the
    radius of Bohr's n-th orbit)."""
    return n * n / Z


def radial_grid(n, Z=1.0, points=6000):
    """A grid of r [a0] wide enough to hold the n-th shell to 1e-12."""
    return np.linspace(0.0, (3.0 * n * n + 25.0 * n + 20.0) / Z, points)


# --------------------------------------------------------------------- sampling


def _inverse_cdf(rng, x, density, size):
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x))])
    return np.interp(rng.random(size) * cdf[-1], cdf, x)


def sample_radius(rng, n, l, Z, size):
    """Radii [a0] drawn from the radial probability density r^2 R_nl^2."""
    r = radial_grid(n, Z)
    return _inverse_cdf(rng, r, (r * radial(n, l, r, Z)) ** 2, size)


def sample_orbital(n, l, m, Z=1.0, count=60000, real=False, rng=None):
    """Points distributed as |psi_nlm|^2, with the wave function's phase at each.

    The density factorises, so r, cos theta and phi are drawn separately,
    each by inverting its cumulative distribution on a fine grid.

    Returns (xyz (count, 3) [a0], phase (count,) [rad]). For a real orbital
    the phase is 0 where psi > 0 and pi where psi < 0.
    """
    rng = rng or np.random.default_rng()
    r = sample_radius(rng, n, l, Z, count)
    x = np.linspace(-1.0, 1.0, 4001)
    cos_theta = _inverse_cdf(rng, x, theta_part(l, m, x) ** 2, count)
    if real and m != 0:
        grid = np.linspace(0.0, 2.0 * np.pi, 4001)
        wave = np.cos(abs(m) * grid) if m > 0 else np.sin(abs(m) * grid)
        phi = _inverse_cdf(rng, grid, wave**2, count)
    else:
        phi = rng.uniform(0.0, 2.0 * np.pi, count)
    value = psi(n, l, m, r, cos_theta, phi, Z, real)
    phase = np.where(value < 0.0, np.pi, 0.0) if real else np.angle(value)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    xyz = np.stack([r * sin_theta * np.cos(phi), r * sin_theta * np.sin(phi), r * cos_theta], 1)
    return xyz, phase


def sample_shell(n, l, Z_eff, count, rng=None):
    """Points for a whole subshell, averaged over directions: radii from
    r^2 R_nl^2, directions uniform. (A filled subshell is exactly spherical:
    the sum over m of |Y_lm|^2 is (2l + 1) / 4 pi.)"""
    rng = rng or np.random.default_rng()
    r = sample_radius(rng, n, l, Z_eff, count)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return r[:, None] * direction


# ----------------------------------------------------------- energies and lines


def reduced_mass_factor(nuclear_mass=None):
    """mu / m_e = 1 / (1 + m_e / M) for a nucleus of mass M [MeV]; 1 for M = None."""
    return 1.0 if nuclear_mass is None else 1.0 / (1.0 + M_ELECTRON / nuclear_mass)


def energy(n, Z=1.0, nuclear_mass=None):
    """E_n [eV] of a one-electron atom: -Ry (mu / m_e) Z^2 / n^2."""
    return -RYDBERG_EV * reduced_mass_factor(nuclear_mass) * Z * Z / (n * n)


def transition(n_upper, n_lower, Z=1.0, nuclear_mass=None):
    """(photon energy [eV], vacuum wavelength [nm]) of the line n_upper -> n_lower:
    the Rydberg formula 1 / lambda = R Z^2 (1 / n_lower^2 - 1 / n_upper^2)."""
    if n_upper <= n_lower:
        raise ValueError("the upper level must lie above the lower one")
    photon = energy(n_upper, Z, nuclear_mass) - energy(n_lower, Z, nuclear_mass)
    return photon, HC_EV_NM / photon


def dipole_allowed(l_upper, m_upper, l_lower, m_lower):
    """Electric-dipole selection rules: delta l = +-1 and delta m = 0, +-1."""
    return abs(l_upper - l_lower) == 1 and abs(m_upper - m_lower) <= 1


def air_wavelength(vacuum_nm):
    """Wavelength in standard dry air (15 C, 101325 Pa), from the vacuum one:
    the refractive index of Edlen's 1966 formula as given by Morton (2000,
    ApJS 130, 403), valid above 200 nm:
    n - 1 = 8.34254e-5 + 2.406147e-2 / (130 - s^2) + 1.5998e-4 / (38.9 - s^2),
    s = 1 / lambda [micrometres]. Entered from memory, to be checked; it
    reproduces H-alpha 656.461 (vacuum) -> 656.280 nm (air)."""
    s2 = (1e3 / vacuum_nm) ** 2
    index = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)
    return vacuum_nm / index


def wavelength_band(nm):
    if nm < 10.0:
        return "X-ray"
    if nm < 380.0:
        return "ultraviolet"
    if nm <= 750.0:
        return "visible"
    if nm < 1e6:
        return "infrared"
    return "microwave or radio"


def wavelength_rgb(nm):
    """An approximate screen colour for a wavelength (display only): the hue
    of visible light, violet-white below 380 nm, deep red above 750 nm."""
    if nm < 380.0:
        return (0.75, 0.6, 1.0)
    if nm > 750.0:
        return (0.8, 0.15, 0.1)
    if nm < 440.0:
        rgb = ((440.0 - nm) / 60.0, 0.0, 1.0)
    elif nm < 490.0:
        rgb = (0.0, (nm - 440.0) / 50.0, 1.0)
    elif nm < 510.0:
        rgb = (0.0, 1.0, (510.0 - nm) / 20.0)
    elif nm < 580.0:
        rgb = ((nm - 510.0) / 70.0, 1.0, 0.0)
    elif nm < 645.0:
        rgb = (1.0, (645.0 - nm) / 65.0, 0.0)
    else:
        rgb = (1.0, 0.0, 0.0)
    return tuple(0.25 + 0.75 * c for c in rgb)


# ------------------------------------------------------------------ many electrons

# Ground-state configurations that depart from the Madelung order (NIST
# Atomic Spectra Database ground levels), written as the changes to make:
# {Z: {(n, l): occupancy}}. Entered from memory, to be checked.
EXCEPTIONS = {
    24: {(3, 2): 5, (4, 0): 1},  # Cr  3d5 4s1
    29: {(3, 2): 10, (4, 0): 1},  # Cu  3d10 4s1
    41: {(4, 2): 4, (5, 0): 1},  # Nb  4d4 5s1
    42: {(4, 2): 5, (5, 0): 1},  # Mo  4d5 5s1
    44: {(4, 2): 7, (5, 0): 1},  # Ru  4d7 5s1
    45: {(4, 2): 8, (5, 0): 1},  # Rh  4d8 5s1
    46: {(4, 2): 10, (5, 0): 0},  # Pd  4d10
    47: {(4, 2): 10, (5, 0): 1},  # Ag  4d10 5s1
    57: {(4, 3): 0, (5, 2): 1},  # La  5d1 6s2
    58: {(4, 3): 1, (5, 2): 1},  # Ce  4f1 5d1 6s2
    64: {(4, 3): 7, (5, 2): 1},  # Gd  4f7 5d1 6s2
    78: {(5, 2): 9, (6, 0): 1},  # Pt  5d9 6s1
    79: {(5, 2): 10, (6, 0): 1},  # Au  5d10 6s1
}
Z_MAX_CONFIGURATION = 86  # through radon: Slater gave effective n only up to n = 6
NOBLE = {2: "He", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 86: "Rn"}


def madelung_order(n_max=8):
    """Subshells (n, l) in filling order: by n + l, then by n."""
    shells = [(n, l) for n in range(1, n_max + 1) for l in range(n)]
    return sorted(shells, key=lambda nl: (nl[0] + nl[1], nl[0]))


def configuration(Z):
    """Ground-state configuration of the neutral atom: {(n, l): electrons},
    in Madelung order with the EXCEPTIONS applied. For Z up to 86."""
    if not 1 <= Z <= Z_MAX_CONFIGURATION:
        raise ValueError(f"configurations are given for Z = 1 to {Z_MAX_CONFIGURATION}")
    left, filled = Z, {}
    for n, l in madelung_order():
        if left <= 0:
            break
        take = min(left, 2 * (2 * l + 1))
        filled[(n, l)] = take
        left -= take
    for shell, count in EXCEPTIONS.get(Z, {}).items():
        if count:
            filled[shell] = count
        else:
            filled.pop(shell, None)
    assert sum(filled.values()) == Z
    return filled


def configuration_text(Z, short=True):
    """'[Ar] 3d5 4s1' (subshells ordered by n, then l)."""
    filled = configuration(Z)
    core, prefix = {}, ""
    if short:
        below = [z for z in NOBLE if z < Z]
        if below:
            core = configuration(max(below))
            prefix = f"[{NOBLE[max(below)]}] "
    parts = [
        f"{n}{L_LETTERS[l]}{count}"
        for (n, l), count in sorted(filled.items())
        if core.get((n, l)) != count
    ]
    return prefix + " ".join(parts)


SLATER_N = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2}  # Slater's effective n


def _slater_group(n, l):
    """Slater's groups: (1s) (2s2p) (3s3p) (3d) (4s4p) (4d) (4f) (5s5p) ..."""
    return (n, 0 if l <= 1 else l)


def slater_charge(Z, n, l, filled=None):
    """Effective nuclear charge Z - S for an electron in subshell (n, l), by Slater's rules:

    - electrons in groups further out shield nothing;
    - others in the same group: 0.35 each (0.30 in 1s);
    - for an s or p electron: 0.85 for each electron with principal number
      n - 1, 1.00 for each further in;
    - for a d or f electron: 1.00 for every electron in a group further in.
    """
    filled = filled or configuration(Z)
    own = _slater_group(n, l)
    screening = 0.0
    for (n2, l2), count in filled.items():
        group = _slater_group(n2, l2)
        if group == own:
            others = count - (1 if (n2, l2) == (n, l) else 0)
            screening += (0.30 if n == 1 else 0.35) * others
        elif group > own:
            continue  # further out: (n, group letter) orders the groups as Slater does
        elif l >= 2:
            screening += 1.00 * count
        elif n2 == n - 1:
            screening += 0.85 * count
        else:
            screening += 1.00 * count
    return Z - screening


def slater_energy(Z, n, l, filled=None):
    """Slater's one-electron energy [eV]: -Ry (Z_eff / n*)^2 (rough: tens of percent)."""
    return -RYDBERG_EV * (slater_charge(Z, n, l, filled) / SLATER_N[n]) ** 2


def subshells(Z):
    """The occupied subshells of element Z, inner first, each as a dict:
    n, l, name, electrons, z_eff (Slater), energy [eV] (Slater), mean radius
    [a0] of the hydrogen-like orbital with that charge."""
    filled = configuration(Z)
    rows = []
    for (n, l), count in sorted(filled.items()):
        z_eff = slater_charge(Z, n, l, filled)
        rows.append(
            {
                "n": n,
                "l": l,
                "name": f"{n}{L_LETTERS[l]}",
                "electrons": count,
                "z_eff": z_eff,
                "energy": slater_energy(Z, n, l, filled),
                "radius": mean_radius(n, l, z_eff),
            }
        )
    return rows


def nuclear_mass_for(Z, A):
    """A nucleus's mass [MeV], to the accuracy the reduced-mass factor needs:
    A u less Z electrons."""
    return A * U_MEV - Z * M_ELECTRON
