"""What the detector's material does to particles: energy loss, scattering, showers.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
4). Review the physics. Formulas and constants are from the Particle Data
Group's review "Passage of particles through matter" (in R. L. Workman et
al., Prog. Theor. Exp. Phys. 2022, 083C01), referred to as PDG below.

Units: GeV, metres. Material constants are given per material in the usual
units (g/cm^3, cm) and converted here.

1. Ionisation (heavy charged particles: muons, pions, kaons, protons).
   Mean energy loss per length, the Bethe formula (PDG eq. 34.5):
       -dE/dx = K z^2 (Z/A) rho / beta^2 [ 1/2 ln(2 m_e beta^2 gamma^2 T_max / I^2) - beta^2 - delta/2 ],
       T_max = 2 m_e beta^2 gamma^2 / (1 + 2 gamma m_e / M + (m_e / M)^2),
   K = 0.307075 MeV cm^2 / mol. The density effect delta is Sternheimer's
   parametrisation (PDG eq. 34.7): with x = log10(beta gamma),
       delta = 2 ln(10) x - Cbar                      for x >= x1,
               2 ln(10) x - Cbar + a (x1 - x)^k       for x0 <= x < x1,
               delta0 10^(2 (x - x0))                 for x < x0 (conductors; else 0),
       Cbar = 2 ln(I / hbar omega_p) + 1,
   hbar omega_p = 28.816 eV sqrt(rho Z/A) the plasma energy. The parameters
   (Cbar, x0, x1, a, k, delta0) of silicon, aluminium, iron and copper are
   those of Sternheimer, Berger & Seltzer 1984 (At. Data Nucl. Data Tables
   30, 261), as listed in the PDG's tables of atomic and nuclear properties.
   Lead tungstate is not in that table: its parameters follow the general
   recipe of Sternheimer & Peierls 1971 (Phys. Rev. B 3, 3681) for a solid
   with I > 100 eV: x1 = 3, k = 3, x0 = 0.326 Cbar - 1.5 (for Cbar > 5.215),
   a = (Cbar - 2 ln(10) x0) / (x1 - x0)^3. The parameters are a property of
   the solid, not of its average over a region with gaps. Each region is
   treated as its main material (the tracker as silicon, the hadron
   calorimeter's brass as copper). The minima that result are checked
   against the PDG's tabulated minimum ionisation in the tests. Only the
   mean loss is applied (no Landau fluctuations), and no radiative losses of
   muons (they pass ionisation only above some hundreds of GeV). A particle
   that runs out of kinetic energy stops.

2. Multiple Coulomb scattering. After a thickness x the direction has
   wandered by a Gaussian angle (per plane) of width (Highland; Lynch & Dahl
   1991; PDG eq. 34.16)
       theta_0 = 13.6 MeV / (beta p) z sqrt(x / X0) [1 + 0.038 ln(x z^2 / (X0 beta^2))].
   The formula is for the whole thickness crossed: because of the logarithm,
   applying it to each short step and adding the variances gives too little
   (15% for the tracker in 3 cm steps). So each step adds the difference
   of the formula's variance between the thickness crossed after and before
   it (scattering_step), with t = x / X0 summed along the path over all
   materials (the PDG's rule for layers: use the total x / X0):
       d(theta_0^2) = (13.6 MeV z / (beta p))^2 [f(t_after) - f(t_before)],
       f(t) = t [1 + 0.038 ln(t z^2 / beta^2)]^2,
   with beta p the current values. At constant momentum the steps then add
   up exactly to the formula for the total. The angle is Gaussian (no
   single-scattering tails); the sideways displacement within a step is
   left out.

3. Electrons radiate (bremsstrahlung). The fraction z of its energy an
   electron keeps after t radiation lengths follows the Bethe-Heitler
   distribution, -ln z ~ Gamma(shape t / ln 2, scale 1) (PDG, sec. 34.4.2;
   Bethe & Heitler 1934). The energy lost in a step is given to one photon
   along the electron's direction if it exceeds BREM_PHOTON_MIN.

4. Photons convert to e+ e- pairs with mean free path 9/7 X0 (PDG eq. 34.25).
   In the tracker the pair is made (the photon's energy shared between the
   two uniformly, a fair approximation to the Bethe-Heitler pair spectrum
   at these energies, both along the photon's direction) and followed; in
   the calorimeter the conversion starts the shower.

5. Electromagnetic showers (electrons and photons in the calorimeter). The
   mean energy deposited per radiation length t is a gamma distribution
   (Longo & Sestili 1975; PDG eq. 34.36):
       dE/dt = E b (b t)^(a-1) e^(-b t) / Gamma(a),  b = 0.5,
       t_max = (a - 1) / b = ln(E / E_c) + C,  C = -0.5 (electrons), +0.5 (photons),
   E_c the critical energy. Drawn as that profile along the particle's
   direction; all the energy is deposited.

6. Hadrons interact with nuclei after a path drawn from exp(-x / lambda_I)
   (nuclear interaction length), and then make a hadronic shower, drawn
   with the same gamma form in units of lambda_I with
       t_max = 0.2 ln(E / 1 GeV) + 0.7,  95% contained within t_max + 2.5 E^0.13
   (parametrisation of Fabjan & Gianotti 2003, Rev. Mod. Phys. 75, 1243,
   sec. V). All the energy is taken as deposited (in reality part escapes
   as neutrinos, muons and nuclear binding energy). Interactions in the thin
   tracker material are ignored.

The detector's materials are averages over real structures (silicon and
services in the tracker; lead tungstate crystals; brass and scintillator;
the aluminium-stabilised coil; iron yoke with gaps for the muon stations),
chosen to give about the right totals: 0.4 X0 of tracker, 26 X0 and 1.1
lambda_I of electromagnetic calorimeter, ~6 lambda_I of hadron calorimeter,
and ~3.5 GeV lost by a muon on its way to the outermost station.
"""

from dataclasses import dataclass

import numpy as np

M_E = 0.000510999  # GeV
K_BETHE = 0.307075e-3  # GeV cm^2 / mol
BREM_PHOTON_MIN = 0.05  # GeV


@dataclass(frozen=True)
class Material:
    name: str
    density: float  # g/cm^3 (averaged over the region)
    z_over_a: float  # mol/g
    excitation: float  # mean excitation energy I [eV]
    x0: float  # radiation length [cm] (averaged over the region)
    lambda_i: float  # nuclear interaction length [cm] (averaged over the region)
    critical: float = 0.0  # critical energy E_c [GeV], for showers
    # Sternheimer's density-effect parameters of the solid: (Cbar, x0, x1, a, k, delta0)
    density_effect: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# Density-effect parameters (Cbar, x0, x1, a, k, delta0), see the module docstring.
_SILICON = (4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14)
_ALUMINIUM = (4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12)
_IRON = (4.2911, -0.0012, 3.1531, 0.14680, 2.9632, 0.12)
_COPPER = (4.4190, -0.0254, 3.2792, 0.14339, 2.9044, 0.08)
# PbWO4: hbar omega_p = 28.816 sqrt(8.30 * 0.41315) = 53.36 eV, Cbar =
# 2 ln(600.7 / 53.36) + 1 = 5.842, x0 = 0.326 * 5.842 - 1.5 = 0.4045,
# a = (5.842 - 4.6052 * 0.4045) / (3 - 0.4045)^3 = 0.2276.
_LEAD_TUNGSTATE = (5.842, 0.4045, 3.0, 0.2276, 3.0, 0.0)

# Silicon (Z/A 0.49848, I 173 eV) and services spread over the tracker
# volume: 0.36 X0 per metre.
TRACKER = Material("tracker", 0.079, 0.49848, 173.0, 277.0, 1350.0, 0.04, _SILICON)
# Lead tungstate, PbWO4: rho 8.30, X0 0.8903 cm, lambda_I 20.27 cm, E_c 9.64 MeV.
ECAL = Material("ecal", 8.30, 0.41315, 600.7, 0.8903, 20.27, 0.00964, _LEAD_TUNGSTATE)
# Brass (rho 8.53, X0 1.49 cm, lambda_I 16.4 cm) with scintillator: 80% brass.
# Ionisation as in copper (Z/A 0.45636, I 322 eV), which is 70% of the brass.
HCAL = Material("hcal", 6.9, 0.45636, 322.0, 1.86, 19.5, 0.019, _COPPER)
# The coil and its cryostat: aluminium, rho 2.70, X0 8.9 cm, lambda_I 39.7 cm.
COIL = Material("coil", 2.70, 0.48181, 166.0, 8.90, 39.7, 0.043, _ALUMINIUM)
# Iron yoke (rho 7.87, X0 1.757 cm, lambda_I 16.8 cm): the 1.56 m of iron
# between the muon stations of the CMS barrel, spread over the 3.5 m they
# span: 45% of the volume.
YOKE = Material("yoke", 3.54, 0.46557, 286.0, 3.90, 37.3, 0.0218, _IRON)

# Barrel regions: (inner radius, outer radius, material). Endcaps: the same
# sequence along |z| beyond the tracker's end (see region()).
ECAL_DEPTH = 0.23  # m of crystal
REGIONS_R = (
    (0.0, 1.2, TRACKER),
    (1.3, 1.3 + ECAL_DEPTH, ECAL),
    (1.9, 2.9, HCAL),
    (3.0, 3.3, COIL),
    (3.8, 7.3, YOKE),
)
REGIONS_Z = (
    (3.0, 3.0 + ECAL_DEPTH, ECAL),
    (3.6, 4.7, HCAL),
    (4.9, 6.5, YOKE),
)
TRACKER_HALF_LENGTH = 2.8


def region(pos):
    """The material at `pos`, or None in a gap."""
    r, z = np.hypot(pos[0], pos[1]), abs(pos[2])
    if z < TRACKER_HALF_LENGTH + 0.2 or r > 3.0:
        for low, high, material in REGIONS_R:
            if low <= r < high:
                if material is TRACKER and z >= TRACKER_HALF_LENGTH:
                    return None
                return material
        if z < TRACKER_HALF_LENGTH + 0.2:
            return None
    for low, high, material in REGIONS_Z:
        if low <= z < high and r < 3.0:
            return material
    return None


def density_effect(material, beta_gamma):
    """Sternheimer's density-effect correction delta (see the module docstring)."""
    cbar, x0, x1, a, k, delta0 = material.density_effect
    x = np.log10(beta_gamma)
    if x >= x1:
        return 2.0 * np.log(10.0) * x - cbar
    if x >= x0:
        return 2.0 * np.log(10.0) * x - cbar + a * (x1 - x) ** k
    return delta0 * 10.0 ** (2.0 * (x - x0))


def bethe(material, mass, momentum, charge=1):
    """Mean ionisation energy loss [GeV/m] of a heavy charged particle."""
    energy = np.hypot(momentum, mass)
    beta2 = (momentum / energy) ** 2
    bg2 = (momentum / mass) ** 2
    gamma = energy / mass
    t_max = 2.0 * M_E * bg2 / (1.0 + 2.0 * gamma * M_E / mass + (M_E / mass) ** 2)
    excitation = material.excitation * 1.0e-9  # eV -> GeV
    delta = density_effect(material, momentum / mass)
    bracket = 0.5 * np.log(2.0 * M_E * bg2 * t_max / excitation**2) - beta2 - 0.5 * delta
    per_cm = K_BETHE * charge**2 * material.z_over_a * material.density / beta2 * max(bracket, 0.0)
    return per_cm * 100.0


def _highland(t, beta, charge):
    """f(t) = t [1 + 0.038 ln(t z^2 / beta^2)]^2 for t radiation lengths (0 at t = 0).

    Below t = 1e-6 (far outside the formula's range, 1e-3 < t < 100) the
    logarithm is held at its value there, which keeps f increasing.
    """
    if t <= 0.0:
        return 0.0
    correction = 1.0 + 0.038 * np.log(max(t, 1e-6) * charge**2 / beta**2)
    return t * max(correction, 0.0) ** 2


def scattering_angle(material, length, momentum, beta, charge=1):
    """Width theta_0 [rad] of the scattering angle (per plane) after `length` metres
    of `material` crossed in one go (the Highland formula)."""
    t = length * 100.0 / material.x0
    if t <= 0.0 or momentum <= 0.0:
        return 0.0
    return 0.0136 / (beta * momentum) * abs(charge) * np.sqrt(_highland(t, beta, charge))


def scattering_step(before, after, momentum, beta, charge=1):
    """Width theta_0 [rad] of the scattering angle (per plane) added by a step that
    takes the thickness crossed from `before` to `after` radiation lengths
    (see the module docstring)."""
    if after <= before or momentum <= 0.0:
        return 0.0
    gained = _highland(after, beta, charge) - _highland(before, beta, charge)
    return 0.0136 / (beta * momentum) * abs(charge) * np.sqrt(max(gained, 0.0))


def _cross(a, b):
    """a x b for two 3-vectors (np.cross is slow for single vectors, and this is
    called at every step of every charged track)."""
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )


def scatter(momentum_vector, theta0, rng):
    """`momentum_vector` turned by two independent Gaussian angles of width theta0."""
    p = np.linalg.norm(momentum_vector)
    if theta0 <= 0.0 or p == 0.0:
        return momentum_vector
    n = momentum_vector / p
    other = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _cross(n, other)
    u /= np.sqrt(u @ u)
    v = _cross(n, u)
    a, b = rng.normal(0.0, theta0, 2)
    new = n + a * u + b * v
    return p * new / np.linalg.norm(new)


def bremsstrahlung_fraction(material, length, rng):
    """Fraction of its energy an electron keeps after `length` metres (Bethe-Heitler)."""
    t = length * 100.0 / material.x0
    if t <= 0.0:
        return 1.0
    return float(np.exp(-rng.gamma(t / np.log(2.0), 1.0)))


def conversion_length(material, rng):
    """Path [m] after which a photon converts in `material` (mean 9/7 X0)."""
    return rng.exponential(9.0 / 7.0 * material.x0 / 100.0)


def interaction_length(material, rng):
    """Path [m] after which a hadron interacts in `material` (mean lambda_I)."""
    return rng.exponential(material.lambda_i / 100.0)


@dataclass
class Shower:
    """A shower for the display: energy deposited along `direction` from `start`."""

    start: np.ndarray
    direction: np.ndarray
    time: float
    energy: float  # GeV
    kind: str  # "em" or "hadronic"
    depth_max: float  # [m] depth of the maximum
    length: float  # [m] containing ~95% of the energy
    shape: float  # a of the gamma profile
    scale: float  # [m] 1 / b in metres

    def profile(self, n=10):
        """(depths [m], fractions of the energy) in n slices out to `length`."""
        from math import gamma as gamma_function

        edges = np.linspace(0.0, self.length, n + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        t = mid / self.scale
        density = t ** (self.shape - 1.0) * np.exp(-t) / gamma_function(self.shape)
        weights = density / density.sum()
        return mid, weights


def em_shower(material, start, direction, time, energy, is_photon):
    """The electromagnetic shower of an electron or photon of `energy` (PDG eq. 34.36)."""
    b = 0.5
    t_max = max(np.log(max(energy / material.critical, 1.01)) + (0.5 if is_photon else -0.5), 0.3)
    a = b * t_max + 1.0
    x0 = material.x0 / 100.0
    # 95% containment: t_max + 0.08 Z + 9.6 radiation lengths (PDG, sec. 34.5);
    # 0.08 Z = 5.5 for lead tungstate's effective Z of 69.
    length = (t_max + 0.08 * 69.0 + 9.6) * x0
    return Shower(
        np.array(start), np.array(direction), time, energy, "em", t_max * x0, length, a, x0 / b
    )


def hadronic_shower(material, start, direction, time, energy):
    """The hadronic shower of a hadron of `energy` (see the module docstring)."""
    lam = material.lambda_i / 100.0
    t_max = 0.2 * np.log(max(energy, 0.3)) + 0.7
    length = (t_max + 2.5 * max(energy, 0.3) ** 0.13) * lam
    # Gamma profile with its maximum at t_max and 95% within `length`:
    # shape 2 puts the maximum one scale length in.
    return Shower(
        np.array(start),
        np.array(direction),
        time,
        energy,
        "hadronic",
        t_max * lam,
        length,
        2.0,
        t_max * lam,
    )
