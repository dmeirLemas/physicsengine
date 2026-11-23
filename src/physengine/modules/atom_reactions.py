"""Fission and fusion: energy release, kinematics, a schematic chain reaction, tunnelling.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, atom
module). Review the physics.

Units: MeV (energies, masses, momenta, c = 1), fm; temperatures in kelvin
(k T in keV where stated). The chain reaction has its own units (below).

FUSION. Computed: the energy each reaction releases (from the AME2020
atomic masses) and how the products share it (energy and momentum
conservation with the reactants at rest: their keV of thermal motion is
neglected against the MeV released). Two products: fixed. Three products
(He-3 + He-3): uniform three-body phase space. p + p -> D + e+ + nu: the
positron's energy from the allowed beta spectrum, as in atom_decay.
Also computed: the Coulomb barrier Z1 Z2 e^2 / (R1 + R2) and the Gamow
tunnelling factor exp(-2 pi eta) = exp(-sqrt(E_G / E)), E_G = 2 mu c^2
(pi alpha Z1 Z2)^2, with the Maxwell-Boltzmann factor exp(-E / kT) and the
Gamow peak E_0 = (E_G (kT)^2 / 4)^(1/3) where their product is largest.
Not included: cross-sections and reaction rates. A rate needs the
astrophysical S-factor S(E) of each reaction (the nuclear part, measured),
and for p + p the weak interaction; none is used here, so the module shows
how the tunnelling probability depends on energy and temperature, not how
often a reaction happens.

FISSION. Computed: the energy released by a given split (from the masses),
the kinematics (four-momentum conserved exactly among fragments, neutrons
and gamma rays), and the energy of the fragments' later beta decays (from
the masses along each isobaric chain). Parametrised from measurements (all
entered from memory, to be checked; see FISSILE): the double-humped mass
yield (one Gaussian for the heavy fragment, plus a rare symmetric
component), the number of prompt neutrons (Terrell's Gaussian), their
energies (Watt spectrum), and the fragments' kinetic energy (Coulomb
repulsion at an effective scission distance tuned to the measured mean).
Charges follow the unchanged-charge-density rule: each fragment keeps the
compound nucleus's ratio Z / A (real fragments differ by about half a
charge unit). Whatever energy is left goes to prompt gamma rays.

CHAIN REACTION. A schematic model, not a reactor or weapon calculation: a
bare sphere of one material, neutrons of one speed moving in straight
lines, each collision either a fission (probability p) or a capture; no
scattering, no energy dependence, no delayed neutrons, no real material
data. Lengths are in mean free paths, times in mean free flight times.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from math import erf, sqrt

import numpy as np

from physengine.modules.atom_data import ALPHA, E2, K_BOLTZMANN_EV, M_ELECTRON, label
from physengine.modules.atom_decay import (
    beta_mean_energy,
    boost,
    invariant_mass,
    isotropic,
    kinetic,
    sample_beta_energy,
    two_body,
    two_body_momentum,
)
from physengine.modules.atom_nucleus import coulomb_barrier

# ============================================================================ fusion


@dataclass(frozen=True)
class Reaction:
    name: str
    reactants: tuple  # ((Z, A), (Z, A))
    products: tuple  # ((Z, A) or a lepton / photon name, ...)
    note: str = ""


PARTICLE_NAMES = {(0, 1): "n", (1, 1): "p", (1, 2): "D", (1, 3): "T"}

REACTIONS = {
    r.name: r
    for r in (
        Reaction("D + T -> He-4 + n", ((1, 2), (1, 3)), ((2, 4), (0, 1)),
                 "The easiest fusion reaction to start; the one planned for power plants."),
        Reaction("D + D -> He-3 + n", ((1, 2), (1, 2)), ((2, 3), (0, 1)),
                 "One of the two D + D branches, about equally likely at low energy."),
        Reaction("D + D -> T + p", ((1, 2), (1, 2)), ((1, 3), (1, 1)),
                 "The other D + D branch."),
        Reaction("D + He-3 -> He-4 + p", ((1, 2), (2, 3)), ((2, 4), (1, 1)),
                 "No neutron; needs a higher temperature (charge 2 on one side)."),
        Reaction("p + p -> D + e+ + nu", ((1, 1), (1, 1)), ((1, 2), "positron", "neutrino"),
                 "Step 1 of the Sun's proton-proton chain. A proton must turn into a neutron"
                 " (weak interaction) during the brief encounter: extremely rare."),
        Reaction("D + p -> He-3 + gamma", ((1, 2), (1, 1)), ((2, 3), "photon"),
                 "Step 2 of the proton-proton chain."),
        Reaction("He-3 + He-3 -> He-4 + 2 p", ((2, 3), (2, 3)), ((2, 4), (1, 1), (1, 1)),
                 "Step 3 of the proton-proton chain (branch pp I, 83% in the Sun)."),
    )
}  # fmt: skip
PP_CHAIN = ("p + p -> D + e+ + nu", "D + p -> He-3 + gamma", "He-3 + He-3 -> He-4 + 2 p")


def particle_name(item):
    if isinstance(item, str):
        return {"positron": "e+", "neutrino": "nu", "photon": "gamma"}[item]
    return PARTICLE_NAMES.get(tuple(item), label(*item))


def _rest_mass(table, item):
    """Mass [MeV] of a reaction partner: bare nuclei (atomic mass less Z electrons)."""
    if item == "positron":
        return M_ELECTRON
    if item in ("neutrino", "photon"):
        return 0.0
    Z, A = item
    return table.mass(Z, A) - Z * M_ELECTRON


def reaction_q(table, reaction):
    """Kinetic energy released [MeV]: masses of the bare reactants less those of the products."""
    before = sum(_rest_mass(table, r) for r in reaction.reactants)
    return before - sum(_rest_mass(table, p) for p in reaction.products)


def annihilation_energy(reaction):
    """Energy [MeV] the positrons add when they meet electrons: 2 m_e each."""
    return 2.0 * M_ELECTRON * sum(1 for p in reaction.products if p == "positron")


def three_body(M, masses, rng):
    """Four-momenta of three products of a body of mass M at rest, distributed
    uniformly in phase space: the mass of the pair (1, 2) is drawn with
    density p*(M; m12, m3) p*(m12; m1, m2) (the standard result of
    integrating the phase space over angles), then two two-body decays."""
    m1, m2, m3 = masses
    low, high = m1 + m2, M - m3
    grid = np.linspace(low, high, 400)[1:-1]
    weight = [two_body_momentum(M, g, m3) * two_body_momentum(g, m1, m2) for g in grid]
    top = 1.05 * max(weight)
    while True:
        m12 = rng.uniform(low, high)
        w = two_body_momentum(M, m12, m3) * two_body_momentum(m12, m1, m2)
        if rng.uniform(0.0, top) <= w:
            break
    pair, third = two_body(M, m12, m3, isotropic(rng))
    a, b = two_body(m12, m1, m2, isotropic(rng))
    velocity = pair[1:] / pair[0]
    return [boost(a, velocity), boost(b, velocity), third]


@dataclass
class ReactionEvent:
    reaction: Reaction
    q: float  # MeV of kinetic energy released
    total_mass: float  # MeV: what the products' energies add up to
    products: list = field(default_factory=list)  # [(name, mass, p4)]

    def kinetic_energies(self):
        return [
            (name, float(p4[0]) if m == 0.0 else float(kinetic(np.linalg.norm(p4[1:]), m)))
            for name, m, p4 in self.products
        ]


def react(table, reaction, rng=None):
    """One reaction with the reactants at rest: the products' four-momenta."""
    rng = rng or np.random.default_rng()
    M = sum(_rest_mass(table, r) for r in reaction.reactants)
    masses = [_rest_mass(table, p) for p in reaction.products]
    names = [particle_name(p) for p in reaction.products]
    q = M - sum(masses)
    if len(masses) == 2:
        momenta = list(two_body(M, masses[0], masses[1], isotropic(rng)))
    elif "neutrino" in reaction.products:
        # Like a beta-plus decay: nucleus + positron + neutrino (atom_decay.decay).
        m_d = masses[0]
        t_max = float(kinetic(two_body_momentum(M, M_ELECTRON, m_d), M_ELECTRON))
        T = float(sample_beta_energy(rng, t_max, reaction.products[0][0], positron=True))
        p_e, e_e = np.sqrt(T * (T + 2.0 * M_ELECTRON)), T + M_ELECTRON
        d_e, d_nu = isotropic(rng), isotropic(rng)
        W = M - e_e
        e_nu = ((W - m_d) * (W + m_d) - p_e * p_e) / (2.0 * (W + p_e * float(d_e @ d_nu)))
        e_nu = max(e_nu, 0.0)
        positron = np.array([e_e, *(p_e * d_e)])
        neutrino = np.array([e_nu, *(e_nu * d_nu)])
        nucleus = np.array([M - e_e - e_nu, *(-(positron[1:] + neutrino[1:]))])
        momenta = [nucleus, positron, neutrino]
    else:
        momenta = three_body(M, masses, rng)
    return ReactionEvent(reaction, q, M, list(zip(names, masses, momenta, strict=True)))


def pp_chain_summary(table):
    """The proton-proton chain (branch I) as a whole: 4 p -> He-4 + 2 e+ + 2 nu.

    Returns a dict: 'steps' [(reaction name, times it runs, Q [MeV], extra
    annihilation energy [MeV])], 'total' (all energy released, annihilation
    included) and 'neutrinos' (the mean energy the two neutrinos carry away,
    from the allowed spectrum of step 1).
    """
    steps = []
    for name, times in zip(PP_CHAIN, (2, 2, 1), strict=True):
        r = REACTIONS[name]
        steps.append((name, times, reaction_q(table, r), annihilation_energy(r)))
    total = sum(times * (q + extra) for _, times, q, extra in steps)
    q_pp = steps[0][2]
    mean_neutrino = q_pp - beta_mean_energy(q_pp, 1, positron=True)
    return {"steps": steps, "total": total, "neutrinos": 2.0 * mean_neutrino}


# ------------------------------------------------------------------ tunnelling


def reduced_mass(table, reaction):
    m1, m2 = (_rest_mass(table, r) for r in reaction.reactants)
    return m1 * m2 / (m1 + m2)


def gamow_energy(Z1, Z2, mu):
    """E_G = 2 mu c^2 (pi alpha Z1 Z2)^2 [MeV], for reduced mass mu [MeV]."""
    return 2.0 * mu * (np.pi * ALPHA * Z1 * Z2) ** 2


def tunnelling(E, e_gamow):
    """Gamow factor exp(-2 pi eta) = exp(-sqrt(E_G / E)): the probability that
    two bare nuclei meeting head-on with centre-of-mass energy E far below
    the barrier tunnel through their Coulomb repulsion (s wave, point
    charges, WKB). E and E_G in the same unit."""
    E = np.asarray(E, dtype=float)
    with np.errstate(divide="ignore"):
        return np.exp(-np.sqrt(e_gamow / np.maximum(E, 1e-300)))


def thermal_energy_kev(temperature):
    """k T [keV] for a temperature in kelvin."""
    return K_BOLTZMANN_EV * temperature * 1e-3


def gamow_peak(e_gamow, kT):
    """(E_0, width): where exp(-E / kT - sqrt(E_G / E)) is largest,
    E_0 = (E_G (kT)^2 / 4)^(1/3), and the 1/e full width 4 sqrt(E_0 kT / 3)
    of the Gaussian that approximates the peak. Same unit as the inputs."""
    e0 = (e_gamow * kT * kT / 4.0) ** (1.0 / 3.0)
    return e0, 4.0 * np.sqrt(e0 * kT / 3.0)


def gamow_window(E, e_gamow, kT):
    """exp(-E / kT) exp(-sqrt(E_G / E)): the energy dependence of the chance
    that a pair from a thermal gas both has energy E and tunnels."""
    E = np.asarray(E, dtype=float)
    return np.exp(-E / kT) * tunnelling(E, e_gamow)


def turning_point(Z1, Z2, E):
    """Distance [fm] at which two nuclei with energy E [MeV] would stop classically."""
    return Z1 * Z2 * E2 / E


def barrier(reaction):
    """Coulomb barrier [MeV] of the reaction's two nuclei, touching."""
    (z1, a1), (z2, a2) = reaction.reactants
    return float(coulomb_barrier(z1, a1, z2, a2))


# =========================================================================== fission

# Measured properties of each fissioning system, entered from memory, to be
# checked (they are typical textbook / evaluated-data values):
#   nu_bar      mean number of prompt neutrons per fission
#   heavy       mean mass number of the heavy fragment before neutron emission
#   sigma       standard deviation of the heavy (and light) mass peak
#   watt        (a [MeV], b [1/MeV]) of the Watt spectrum exp(-E/a) sinh(sqrt(b E))
#   tke         measured mean total kinetic energy of the fragments [MeV], for the test
FISSILE = {
    "U-235": {"target": (92, 235), "induced": True, "nu_bar": 2.43, "heavy": 139.5,
              "sigma": 6.0, "watt": (0.988, 2.249), "tke": 170.5},
    "Pu-239": {"target": (94, 239), "induced": True, "nu_bar": 2.88, "heavy": 139.5,
               "sigma": 6.4, "watt": (0.966, 2.842), "tke": 177.9},
    "Cf-252": {"target": (98, 252), "induced": False, "nu_bar": 3.76, "heavy": 143.5,
               "sigma": 6.8, "watt": (1.025, 2.926), "tke": 184.1},
}  # fmt: skip
SYMMETRIC_SHARE = 0.0005  # of fissions: the rare symmetric splits (valley about 0.01% per mass)
SYMMETRIC_SIGMA = 4.0
TERRELL_SIGMA = 1.08  # width of the neutron-number distribution (Terrell, Phys. Rev. 108, 783)
# Effective radius constant of the two fragments at scission [fm]: the
# fragments' kinetic energy is Z1 Z2 e^2 / (r (A1^(1/3) + A2^(1/3))). With
# r = 1.79 fm (against 1.2 for nuclei at rest: the fragments are stretched
# and apart) the estimate averages 168 MeV over U-235's mass yield; the
# measured mean is about 170 MeV.
SCISSION_RADIUS = 1.79
GAMMA_ENERGY = 1.0  # MeV: prompt gamma rays are emitted with about this energy each


def generic_system(Z, A):
    """Parameters for the spontaneous fission of a nuclide that FISSILE does
    not list: rough values (heavy peak near 140, 2.5 neutrons)."""
    return {"target": (Z, A), "induced": False, "nu_bar": 2.5, "heavy": min(140.0, 0.58 * A),
            "sigma": 6.5, "watt": (1.0, 2.5), "tke": float("nan")}  # fmt: skip


def mass_yield(system, masses=None):
    """Yield of each fragment mass number before neutron emission, summing to
    2 (two fragments per fission): (A array, yield array).

    The parametrisation: the heavy fragment's mass is Gaussian about
    `heavy` with width `sigma`; the light fragment is the rest; a share
    SYMMETRIC_SHARE of fissions is symmetric (Gaussian about half the mass).
    """
    Z, A = system["target"]
    total = A + (1 if system["induced"] else 0)
    if masses is None:
        masses = np.arange(60, 181)
    masses = np.asarray(masses, dtype=float)

    def gauss(x, centre, sigma):
        return np.exp(-0.5 * ((x - centre) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    heavy, sigma = system["heavy"], system["sigma"]
    asymmetric = gauss(masses, heavy, sigma) + gauss(masses, total - heavy, sigma)
    symmetric = 2.0 * gauss(masses, 0.5 * total, SYMMETRIC_SIGMA)
    return masses, (1.0 - SYMMETRIC_SHARE) * asymmetric + SYMMETRIC_SHARE * symmetric


@lru_cache(maxsize=16)
def multiplicity_distribution(nu_bar, sigma=TERRELL_SIGMA, n_max=10):
    """P(0..n_max prompt neutrons): Terrell's rule, a Gaussian of width
    `sigma` cut into unit bins (everything below 0.5 counts as 0), shifted
    so that the mean is exactly nu_bar."""

    def probabilities(centre):
        edges = (np.arange(n_max + 2) - 0.5 - centre) / sigma
        cumulative = np.array([0.5 * (1.0 + erf(e / sqrt(2.0))) for e in edges])
        cumulative[0], cumulative[-1] = 0.0, 1.0
        return np.diff(cumulative)

    low, high = nu_bar - 1.0, nu_bar + 1.0
    for _ in range(60):
        centre = 0.5 * (low + high)
        if probabilities(centre) @ np.arange(n_max + 1) < nu_bar:
            low = centre
        else:
            high = centre
    result = probabilities(0.5 * (low + high))
    result.setflags(write=False)  # shared by every caller (cached)
    return result


def sample_watt(rng, a, b, size=None):
    """Neutron energies [MeV] from the Watt spectrum exp(-E / a) sinh(sqrt(b E)).

    Exact construction: a Maxwellian of temperature a in a frame moving with
    kinetic energy per nucleon E_f = a^2 b / 4, seen from the laboratory:
    E = E' + E_f + 2 sqrt(E' E_f) cos(angle), cos uniform. Mean 3a/2 + E_f.
    """
    n = 1 if size is None else int(size)
    r1, r2, r3 = 1.0 - rng.random(n), 1.0 - rng.random(n), rng.random(n)
    maxwell = a * (-np.log(r1) - np.log(r2) * np.cos(0.5 * np.pi * r3) ** 2)
    e_f = 0.25 * a * a * b
    energy = maxwell + e_f + 2.0 * np.sqrt(maxwell * e_f) * rng.uniform(-1.0, 1.0, n)
    return float(energy[0]) if size is None else energy


def beta_chain_energy(table, Z, A):
    """Energy [MeV] a fragment releases by beta-minus decays until it is
    stable (shared between electrons, antineutrinos and gamma rays), and the
    stable nuclide it ends as: (energy, Z_final). From the masses."""
    released, z = 0.0, Z
    while z < A:
        step = table.mass_excess(z, A) - table.mass_excess(z + 1, A)
        if step <= 0.0:
            break
        released += step
        z += 1
    return released, z


@dataclass
class FissionEvent:
    system: str
    induced: bool
    compound: tuple  # (Z, A) of the nucleus that splits
    initial_mass: float  # MeV: what the products' energies add up to
    fragments: list = field(default_factory=list)  # [(Z, A after neutrons, mass, p4)] light first
    primary_masses: tuple = ()  # mass numbers of the two fragments before neutron emission
    neutrons: list = field(default_factory=list)  # [p4]
    gammas: list = field(default_factory=list)  # [p4]
    q: float = 0.0  # MeV released promptly (fragments + neutrons + gamma rays)
    later: float = 0.0  # MeV released afterwards by the fragments' beta decays
    final: tuple = ()  # the stable nuclides the fragments end as [(Z, A)]
    estimated_masses: bool = False  # a fragment mass is not a measured one
    neutron_mass: float = 0.0

    @property
    def fragment_energy(self):
        return sum(float(kinetic(np.linalg.norm(p4[1:]), m)) for _, _, m, p4 in self.fragments)

    @property
    def neutron_energy(self):
        return sum(float(p4[0]) for p4 in self.neutrons) - len(self.neutrons) * self.neutron_mass

    @property
    def gamma_energy(self):
        return sum(float(p4[0]) for p4 in self.gammas)


def fission(table, system_name, rng=None, system=None):
    """One fission: fragments, prompt neutrons and gamma rays with their four-momenta.

    `system_name` is a key of FISSILE (thermal-neutron fission of U-235 or
    Pu-239: the neutron's own 0.025 eV is neglected; spontaneous fission of
    Cf-252), or any name with `system` given (generic_system).

    Steps: heavy-fragment mass from the yield; charges by unchanged charge
    density; number of neutrons from Terrell's distribution, each taken
    from either fragment with equal chance; neutron energies from the Watt
    spectrum, directions isotropic in the laboratory; fragment kinetic
    energy from the Coulomb estimate (reduced if the masses do not leave
    enough); the rest of the energy to gamma rays of about 1 MeV. The two
    fragments then recoil against everything else, so that energy and
    momentum are conserved exactly. A split the masses do not allow is
    drawn again.
    """
    rng = rng or np.random.default_rng()
    system = system or FISSILE[system_name]
    Zt, At = system["target"]
    induced = system["induced"]
    Zc, Ac = Zt, At + (1 if induced else 0)
    m_n = table.mass(0, 1)
    initial = table.mass(Zt, At) + (m_n if induced else 0.0)
    probabilities = multiplicity_distribution(system["nu_bar"])
    for _ in range(200):
        if rng.random() < SYMMETRIC_SHARE:
            heavy = rng.normal(0.5 * Ac, SYMMETRIC_SIGMA)
        else:
            heavy = rng.normal(system["heavy"], system["sigma"])
        a_heavy = int(round(max(heavy, Ac - heavy)))
        a_light = Ac - a_heavy
        if a_light < 30:
            continue
        z_heavy = int(round(Zc * a_heavy / Ac))
        z_light = Zc - z_heavy
        nu = int(rng.choice(len(probabilities), p=probabilities))
        from_heavy = int(rng.binomial(nu, 0.5))
        ah, al = a_heavy - from_heavy, a_light - (nu - from_heavy)
        if al <= z_light or ah <= z_heavy:
            continue
        m_light, m_heavy = table.mass(z_light, al), table.mass(z_heavy, ah)
        q = initial - m_light - m_heavy - nu * m_n
        neutron_kinetic = sample_watt(rng, *system["watt"], size=nu) if nu else np.zeros(0)
        coulomb = (
            z_light
            * z_heavy
            * E2
            / (SCISSION_RADIUS * (a_light ** (1.0 / 3.0) + a_heavy ** (1.0 / 3.0)))
        )
        available = q - float(neutron_kinetic.sum())
        if available <= 1.0:
            continue  # the masses leave nothing for the fragments: not a possible split
        tke = min(coulomb, available)
        gamma_total = available - tke
        break
    else:
        raise RuntimeError("no fission split allowed by the masses was found")

    neutrons = []
    for T in neutron_kinetic:
        p = np.sqrt(T * (T + 2.0 * m_n))
        neutrons.append(np.array([T + m_n, *(p * isotropic(rng))]))
    count = int(round(gamma_total / GAMMA_ENERGY))
    gammas = []
    if gamma_total > 0.0:
        count = max(count, 1)
        shares = rng.dirichlet(np.full(count, 4.0)) * gamma_total
        gammas = [np.array([e, *(e * isotropic(rng))]) for e in shares]
    # The two fragments together carry what is left of the four-momentum.
    rest = np.array([initial, 0.0, 0.0, 0.0]) - sum(neutrons) - sum(gammas)
    W = invariant_mass(rest)
    light, heavy_p4 = two_body(W, m_light, m_heavy, isotropic(rng))
    velocity = rest[1:] / rest[0]
    light, heavy_p4 = boost(light, velocity), boost(heavy_p4, velocity)

    later_l, final_l = beta_chain_energy(table, z_light, al)
    later_h, final_h = beta_chain_energy(table, z_heavy, ah)
    estimated = any(
        not table.measured(z, a) or table.masses[(z, a)][1]
        for z, a in ((z_light, al), (z_heavy, ah))
    )
    return FissionEvent(
        system=system_name,
        induced=induced,
        compound=(Zc, Ac),
        initial_mass=initial,
        fragments=[(z_light, al, m_light, light), (z_heavy, ah, m_heavy, heavy_p4)],
        primary_masses=(a_light, a_heavy),
        neutrons=neutrons,
        gammas=gammas,
        q=q,
        later=later_l + later_h,
        final=((final_l, al), (final_h, ah)),
        estimated_masses=estimated,
        neutron_mass=m_n,
    )


# ===================================================================== chain reaction


def escape_probability(R):
    """Probability that a neutron born at a uniformly random place in a sphere
    of radius R (in mean free paths), with a random direction, leaves without
    a collision: P = 3 / (8 R^3) (2 R^2 - 1 + (1 + 2 R) exp(-2 R))
    (Case, de Hoffmann and Placzek, 'Introduction to the Theory of Neutron
    Diffusion', 1953; entered from memory, checked against this module's own
    transport in test_atom.py)."""
    R = np.asarray(R, dtype=float)
    return 3.0 / (8.0 * R**3) * (2.0 * R**2 - 1.0 + (1.0 + 2.0 * R) * np.exp(-2.0 * R))


class ChainReaction:
    """Neutrons multiplying in a bare sphere (the schematic model of the module docstring).

    Units: mean free path (length) and mean free flight time (time); the
    neutrons' speed is 1. Each neutron flies straight for a distance drawn
    from the exponential distribution with mean 1; if it is still inside the
    sphere of radius `R` it collides: with probability `p_fission` it causes
    a fission, which emits a number of new neutrons drawn from
    `multiplicity` (isotropic directions) and removes the old one; otherwise
    it is captured. A neutron that reaches the surface is lost.

    In an infinite medium each neutron would be followed by
    k_inf = nu_bar p_fission others; in the sphere by k = k_inf times the
    chance of not leaking. k > 1: the number grows (supercritical); k < 1:
    it dies out (subcritical).
    """

    def __init__(self, R, p_fission, multiplicity, rng=None, max_neutrons=20000):
        self.R = float(R)
        self.p_fission = float(p_fission)
        self.multiplicity = np.asarray(multiplicity, dtype=float)
        self.nu_bar = float(self.multiplicity @ np.arange(len(self.multiplicity)))
        self.rng = rng or np.random.default_rng()
        self.max_neutrons = int(max_neutrons)
        self.pos = np.zeros((0, 3))
        self.dir = np.zeros((0, 3))
        self.path = np.zeros(0)  # distance still to fly before the next collision
        self.generation = np.zeros(0, dtype=np.int64)
        self.time = 0.0
        self.saturated = False
        # Tallies.
        self.fissions = 0
        self.captures = 0
        self.leaks = 0
        self.produced = 0  # neutrons born in fissions
        self.born = {}  # generation -> neutrons born into it
        self.recent_fissions = []  # positions of the fissions of the last step (display)
        self.history = [(0.0, 0)]  # (time, neutrons in flight)

    @property
    def k_infinity(self):
        return self.nu_bar * self.p_fission

    @property
    def count(self):
        return len(self.pos)

    def measured_k(self):
        """Neutrons produced per neutron lost so far (None until 20 are lost)."""
        ended = self.fissions + self.captures + self.leaks
        return self.produced / ended if ended >= 20 else None

    def generation_ratios(self):
        """[(generation, neutrons born into it, ratio to the generation before)]."""
        rows = []
        for g in sorted(self.born):
            before = self.born.get(g - 1)
            rows.append((g, self.born[g], self.born[g] / before if before else None))
        return rows

    def inject(self, n, at_centre=False):
        """Add n neutrons: at the centre, or at uniformly random places in the sphere."""
        if at_centre:
            pos = np.zeros((n, 3))
        else:
            pos = self._directions(n) * (self.R * self.rng.random(n) ** (1.0 / 3.0))[:, None]
        self._add(pos, np.zeros(n, dtype=np.int64))

    def _directions(self, n):
        d = self.rng.normal(size=(n, 3))
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def _add(self, pos, generation):
        n = len(pos)
        if n == 0:
            return
        self.pos = np.vstack([self.pos, pos])
        self.dir = np.vstack([self.dir, self._directions(n)])
        self.path = np.concatenate([self.path, self.rng.exponential(1.0, n)])
        self.generation = np.concatenate([self.generation, generation])
        for g, c in zip(*np.unique(generation, return_counts=True), strict=True):
            self.born[int(g)] = self.born.get(int(g), 0) + int(c)

    def step(self, dt):
        """Advance by dt (in mean free flight times). Exact for any dt: flights,
        collisions and leaks are followed in order within the step."""
        self.recent_fissions = []
        if self.saturated:
            return
        left = np.full(self.count, float(dt))  # time each neutron still has in this step
        active = np.arange(self.count)
        alive = np.ones(self.count, dtype=bool)
        while len(active) and self.count <= 2 * self.max_neutrons:
            pos, direction, budget = self.pos[active], self.dir[active], left[active]
            # Distance to the surface along the flight.
            b = np.sum(pos * direction, axis=1)
            c = np.sum(pos * pos, axis=1) - self.R**2
            to_surface = -b + np.sqrt(np.maximum(b * b - c, 0.0))
            to_collision = self.path[active]
            travel = np.minimum(np.minimum(to_collision, to_surface), budget)
            self.pos[active] = pos + travel[:, None] * direction
            self.path[active] -= travel
            left[active] -= travel
            leaked = (to_surface <= to_collision) & (to_surface <= budget)
            collided = ~leaked & (to_collision <= budget)
            self.leaks += int(leaked.sum())
            alive[active[leaked]] = False
            hit = active[collided]
            alive[hit] = False
            active = np.zeros(0, dtype=np.int64)
            if len(hit) == 0:
                break
            fissioning = self.rng.random(len(hit)) < self.p_fission
            self.captures += int((~fissioning).sum())
            sites = hit[fissioning]
            self.fissions += len(sites)
            if len(sites) == 0:
                break
            for site in sites[:300]:
                self.recent_fissions.append(self.pos[site].copy())
            numbers = self.rng.choice(len(self.multiplicity), size=len(sites), p=self.multiplicity)
            parents = np.repeat(sites, numbers)
            if len(parents):
                first = self.count
                self.produced += len(parents)
                self._add(self.pos[parents], self.generation[parents] + 1)
                left = np.concatenate([left, left[parents]])
                alive = np.concatenate([alive, np.ones(len(parents), dtype=bool)])
                active = np.arange(first, self.count)
        self.pos, self.dir = self.pos[alive], self.dir[alive]
        self.path, self.generation = self.path[alive], self.generation[alive]
        self.time += dt
        self.history.append((self.time, self.count))
        if self.count > self.max_neutrons:
            self.saturated = True


def expected_tke(system):
    """Mean of the Coulomb estimate of the fragments' kinetic energy over the
    mass yield [MeV] (before the limit set by the masses)."""
    Z, A = system["target"]
    total = A + (1 if system["induced"] else 0)
    masses, weights = mass_yield(system, np.arange(total // 2, total - 30))
    light = total - masses
    z_heavy = np.round(Z * masses / total)
    energy = (
        z_heavy
        * (Z - z_heavy)
        * E2
        / (SCISSION_RADIUS * (masses ** (1.0 / 3.0) + light ** (1.0 / 3.0)))
    )
    return float(np.sum(weights * energy) / np.sum(weights))
