"""Radioactive decay: modes, Q-values, kinematics, decay statistics and decay chains.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, atom
module). Review the physics.

Units: MeV (energies, masses, momenta with c = 1), seconds. Four-momenta are
arrays (E, px, py, pz).

Computed from the data (atom_data): which decays a nuclide has and how often
(NUBASE2020), the energy each releases (Q, from the AME2020 atomic masses),
and how that energy is shared:
- two bodies (alpha decay, electron capture, a gamma ray): fixed by energy
  and momentum conservation, recoil of the daughter included, relativistic;
- beta decay (three bodies): the electron's energy is drawn from the
  allowed spectrum p E (Q - T)^2 times the non-relativistic Coulomb (Fermi)
  factor; the neutrino's direction is isotropic and independent of the
  electron's (no electron-neutrino angular correlation), and its energy and
  the daughter's recoil then follow from conservation, exactly.

Approximations, stated:
- Every decay goes to the daughter's ground state (or to the isomer named in
  atom_data.FEEDS_ISOMER). Real decays often feed excited states, which then
  emit gamma rays: those are not in the two data files, so the energies here
  are the largest possible ones (alpha: the highest-energy line; beta: the
  endpoint of the highest-energy branch).
- The Coulomb factor is the non-relativistic one, F = x / (1 - exp(-x)),
  x = +-2 pi alpha Z / beta; the relativistic factor, the screening by the
  atom's electrons, and the shape factors of forbidden decays are left out.
- NUBASE's 'B+' is positron emission and electron capture together. Their
  split is in the file for some nuclides only ('e+' and 'EC' listed
  separately); otherwise positron emission is shown when it is energetically
  possible (Q_EC > 2 m_e) and capture when it is not. For nuclides where
  capture dominates although positrons are possible this is wrong; K-40 is
  corrected by hand (atom_data.CAPTURE_ONLY).
- An isomer's decay to its ground state ('IT') is shown as one gamma ray; in
  reality part of these transitions eject an atomic electron instead
  (internal conversion).
- The binding energies of the atomic electrons are neglected in the
  kinematics (at most 0.1 MeV in the heaviest atoms, of several GeV).

Decay statistics are exact for any length of time step: each nucleus is given
its whole history in advance (successive exponential waiting times and
branch choices), and the populations at a time t are counted from it. The
Bateman solution for the same chain is evaluated for comparison.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from physengine.modules.atom_data import ALPHA, CAPTURE_ONLY, FEEDS_ISOMER, M_ELECTRON, label

LN2 = np.log(2.0)

# What each mode is called in the interface.
MODE_TEXT = {
    "alpha": "alpha decay",
    "beta-": "beta-minus decay",
    "beta+": "beta-plus decay (positron emission)",
    "EC": "electron capture",
    "IT": "gamma decay of an isomer",
    "SF": "spontaneous fission",
    "n": "neutron emission",
    "p": "proton emission",
}
_NUBASE_MODES = {
    "A": "alpha", "B-": "beta-", "B+": "beta+", "e+": "beta+", "EC+B+": "beta+", "EC": "EC",
    "IT": "IT", "SF": "SF", "n": "n", "p": "p",
}  # fmt: skip
_DAUGHTER = {
    "alpha": (-2, -4), "beta-": (1, 0), "beta+": (-1, 0), "EC": (-1, 0), "IT": (0, 0),
    "n": (0, -1), "p": (-1, -1),
}  # fmt: skip


# ----------------------------------------------------------------- kinematics


def isotropic(rng):
    """A unit vector uniformly distributed over the sphere."""
    cos_theta = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def two_body_momentum(M, m1, m2):
    """Momentum of each product when a body of mass M at rest becomes two of
    masses m1 and m2: p* = sqrt((M^2 - (m1 + m2)^2) (M^2 - (m1 - m2)^2)) / (2 M).

    For small energy release the two factors are written with Q = M - m1 - m2
    and without subtracting nearly equal squares, so that keV energies on GeV
    masses keep their precision.
    """
    q = M - m1 - m2
    if q < 0.0:
        raise ValueError(f"decay not allowed: masses {m1} + {m2} exceed {M}")
    first = q * (M + m1 + m2)
    second = (M - m1 + m2) * (M + m1 - m2)
    return np.sqrt(first * second) / (2.0 * M)


def kinetic(p, m):
    """Kinetic energy of a body of mass m and momentum p, without cancellation:
    T = p^2 / (E + m)."""
    return p * p / (np.sqrt(p * p + m * m) + m)


def two_body(M, m1, m2, direction):
    """Four-momenta of the two products of a body of mass M at rest; product 1
    goes along `direction` (a unit vector)."""
    p = two_body_momentum(M, m1, m2)
    direction = np.asarray(direction, dtype=float)
    return (
        np.array([np.sqrt(p * p + m1 * m1), *(p * direction)]),
        np.array([np.sqrt(p * p + m2 * m2), *(-p * direction)]),
    )


def boost(p4, beta):
    """The four-momentum p4 seen from a frame in which its own frame moves with
    velocity `beta` (a 3-vector, in units of c)."""
    beta = np.asarray(beta, dtype=float)
    b2 = float(beta @ beta)
    if b2 == 0.0:
        return np.array(p4, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = float(beta @ p4[1:])
    energy = gamma * (p4[0] + bp)
    momentum = p4[1:] + ((gamma - 1.0) * bp / b2 + gamma * p4[0]) * beta
    return np.array([energy, *momentum])


def invariant_mass(p4):
    return float(np.sqrt(max(p4[0] ** 2 - p4[1:] @ p4[1:], 0.0)))


# -------------------------------------------------------------- beta spectrum


def coulomb_factor(Z_daughter, T, positron=False):
    """Non-relativistic Coulomb (Fermi) factor F = x / (1 - exp(-x)) with
    x = 2 pi alpha Z / beta for electrons and -x for positrons. T [MeV] is
    the kinetic energy of the electron or positron."""
    T = np.asarray(T, dtype=float)
    energy = T + M_ELECTRON
    momentum = np.sqrt(np.maximum(T * (T + 2.0 * M_ELECTRON), 1e-300))
    x = 2.0 * np.pi * ALPHA * Z_daughter * energy / momentum  # > 0
    x = np.maximum(x, 1e-12)
    attract = x / -np.expm1(-x)  # electrons: pulled back by the nucleus
    return attract * np.exp(-x) if positron else attract


def beta_spectrum(T, endpoint, Z_daughter=0, positron=False, coulomb=True):
    """dN/dT (not normalised) of the electron in an allowed beta decay:
    p E (T0 - T)^2 F(Z, T), with T0 = `endpoint` [MeV] and F the Coulomb factor."""
    T = np.asarray(T, dtype=float)
    energy = T + M_ELECTRON
    momentum = np.sqrt(np.maximum(T * (T + 2.0 * M_ELECTRON), 0.0))
    shape = momentum * energy * np.maximum(endpoint - T, 0.0) ** 2
    if coulomb and Z_daughter > 0:
        shape = shape * coulomb_factor(Z_daughter, np.maximum(T, 1e-9 * endpoint), positron)
    return shape


_GRID = 801


@lru_cache(maxsize=64)
def beta_table(endpoint, Z_daughter=0, positron=False, coulomb=True):
    """(T grid, cumulative distribution, density) of the spectrum, for sampling
    and means. Kept for the endpoints used last (do not modify the arrays)."""
    T = np.linspace(0.0, endpoint, _GRID)
    density = beta_spectrum(T, endpoint, Z_daughter, positron, coulomb)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]))])
    return T, cdf / cdf[-1], density


def beta_mean_energy(endpoint, Z_daughter=0, positron=False, coulomb=True):
    """Mean kinetic energy [MeV] of the electron (trapezoid rule on the grid)."""
    T, _, density = beta_table(endpoint, Z_daughter, positron, coulomb)
    return float(np.sum(T * density) / np.sum(density))


def sample_beta_energy(rng, endpoint, Z_daughter=0, positron=False, coulomb=True, size=None):
    """Kinetic energies [MeV] drawn from the spectrum (inverse of the cumulative)."""
    T, cdf, _ = beta_table(endpoint, Z_daughter, positron, coulomb)
    return np.interp(rng.random(size), cdf, T)


# -------------------------------------------------------------------- branches


@dataclass
class Branch:
    """One way a nuclide decays."""

    mode: str  # a key of MODE_TEXT
    fraction: float  # normalised over the branches kept
    daughter: tuple | None  # (Z, A, isomer); None for fission
    q: float  # energy released [MeV], from atomic masses

    @property
    def daughter_name(self):
        return "fission fragments" if self.daughter is None else label(*self.daughter)


def q_value(table, Z, A, isomer, mode, daughter):
    """Energy released [MeV] by the decay, from the atomic masses:
    alpha     M(parent) - M(daughter) - M(He-4)
    beta-     M(parent) - M(daughter)
    beta+     M(parent) - M(daughter) - 2 m_e
    EC        M(parent) - M(daughter)   (less the captured electron's binding, neglected)
    IT        the isomer's excitation energy
    n, p      M(parent) - M(daughter) - M(n) or M(H-1)
    """
    parent = table.mass_excess(Z, A, isomer)
    if daughter is None:
        return float("nan")
    child = table.mass_excess(*daughter)
    released = parent - child
    if mode == "alpha":
        released -= table.mass_excess(2, 4)
    elif mode == "beta+":
        released -= 2.0 * M_ELECTRON
    elif mode == "n":
        released -= table.mass_excess(0, 1)
    elif mode == "p":
        released -= table.mass_excess(1, 1)
    return released


def branches(table, Z, A, isomer="", threshold=0.0):
    """The decays of a nuclide as [Branch], most frequent first.

    Only modes this module follows are kept (alpha, beta, capture, isomer
    gamma, fission, single nucleon emission), and only those at least
    `threshold` of all decays; the fractions are renormalised to sum to 1.
    A stable or unlisted nuclide gives [].
    """
    nuclide = table.get(Z, A, isomer)
    if nuclide is None or nuclide.half_life is None or nuclide.half_life == np.inf:
        return []
    found = []
    listed = {name for name, _ in nuclide.modes}
    for name, fraction in nuclide.modes:
        mode = _NUBASE_MODES.get(name)
        if mode is None:
            continue
        if name == "B+" and listed & {"e+", "EC"}:
            continue  # the sum of the two parts that are listed separately
        if name == "B+" and (Z, A) in CAPTURE_ONLY:
            mode = "EC"
        if mode == "SF":
            found.append(Branch("SF", fraction, None, float("nan")))
            continue
        dz, da = _DAUGHTER[mode]
        zd, ad = Z + dz, A + da
        if zd < 0 or ad < 1 or ad < zd:
            continue
        feeds = [("", 1.0)]
        if (Z, A, name) in FEEDS_ISOMER and not isomer:
            state, share = FEEDS_ISOMER[(Z, A, name)]
            if table.get(zd, ad, state) is not None:
                feeds = [(state, share), ("", 1.0 - share)]
        for state, share in feeds:
            if share <= 0.0:
                continue
            daughter = (zd, ad, state)
            actual = mode
            if mode == "beta+":
                # Positron emission needs 2 m_e; below that only capture is possible.
                if q_value(table, Z, A, isomer, "beta+", daughter) <= 0.0:
                    actual = "EC"
            q = q_value(table, Z, A, isomer, actual, daughter)
            if q <= 0.0:
                continue  # not allowed by the masses used (an estimated mass, or the formula)
            found.append(Branch(actual, fraction * share, daughter, q))
    found = [b for b in found if b.fraction >= threshold]
    total = sum(b.fraction for b in found)
    if total <= 0.0:
        return []
    for b in found:
        b.fraction /= total
    return sorted(found, key=lambda b: -b.fraction)


# ------------------------------------------------------------- a single decay


@dataclass
class Product:
    name: str
    # 'nucleus', 'alpha', 'electron', 'positron', 'neutrino', 'photon', 'neutron' or 'proton'
    kind: str
    mass: float  # MeV
    p4: np.ndarray
    Z: int = 0
    A: int = 0

    @property
    def kinetic(self):
        p = float(np.linalg.norm(self.p4[1:]))
        return float(self.p4[0]) if self.mass == 0.0 else float(kinetic(p, self.mass))


@dataclass
class DecayEvent:
    parent: tuple
    parent_mass: float  # MeV: the mass the products' energies add up to
    branch: Branch
    products: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def decay(table, Z, A, isomer="", rng=None, branch=None, coulomb=True):
    """One decay of the nuclide at rest: a DecayEvent with the products' four-momenta.

    The branch is drawn from branches() unless given. Spontaneous fission is
    not handled here (atom_reactions.fission). Energy and momentum are
    conserved to rounding: the products' four-momenta add up to
    (parent_mass, 0, 0, 0).
    """
    rng = rng or np.random.default_rng()
    if branch is None:
        options = branches(table, Z, A, isomer)
        if not options:
            raise ValueError(f"{label(Z, A, isomer)} has no decay in the data")
        branch = options[rng.choice(len(options), p=[b.fraction for b in options])]
    if branch.mode == "SF":
        raise ValueError("spontaneous fission is generated by atom_reactions.fission")
    zd, ad, state = branch.daughter
    M = table.mass(Z, A, isomer)
    m_daughter_atom = table.mass(zd, ad, state)
    name = label(zd, ad, state)
    event = DecayEvent((Z, A, isomer), M, branch)
    direction = isotropic(rng)

    def nucleus(mass, p4):
        return Product(name, "nucleus", mass, p4, zd, ad)

    if branch.mode == "alpha":
        # Daughter with all Z electrons (a 2- ion) and a bare alpha particle.
        m_alpha = table.mass(2, 4) - 2.0 * M_ELECTRON
        m_d = m_daughter_atom + 2.0 * M_ELECTRON
        pa, pd = two_body(M, m_alpha, m_d, direction)
        event.products = [Product("alpha", "alpha", m_alpha, pa, 2, 4), nucleus(m_d, pd)]
    elif branch.mode in ("n", "p"):
        light = (0, 1) if branch.mode == "n" else (1, 1)
        m_light = table.mass(*light) - light[0] * M_ELECTRON
        m_d = M - branch.q - m_light
        pl, pd = two_body(M, m_light, m_d, direction)
        kind = "neutron" if branch.mode == "n" else "proton"
        event.products = [Product(kind, kind, m_light, pl, light[0], 1), nucleus(m_d, pd)]
    elif branch.mode == "EC":
        pv, pd = two_body(M, 0.0, m_daughter_atom, direction)
        event.products = [Product("neutrino", "neutrino", 0.0, pv), nucleus(m_daughter_atom, pd)]
    elif branch.mode == "IT":
        pg, pd = two_body(M, 0.0, m_daughter_atom, direction)
        event.products = [Product("gamma", "photon", 0.0, pg), nucleus(m_daughter_atom, pd)]
    else:
        positron = branch.mode == "beta+"
        # Daughter ion: Z+1 protons with Z electrons (beta-), or Z-1 with Z (beta+).
        m_d = m_daughter_atom + (M_ELECTRON if positron else -M_ELECTRON)
        # Largest electron energy: the neutrino gets nothing (a two-body decay).
        p_max = two_body_momentum(M, M_ELECTRON, m_d)
        t_max = float(kinetic(p_max, M_ELECTRON))
        T = float(sample_beta_energy(rng, t_max, zd, positron, coulomb))
        T = min(max(T, 0.0), t_max)
        p_e = np.sqrt(T * (T + 2.0 * M_ELECTRON))
        e_e = T + M_ELECTRON
        nu_direction = isotropic(rng)
        cos_angle = float(direction @ nu_direction)
        # Energy conservation with the recoiling daughter, solved for the
        # neutrino's energy: sqrt(m_d^2 + |p_e + p_nu|^2) = W - E_nu, W = M - E_e.
        W = M - e_e
        # W^2 - m_d^2 - p_e^2 without cancellation: (W - m_d)(W + m_d) - p_e^2.
        e_nu = max(((W - m_d) * (W + m_d) - p_e * p_e) / (2.0 * (W + p_e * cos_angle)), 0.0)
        p_electron = np.array([e_e, *(p_e * direction)])
        p_nu = np.array([e_nu, *(e_nu * nu_direction)])
        recoil = -(p_electron[1:] + p_nu[1:])
        p_daughter = np.array([M - e_e - e_nu, *recoil])
        light = "positron" if positron else "electron"
        nu = "neutrino" if positron else "antineutrino"
        event.products = [
            Product(light, light, M_ELECTRON, p_electron),
            Product(nu, "neutrino", 0.0, p_nu),
            nucleus(m_d, p_daughter),
        ]
    return event


# ------------------------------------------------------------ chains and samples


@dataclass
class Species:
    """A member of a decay chain."""

    key: tuple | None  # (Z, A, isomer); None: fission fragments (end of the chain here)
    name: str
    half_life: float  # s; inf if stable or not followed further
    branches: list  # [(index of the daughter species, fraction, mode)]
    note: str = ""
    pending: list = field(default_factory=list, repr=False)  # [Branch] still to link (build_chain)

    @property
    def rate(self):
        return 0.0 if self.half_life == np.inf else LN2 / self.half_life


def build_chain(table, Z, A, isomer="", threshold=1e-3, max_species=60):
    """All nuclides reached from (Z, A) by decays, as [Species], the start first.

    Branches rarer than `threshold` are dropped (and the rest renormalised).
    A nuclide whose decay or half-life is not in the data ends the chain
    there and is marked in its note.
    """
    species, index = [], {}

    def add(key):
        if key in index:
            return index[key]
        index[key] = len(species)
        if key is None:
            species.append(Species(None, "fission fragments", np.inf, [], "not followed"))
            return index[key]
        nuclide = table.get(*key)
        entry = Species(key, label(*key), np.inf, [])
        species.append(entry)
        if nuclide is None:
            entry.note = "not in the data"
        elif nuclide.half_life is None:
            entry.note = "half-life not known"
        elif nuclide.half_life != np.inf:
            options = branches(table, *key, threshold=threshold)
            if not options:
                entry.note = "decay modes not in the data"
            elif len(species) >= max_species:
                entry.note = "chain cut off here"
            else:
                entry.half_life = nuclide.half_life
                entry.pending = options
        return index[key]

    add((Z, A, isomer))
    k = 0
    while k < len(species):
        entry = species[k]
        for b in entry.pending:
            entry.branches.append((add(b.daughter), b.fraction, b.mode))
        entry.pending = []
        if entry.half_life != np.inf and not entry.branches:
            entry.half_life = np.inf
        k += 1
    return species


class DecaySample:
    """N nuclei of one kind and everything they decay into, followed exactly.

    Each nucleus gets its whole history when the sample is made: waiting
    times drawn from the exponential distribution of the nuclide it is at
    that moment (mean = half-life / ln 2), and a branch drawn with the
    branching fractions. Nothing depends on a time step, so the populations
    are exact at any time, whether the half-lives are microseconds or
    billions of years.
    """

    def __init__(self, species, n, rng=None):
        self.species = species
        self.n = int(n)
        rng = rng or np.random.default_rng()
        state = np.zeros(self.n, dtype=np.int32)
        clock = np.zeros(self.n)
        self.states = [state.copy()]  # states[g][i]: species of nucleus i after g decays
        self.times = []  # times[g][i]: when nucleus i makes its (g+1)-th decay (inf: never)
        for _ in range(4 * len(species) + 4):
            when = np.full(self.n, np.inf)
            following = state.copy()
            for s, entry in enumerate(species):
                if not entry.branches:
                    continue
                here = np.flatnonzero((state == s) & np.isfinite(clock))
                if len(here) == 0:
                    continue
                when[here] = clock[here] + rng.exponential(entry.half_life / LN2, len(here))
                fractions = np.array([b[1] for b in entry.branches])
                choice = rng.choice(len(fractions), size=len(here), p=fractions / fractions.sum())
                following[here] = np.array([b[0] for b in entry.branches])[choice]
            if not np.isfinite(when).any():
                break
            self.times.append(when)
            self.states.append(following)
            state, clock = following, when
        self._times = np.array(self.times).T if self.times else np.zeros((self.n, 0))
        self._states = np.array(self.states).T

    def stage(self, t):
        """How many decays each nucleus has made by time t."""
        return np.sum(self._times <= t, axis=1)

    def state_at(self, t):
        """Species index of each nucleus at time t."""
        return self._states[np.arange(self.n), self.stage(t)]

    def populations(self, t):
        """Number of nuclei of each species at time t."""
        return np.bincount(self.state_at(t), minlength=len(self.species))

    def decays_between(self, t0, t1):
        """(nucleus, generation) of the decays that happen in (t0, t1]."""
        return np.nonzero((self._times > t0) & (self._times <= t1))


def decay_paths(species, max_paths=4000):
    """Every route from the first species through the chain, as lists of
    (species index, fraction of that species' decays taking the next step).
    One entry per node of the decay tree: a nuclide reached two ways appears twice."""
    paths = []
    stack = [[(0, 1.0)]]
    while stack and len(paths) < max_paths:
        path = stack.pop()
        paths.append(path)
        last = path[-1][0]
        for daughter, fraction, _ in species[last].branches:
            stack.append([*path[:-1], (last, fraction), (daughter, 1.0)])
    return paths


def bateman(species, t, n0=1.0):
    """Expected number of each species at time(s) t, starting from n0 of the first.

    Bateman's solution (H. Bateman, Proc. Cambridge Philos. Soc. 15, 423,
    1910) for a linear chain with decay constants l_1 ... l_n:

        N_n(t) = n0 (l_1 ... l_(n-1)) sum_j exp(-l_j t) / prod_(k != j) (l_k - l_j),

    applied to every route through the chain (decay_paths), each weighted by
    its branching fractions, and summed per nuclide. Decay constants in one
    route that coincide to better than 1e-9 are moved apart by that much (the
    formula has a removable singularity there).

    The terms are of order n0 or less when the constants are well separated,
    so the result is good to about 1e-13 n0 in absolute terms; a population
    much smaller than that (a late member at very early times) is lost in
    rounding and returned as 0.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    result = np.zeros((len(species), len(t)))
    for path in decay_paths(species):
        rates = np.array([species[s].rate for s, _ in path])
        for a in range(len(rates)):
            for b in range(a):
                if rates[a] > 0.0 and abs(rates[a] - rates[b]) < 1e-9 * rates[a]:
                    rates[a] *= 1.0 + 1e-9 * (a + 1)
        weights = [f for _, f in path[:-1]]
        if len(path) > 1 and (np.prod(weights) == 0.0 or np.any(rates[:-1] == 0.0)):
            continue
        weight = n0 * np.prod(weights)
        total = np.zeros(len(t))
        for j in range(len(rates)):
            # (l_1 ... l_(n-1)) / prod (l_k - l_j), taken as a product of
            # ratios so that it neither overflows nor underflows.
            others = np.delete(rates, j) - rates[j]
            total += np.exp(-rates[j] * t) * np.prod(rates[:-1] / others)
        result[path[-1][0]] += weight * total
    return np.maximum(result, 0.0)


def activity(species, populations):
    """Decays per second of each species: decay constant times number."""
    return np.array([s.rate for s in species]) * np.asarray(populations)
