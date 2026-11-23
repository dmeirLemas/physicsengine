"""The "Custom" scene of the particle-physics module: particles the user places.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, module
4). Review the physics. No GL and no ImGui here: the window that edits a
setup is particle_editor.py.

A custom event is a list of particles put into the detector by hand, each
with a momentum, a direction and a starting point the user chose. Nothing in
it says what a proton-proton collision would produce, how often, or with what
momenta (see the module docstring of particle_events.py); the texts shown
with a custom event say so. Placed particles pass through each other, except
for a pair of entries set to collide (see "Collisions" below).

Each entry of a setup does one of two things.

1. "fire": the particle starts with the given momentum and is followed like
   any other (particle_events._produce): path in the field, energy loss,
   scattering, showers, and decay in flight after a proper time drawn from
   exp(-t / tau), with a mode drawn from the PDG fractions.

2. "decay": the particle is a parent. Its first decay is the one the user
   chose: a listed mode, or a mode drawn from the listed fractions, or
   daughters typed by hand. Where it decays:
   - a resonance (Z, J/psi, Higgs) decays at once where it is put, with a
     mass drawn from its Breit-Wigner line shape
     (particle_kinematics.breit_wigner_mass);
   - any other parent flies for a time gamma t, t drawn from exp(-t / tau),
     as in every scene, so on average a distance beta gamma c tau
     = (p / m) c tau; it may shower or leave before that, and then there is
     no decay. Or, if the user asks, it decays at its starting point (forced:
     useful for particles that would usually leave, such as charged pions
     and kaons with metres of c tau, or the muon with 659 m);
   - a parent at rest is shown decaying at once (its mean life is not waited
     for in the playback).
   The daughters' momenta are those of particle_events.decay_momenta in the
   parent's rest frame (energy and momentum conserved exactly; phase space
   where no matrix element is implemented), boosted to the lab. A Z placed
   by hand decays isotropically: the 1 + cos^2(theta) of the other scenes is
   about the beam axis of the collision that made the Z, and here there is
   none. The later decays of the chain are drawn from the PDG fractions.

Modes offered for a parent (modes_of): the table of particle_data.py (with
its stand-in mode), plus EXTRA_MODES for parents whose table there is empty.
For the parents with partial tables (the D and B mesons, the J/psi and the
Z; see particle_data.py) and the Higgs only some modes are listed (decays to
D*, eta, phi, quarks, gluons, neutrinos and W bosons have no particles in
the table), so "random" draws among the listed ones in proportion to their
fractions, renormalised; the window says which share of all decays that is
(INCOMPLETE, listed_share). Typed daughters are checked
for energy (the masses must fit in the parent's mass) and charge only;
lepton number, baryon number and the like are not checked.

Kinematics of an entry (kinematics): with m the mass and c = 1,
    E = sqrt(p^2 + m^2),  T = E - m,  beta = p / E,  gamma = E / m,
    direction (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)), theta from the beam (z) axis,
    pseudorapidity eta = -ln tan(theta / 2), i.e. theta = 2 atan(exp(-eta)),
    mean decay length beta gamma c tau = (p / m) c tau,
    radius of curvature R = p_T / (0.2998 |q| B)  (m, GeV, T; particle_tracks.py).
The "gun with smearing": the momentum of each particle is drawn from a
Gaussian of the given relative width about the set value, and its direction
is turned by two independent Gaussian angles of the given width (a model of
a spread, nothing more).

Measured mass of a parent (_measure): the invariant mass of everything
visible that came from it, each particle with its own mass: all descendants
that did not decay, except neutrinos (particle_events.visible_tracks, within
the parent's family). Charged ones as the model tracker measures them
(particle_events.measured: true momentum smeared by the resolution),
electrons from the calorimeter (lepton_measured), photons with their true
direction and an energy smeared by the same calorimeter resolution, neutral
hadrons with their true momenta. A charged particle that decays in flight
(a K or pi from a D or B) is measured itself, over its path from its start
to the kink, if that segment has the layers for it (measured_to_kink); what
it decayed into is then left out. If the chain has neutrinos this is a
"visible mass", below the parent's. If a charged particle cannot be measured
(it starts outside the tracker or too close to the beam line) the event
gives no mass. The histogram (histogram_for) is that of the first "decay"
entry: centred on the parent's mass, +- max(8 widths, 15% of the mass), or
from 0 to 1.2 masses when most decays have neutrinos. A setup that only
fires particles has no histogram: there is no parent whose mass would be
measured.

Collisions (written 2026-11-23 under the owner's override for this module;
review the physics). Two "fire" entries may be set to collide (Entry.collide,
the same non-empty name on both). Only a pair that can form a single
resonance of the table is offered (FORMATION): e+ e- -> Z and mu+ mu- -> Z.
Not included: e+ e- -> J/psi (and other e+ e- resonances): no sourced
formation coupling is used here; tau+ tau- -> Z: not asked for. Nothing else
interacts: all other particles pass through each other.
- The paths must meet (meeting): the two start at one point, or they start
  on one line parallel to the beam (z) axis and fly at each other along it
  ("Face each other" gives this for two starts on the axis). Along the axis
  the solenoid does not bend them; off the axis it would, so such a pair is
  refused. Energy loss and scattering on the way to the meeting point are
  left out. Each count must be 1 and the direction spread 0 (else they miss).
- sqrt(s): s = (p1 + p2)^2 = (E1 + E2)^2 - |p1 + p2|^2, with the two
  four-momenta as fired (PDG 2024, review "Kinematics": the invariant mass
  of a system; head on with equal energies E, sqrt(s) = 2 E up to the
  lepton masses).
- Formation. No resonance is formed unless |sqrt(s) - M| <= 10 widths
  (WINDOW_WIDTHS). Within that window it is formed with probability
      w(sqrt(s)) = (Gamma/2)^2 / ((sqrt(s) - M)^2 + (Gamma/2)^2)
  (particle_kinematics.breit_wigner_formation; PDG 2024, review "Resonances",
  non-relativistic Breit-Wigner; Z boson: M = 91.1876 GeV, Gamma = 2.4952 GeV
  from particle_data.py). So for pairs with a spread of sqrt(s) (a momentum
  spread), the sqrt(s) of the pairs that form a Z follows the Breit-Wigner
  times that spread. STATED APPROXIMATION: the absolute rate is not computed
  from a cross section. A pair exactly at the peak always forms the Z
  (probability 1), which a real pair would do only rarely (it would mostly
  pass by: the peak cross section is a few tens of nanobarns).
- The resonance's four-momentum is p1 + p2 exactly, so its mass is sqrt(s).
  The Breit-Wigner gives how likely each sqrt(s) is to form it, not a mass
  drawn apart from the pair (that would break energy-momentum conservation).
- It is made where and when the two meet (straight paths, speeds beta c)
  and decays at once there, with a mode drawn from the listed PDG fractions
  (for the Z: e+ e-, mu+ mu-, tau+ tau-; Z -> nu nu and Z -> hadrons are not
  in the table). STATED APPROXIMATION: the decay is isotropic in its rest
  frame, as for a Z placed by hand: the 1 + cos^2(theta) about the
  direction of the incoming pair (and its forward-backward asymmetry) is
  left out. The two fired particles are removed (absorbed); the decay
  products are followed through the detector as usual.
- Not formed (outside the window, or not drawn): the two are fired as
  without the option, and pass through each other.
Each collision is recorded in Event.collisions with its sqrt(s).

Limits, so that no setup can stall the module: at most MAX_ENTRIES entries,
MAX_PLACED particles in all, momenta up to P_MAX; an event stops taking on
more of the placed particles after TIME_BUDGET seconds or MAX_TRACKS tracks,
and says so.
"""

import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np

from physengine.modules.particle_data import PARTICLES
from physengine.modules.particle_events import (
    ECAL_A,
    ECAL_C,
    Event,
    Hist,
    _produce,
    decay_momenta,
    lepton_measured,
    lever_arm,
    measured,
    momentum_resolution,
    z_pair_masses,
)
from physengine.modules.particle_kinematics import (
    breit_wigner_formation,
    breit_wigner_mass,
    four_momentum,
    mass,
    two_body_decay,
)
from physengine.modules.particle_material import TRACKER_HALF_LENGTH, scatter
from physengine.modules.particle_tracks import (
    OUTER_RADIUS,
    OUTER_Z,
    TRACKER_RADIUS,
    C,
    Track,
    helix_radius,
    propagate,
)

SCENE = "Custom"
FORMAT = 1  # of the dictionary made by to_dict

MAX_ENTRIES = 20
MAX_COUNT = 100  # repeats of one entry
MAX_PLACED = 300  # particles placed in all
MAX_DAUGHTERS = 6  # of a typed decay (as many as the largest listed mode)
MAX_TRACKS = 4000  # an event stops taking on placed particles beyond this
TIME_BUDGET = 1.5  # s, likewise
P_MAX = 7000.0  # GeV: the LHC's beam energy; nothing above it is offered
AT_REST = 1.0e-6  # GeV: a massive particle with less momentum than this is at rest
ETA_MAX = 10.0
SPREAD_P_MAX = 50.0  # percent
SPREAD_ANGLE_MAX = 30.0  # degrees
LONG_FLIGHT = 1.0  # m: mean decay lengths above this get a warning
WINDOW_WIDTHS = 10.0  # a pair forms a resonance only within this many widths of its mass

# Pairs of fired particles that can collide, and the single resonance they
# form (see "Collisions" in the module docstring). The Z couples to a lepton
# and its own antilepton only, so e+ mu- forms nothing.
FORMATION = {
    frozenset(("e+", "e-")): "Z",
    frozenset(("mu+", "mu-")): "Z",
}

ACTIONS = ("fire", "decay")
QUANTITIES = ("momentum", "energy", "kinetic")
ANGLES = ("theta", "eta")
WHERE = ("lifetime", "start")

# Decays of parents whose table in particle_data.py is empty.
# - Muon: e nu nu, practically 100% (PDG). (The other scenes leave the muon
#   stable: with c tau = 659 m it does not decay inside the detector.)
# - Higgs boson, 125 GeV: Standard Model predictions of the LHC Higgs Cross
#   Section Working Group (CERN Yellow Report 4, CERN-2017-002): tau tau
#   6.27%, gamma gamma 0.227%, mu mu 0.0218%, and Z Z* -> four charged
#   leptons (electrons or muons) 0.0124%. The last is written ("Z", "Z") and
#   made as in the "Higgs -> 4 leptons" scene (particle_events.z_pair_masses:
#   one Z is far off its mass shell).
HIGGS_ZZ = ("Z", "Z")
EXTRA_MODES = {
    "mu-": ((1.0, ("e-", "nu", "nu")),),
    "mu+": ((1.0, ("e+", "nu", "nu")),),
    "H": (
        (0.0627, ("tau+", "tau-")),
        (0.00227, ("gamma", "gamma")),
        (0.000218, ("mu+", "mu-")),
        (0.000124, HIGGS_ZZ),
    ),
}
# Parents with only some of their modes listed: the partial tables of
# particle_data.py (D and B mesons, J/psi, Z) and the Higgs above.
INCOMPLETE = tuple(name for name, p in PARTICLES.items() if p.partial) + ("H",)


@dataclass
class Entry:
    """One line of a custom setup. Units: GeV, degrees, metres."""

    action: str = "fire"  # "fire" or "decay"
    name: str = "mu-"
    quantity: str = "momentum"  # what `value` is: "momentum", "energy" (total) or "kinetic"
    value: float = 10.0  # GeV
    angles: str = "theta"  # the direction is given by "theta" or by "eta" (and phi)
    theta: float = 90.0  # polar angle from the beam (z) axis
    eta: float = 0.0  # pseudorapidity
    phi: float = 0.0  # azimuth
    start: tuple = (0.0, 0.0, 0.0)
    count: int = 1  # how many of it
    spread_p: float = 0.0  # percent: Gaussian spread of the momentum
    spread_angle: float = 0.0  # degrees: Gaussian spread of the direction
    # "decay" only:
    daughters: str = ""  # the mode, e.g. "pi+ pi-"; empty: drawn from the listed fractions
    typed: bool = False  # the daughters were typed by hand (not picked from the list)
    where: str = "lifetime"  # decays after its "lifetime", or at the "start" point
    # "fire" only: name of the collision this entry takes part in; the two
    # entries with the same non-empty name collide ("" : it collides with nothing).
    collide: str = ""


@dataclass
class Setup:
    entries: list = field(default_factory=list)


@dataclass(frozen=True)
class Kinematics:
    momentum: float  # GeV
    energy: float
    kinetic: float
    beta: float
    gamma: float  # inf for a massless particle
    pt: float
    theta: float  # degrees
    eta: float  # +-inf along the beam
    direction: tuple
    decay_length: float | None  # beta gamma c tau [m]; None: does not decay here
    radius: float | None  # of curvature [m]; None: not bent


# ------------------------------------------------------------------ modes


def modes_of(name):
    """((branching fraction, daughter names), ...) that `name` can be made to decay
    into: its table (with the stand-in mode last), else EXTRA_MODES. Empty
    for a particle that does not decay here."""
    particle = PARTICLES.get(name)
    if particle is None:
        return ()
    if particle.decays:
        listed = sum(b for b, _ in particle.decays)
        rest = ((1.0 - listed, particle.rest),) if particle.rest and listed < 1.0 else ()
        return tuple(particle.decays) + rest
    return EXTRA_MODES.get(name, ())


def listed_share(name):
    """Sum of the fractions modes_of lists: below one for the INCOMPLETE parents."""
    return float(sum(b for b, _ in modes_of(name)))


def parents():
    """Names of the particles that can be a parent, in the table's order."""
    return [name for name in PARTICLES if modes_of(name)]


def parse_daughters(text):
    """Daughter names from a typed string ("pi+ pi-", commas allowed)."""
    return tuple(str(text).replace(",", " ").split())


def _is_listed(name, daughters):
    return any(tuple(d) == tuple(daughters) for _, d in modes_of(name))


def _mode_problem(name, daughters):
    """Why `name` cannot decay to `daughters`, or "" if it can."""
    parent = PARTICLES[name]
    if _is_listed(name, daughters):
        return ""  # the table's own modes (H -> Z Z* is below threshold on purpose)
    if len(daughters) < 2:
        return "a decay needs at least two daughters (names separated by spaces)"
    if len(daughters) > MAX_DAUGHTERS:
        return f"at most {MAX_DAUGHTERS} daughters"
    unknown = [d for d in daughters if d not in PARTICLES]
    if unknown:
        return f"'{unknown[0]}' is not in the particle table (known: {' '.join(PARTICLES)})"
    at_once = [d for d in daughters if PARTICLES[d].width > 0.0]
    if at_once:
        return (
            f"{at_once[0]} as a daughter is not supported here:"
            " only some of its own decays are in the table"
        )
    total = sum(PARTICLES[d].mass for d in daughters)
    if total >= parent.mass:
        return (
            f"kinematically forbidden: the daughters weigh {total:.5g} GeV together,"
            f" the {name} only {parent.mass:.5g} GeV"
        )
    charge = sum(PARTICLES[d].charge for d in daughters)
    if charge != parent.charge:
        return (
            f"charge is not conserved: the {name} has charge {parent.charge},"
            f" the daughters {charge} in all"
        )
    return ""


# ------------------------------------------------------------- kinematics


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _momentum(entry):
    """Momentum [GeV] the entry asks for, from its quantity and value (nominal mass).

    Raises ValueError with the reason if the value gives none.
    """
    m = PARTICLES[entry.name].mass
    value = entry.value
    if not _finite(value):
        raise ValueError("the value is not a number")
    if value < 0.0:
        raise ValueError("the value cannot be negative")
    if entry.quantity == "momentum":
        p = float(value)
    elif entry.quantity == "energy":
        if value < m:
            raise ValueError(
                f"a total energy of {value:.5g} GeV is below the rest energy"
                f" of the {entry.name}, {m:.5g} GeV"
            )
        p = math.sqrt(max(value * value - m * m, 0.0))
    elif entry.quantity == "kinetic":
        p = math.sqrt(value * (value + 2.0 * m))
    else:
        raise ValueError(f"unknown quantity '{entry.quantity}'")
    if p > P_MAX:
        raise ValueError(f"at most {P_MAX:g} GeV of momentum (the LHC's beam energy)")
    if m > 0.0 and p < AT_REST:
        return 0.0
    return p


def _direction(entry):
    """(unit vector, theta in degrees) of the entry's direction. Raises ValueError."""
    if not _finite(entry.phi):
        raise ValueError("the azimuth is not a number")
    if entry.angles == "theta":
        if not _finite(entry.theta) or not 0.0 <= entry.theta <= 180.0:
            raise ValueError("the polar angle must be between 0 and 180 degrees")
        theta = math.radians(entry.theta)
    elif entry.angles == "eta":
        if not _finite(entry.eta) or abs(entry.eta) > ETA_MAX:
            raise ValueError(f"the pseudorapidity must be between -{ETA_MAX:g} and {ETA_MAX:g}")
        theta = 2.0 * math.atan(math.exp(-entry.eta))
    else:
        raise ValueError(f"unknown way of giving the direction '{entry.angles}'")
    phi = math.radians(entry.phi)
    sin_theta = math.sin(theta)
    return (
        np.array([sin_theta * math.cos(phi), sin_theta * math.sin(phi), math.cos(theta)]),
        math.degrees(theta),
    )


def kinematics(entry, field_strength):
    """The derived quantities of an entry (see the module docstring), for the
    nominal mass and the set momentum (before any spread). Raises ValueError
    if the entry's particle, value or direction is not usable."""
    particle = PARTICLES.get(entry.name)
    if particle is None:
        raise ValueError(f"'{entry.name}' is not in the particle table")
    p = _momentum(entry)
    direction, theta = _direction(entry)
    m = particle.mass
    energy = math.hypot(p, m)
    if energy <= 0.0:
        raise ValueError(
            f"a {entry.name} has no mass: at zero energy there is no particle. Give it some energy"
        )
    pt = p * math.hypot(direction[0], direction[1])
    if pt < 1e-12 * max(p, 1.0):
        pt = 0.0  # along the beam (sin(180 degrees) is not exactly zero)
    tan_half = math.tan(math.radians(theta) / 2.0)
    if theta <= 0.0:
        eta = math.inf
    elif theta >= 180.0:
        eta = -math.inf
    else:
        eta = -math.log(tan_half)
    decays = particle.ctau is not None and bool(modes_of(entry.name))
    radius = None
    if particle.charge != 0 and field_strength > 0.0 and pt > 0.0:
        radius = float(helix_radius(pt, particle.charge, field_strength))
    return Kinematics(
        momentum=p,
        energy=energy,
        kinetic=energy - m,
        beta=p / energy,
        gamma=energy / m if m > 0.0 else math.inf,
        pt=pt,
        theta=theta,
        eta=eta,
        direction=tuple(float(v) for v in direction),
        decay_length=p / m * particle.ctau if decays and m > 0.0 else None,
        radius=radius,
    )


# -------------------------------------------------------------- checking


def check(entry, field_strength=3.8):
    """(errors, warnings) of one entry, as sentences. An entry with errors cannot
    be generated; warnings say what to expect."""
    errors, warnings = [], []
    particle = PARTICLES.get(entry.name)
    if particle is None:
        return [f"'{entry.name}' is not in the particle table"], warnings
    if entry.action not in ACTIONS:
        return [f"unknown action '{entry.action}' (fire or decay)"], warnings
    kin = None
    try:
        kin = kinematics(entry, field_strength)
    except ValueError as error:
        errors.append(str(error))
    start = entry.start
    if (
        not isinstance(start, (tuple, list))
        or len(start) != 3
        or not all(_finite(v) for v in start)
    ):
        errors.append("the starting point needs three numbers (x, y, z in metres)")
    elif math.hypot(start[0], start[1]) >= OUTER_RADIUS or abs(start[2]) >= OUTER_Z:
        errors.append(
            f"the starting point is outside the detector (radius {OUTER_RADIUS:g} m,"
            f" half length {OUTER_Z:g} m)"
        )
    elif math.hypot(start[0], start[1]) >= TRACKER_RADIUS or abs(start[2]) >= TRACKER_HALF_LENGTH:
        if particle.charge != 0 or entry.action == "decay":
            warnings.append("starts outside the tracker: momenta from there are not measured")
    if isinstance(entry.count, bool) or not isinstance(entry.count, int):
        errors.append("the number of particles must be a whole number")
    elif not 1 <= entry.count <= MAX_COUNT:
        errors.append(f"between 1 and {MAX_COUNT} particles per entry")
    if not _finite(entry.spread_p) or not 0.0 <= entry.spread_p <= SPREAD_P_MAX:
        errors.append(f"the momentum spread must be between 0 and {SPREAD_P_MAX:g} percent")
    if not _finite(entry.spread_angle) or not 0.0 <= entry.spread_angle <= SPREAD_ANGLE_MAX:
        errors.append(f"the direction spread must be between 0 and {SPREAD_ANGLE_MAX:g} degrees")

    if entry.action == "fire":
        if particle.width > 0.0:
            errors.append(
                f"a {entry.name} decays at once and never crosses the detector:"
                " use 'Decay a parent' for it"
            )
    else:
        if entry.where not in WHERE:
            errors.append(f"unknown place of decay '{entry.where}'")
        if not modes_of(entry.name):
            errors.append(
                f"the {entry.name} is stable here (it has no decay in the table),"
                " so it cannot be a parent: use 'Fire' for it"
            )
        else:
            daughters = parse_daughters(entry.daughters)
            if daughters or entry.typed:
                problem = _mode_problem(entry.name, daughters)
                if problem:
                    errors.append(problem)
                elif not _is_listed(entry.name, daughters):
                    warnings.append(
                        "typed decay: only energy and charge are checked (not lepton or"
                        " baryon number); the momenta are shared by phase space"
                    )
            elif entry.name in INCOMPLETE:
                warnings.append(
                    f"random among the listed modes only, which are {100.0 * listed_share(entry.name):.3g}%"
                    f" of all {entry.name} decays (the others have particles not in the table);"
                    " drawn in proportion to their PDG fractions, divided by the listed share so that the fractions sum to one"
                )
    if kin is None or errors:
        return errors, warnings

    if kin.momentum == 0.0:
        if entry.action == "decay" or (kin.decay_length is not None):
            warnings.append("at rest: shown decaying at once (its mean life is not waited for)")
        else:
            warnings.append("at rest: it stays where it is put")
    elif entry.action == "decay" and particle.width == 0.0 and entry.where == "lifetime":
        if kin.decay_length is not None and kin.decay_length > LONG_FLIGHT:
            warnings.append(
                f"mean decay length {kin.decay_length:.3g} m: it will usually shower or leave"
                " before it decays. 'At its starting point' forces the decay"
            )
    if particle.charge != 0 and kin.momentum > 0.0:
        if abs(kin.eta) > 3.0:
            warnings.append("closer to the beam line than |eta| = 3: the tracker cannot measure it")
        elif kin.radius is not None and 2.0 * kin.radius < TRACKER_RADIUS:
            warnings.append(f"curls up inside the tracker (circle of radius {kin.radius:.3g} m)")
    return errors, warnings


def placed(setup):
    """Number of particles a setup places (valid counts only)."""
    return sum(e.count for e in setup.entries if isinstance(e.count, int) and e.count > 0)


def problems(setup, field_strength=3.8):
    """Every reason the setup cannot be generated, as sentences (empty: it can)."""
    out = []
    if not setup.entries:
        out.append("The setup is empty: add a particle")
    if len(setup.entries) > MAX_ENTRIES:
        out.append(f"At most {MAX_ENTRIES} entries (there are {len(setup.entries)})")
    for k, entry in enumerate(setup.entries):
        out += [f"Entry {k + 1} ({entry.name}): {text}" for text in check(entry, field_strength)[0]]
    if placed(setup) > MAX_PLACED:
        out.append(f"At most {MAX_PLACED} particles in all (the setup places {placed(setup)})")
    for members in collision_groups(setup).values():
        numbers = " and ".join(str(k + 1) for k in members)
        if len(members) == 1:
            out.append(f"Entry {numbers} is set to collide, but no other entry collides with it")
        elif len(members) > 2:
            out.append(f"Entries {numbers}: only two entries can collide with each other")
        else:
            a, b = (setup.entries[k] for k in members)
            problem = collision_problem(a, b)
            if problem:
                out.append(f"Entries {numbers} cannot collide: {problem}")
    return out


# -------------------------------------------------------------- collisions


def meeting(start_a, p4a, start_b, p4b):
    """(point [m], time [ns]) where two particles starting at start_a and
    start_b with four-momenta p4a and p4b meet on straight paths, or None if
    their paths do not meet in the sense of the module docstring: one start
    point, or both on one line parallel to the beam axis, each flying
    towards the other (or at rest)."""
    a, b = np.asarray(start_a, dtype=float), np.asarray(start_b, dtype=float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1.0e-9:
        return a.copy(), 0.0
    u = d / length
    if math.hypot(u[0], u[1]) > 1.0e-9:
        return None  # off the beam axis the solenoid bends charged tracks
    speeds = []
    for p4, towards in ((p4a, u), (p4b, -u)):
        p = np.asarray(p4[1:], dtype=float)
        size = float(np.linalg.norm(p))
        if size == 0.0:
            speeds.append(0.0)
            continue
        if float(p @ towards) / size < 1.0 - 1.0e-9:
            return None
        speeds.append(size / float(p4[0]))  # beta
    if speeds[0] + speeds[1] == 0.0:
        return None  # both at rest, apart
    when = length / ((speeds[0] + speeds[1]) * C)
    return a + speeds[0] * C * when * u, when


def _fired_p4(entry, kin):
    return four_momentum(PARTICLES[entry.name].mass, kin.momentum * np.array(kin.direction))


def collision_problem(a, b):
    """Why entries a and b cannot be set to collide, or "" if they can."""
    if a.action != "fire" or b.action != "fire":
        return "only two fired particles can collide (not a parent that decays)"
    if frozenset((a.name, b.name)) not in FORMATION:
        return (
            f"{a.name} and {b.name} form no resonance simulated here"
            " (only e+ e- and mu+ mu-, which form a Z)"
        )
    if a.count != 1 or b.count != 1:
        return "each of the two must be a single particle (count 1)"
    if a.spread_angle != 0.0 or b.spread_angle != 0.0:
        return "a direction spread would make them miss each other: set it to 0"
    try:
        kin_a, kin_b = kinematics(a, 0.0), kinematics(b, 0.0)
    except ValueError as error:
        return str(error)
    if meeting(a.start, _fired_p4(a, kin_a), b.start, _fired_p4(b, kin_b)) is None:
        return (
            "their paths do not meet: start them at one point, or on one line along"
            " the beam axis aimed at each other ('Face each other'); off the axis"
            " the magnet would bend them apart"
        )
    return ""


def collision_groups(setup):
    """{collision name: [entry indices]} of the entries set to collide."""
    groups = {}
    for k, entry in enumerate(setup.entries):
        if entry.collide:
            groups.setdefault(entry.collide, []).append(k)
    return groups


def collision_partner(setup, index):
    """Index of the entry that entry `index` is set to collide with, or None."""
    label = setup.entries[index].collide
    if not label:
        return None
    others = [k for k in collision_groups(setup).get(label, []) if k != index]
    return others[0] if len(others) == 1 else None


def unpair(setup, index):
    """Entry `index` and every entry set to collide with it collide no more."""
    label = setup.entries[index].collide
    if label:
        for entry in setup.entries:
            if entry.collide == label:
                entry.collide = ""


def pair_up(setup, i, j):
    """Set entries i and j to collide with each other (and with nothing else)."""
    unpair(setup, i)
    unpair(setup, j)
    used = {e.collide for e in setup.entries}
    n = 1
    while f"c{n}" in used:
        n += 1
    setup.entries[i].collide = setup.entries[j].collide = f"c{n}"


def nominal_sqrt_s(a, b):
    """sqrt(s) [GeV] of entries a and b at their set momenta (no spread), or None."""
    try:
        kin_a, kin_b = kinematics(a, 0.0), kinematics(b, 0.0)
    except ValueError:
        return None
    return float(mass(_fired_p4(a, kin_a) + _fired_p4(b, kin_b)))


def collide(p4a, p4b, resonance, rng):
    """Does a pair with four-momenta p4a, p4b form `resonance`? (module
    docstring, "Collisions"). A dict: sqrt_s, within (WINDOW_WIDTHS),
    probability (the Breit-Wigner weight, 0 outside the window) and formed.
    Draws a random number only within the window."""
    particle = PARTICLES[resonance]
    sqrt_s = float(mass(np.asarray(p4a, dtype=float) + np.asarray(p4b, dtype=float)))
    within = abs(sqrt_s - particle.mass) <= WINDOW_WIDTHS * particle.width
    probability = (
        float(breit_wigner_formation(sqrt_s, particle.mass, particle.width)) if within else 0.0
    )
    formed = bool(within and rng.uniform() < probability)
    return {
        "resonance": resonance,
        "sqrt_s": sqrt_s,
        "within": bool(within),
        "probability": probability,
        "formed": formed,
    }


def collision_caption(record):
    """One line about a collision record (Event.collisions)."""
    name, root_s = record["resonance"], record["sqrt_s"]
    if record["formed"]:
        return (
            f"Forms a {name} at sqrt(s) = {root_s:.4f} GeV"
            f" (within {WINDOW_WIDTHS:g} widths: yes)"
        )
    if record["within"]:
        return (
            f"No {name} this event at sqrt(s) = {root_s:.4f} GeV (within {WINDOW_WIDTHS:g} widths:"
            f" yes; formed with probability {record['probability']:.3g} at this sqrt(s));"
            " they pass through"
        )
    return (
        f"No {name} at sqrt(s) = {root_s:.4g} GeV (within {WINDOW_WIDTHS:g} widths: no);"
        " they pass through"
    )


# -------------------------------------------------------------- histogram


def _histogram_entry(setup):
    """The first "decay" entry: the one whose parent the histogram is about."""
    for entry in setup.entries:
        if entry.action == "decay" and entry.name in PARTICLES and modes_of(entry.name):
            return entry
    return None


def _neutrino_share(entry):
    """Fraction of the entry's first decays that have a neutrino among the products
    (directly, or through a tau)."""
    unseen = ("nu", "tau-", "tau+")
    daughters = parse_daughters(entry.daughters)
    if daughters:
        return 1.0 if any(d in unseen for d in daughters) else 0.0
    modes = modes_of(entry.name)
    total = sum(b for b, _ in modes)
    return sum(b for b, d in modes if any(x in unseen for x in d)) / total


def _round(value, digits=3):
    if value == 0.0:
        return 0.0
    return round(value, digits - 1 - int(math.floor(math.log10(abs(value)))))


def histogram_for(setup):
    """The Hist of a setup (see the module docstring), or None if it has no parent."""
    entry = _histogram_entry(setup)
    if entry is None:
        return None
    particle = PARTICLES[entry.name]
    m = particle.mass
    line = ((m, entry.name),)
    if _neutrino_share(entry) > 0.5:
        return Hist(
            f"visible mass of the {entry.name}'s products [GeV]", 0.0, _round(1.2 * m), 60, line
        )
    half = max(8.0 * particle.width, 0.15 * m)
    low, high = _round(max(m - half, 0.0)), _round(m + half)
    return Hist(f"mass of the {entry.name}'s products [GeV]", low, high, 60, line)


# ------------------------------------------------------------- generation


def _draw(entry, kin, rng):
    """Three-momentum of one particle of the entry: the set one, with its spread."""
    p = kin.momentum
    if p == 0.0:
        return np.zeros(3)
    if entry.spread_p > 0.0:
        factor = 1.0 + 0.01 * entry.spread_p * rng.normal()
        p = min(p * max(abs(factor), 1e-3), P_MAX)  # a spread of 50% can reach below zero
    vector = p * np.array(kin.direction)
    if entry.spread_angle > 0.0:
        vector = scatter(vector, math.radians(entry.spread_angle), rng)
    return vector


def _instant(event, name, p4, where, when, parent=-1, origin="collision"):
    """Add a particle that decays where and when it is made; returns its index."""
    point = np.array(where, dtype=float)
    p4 = np.array(p4, dtype=float)
    track = Track(
        name, p4, np.stack([point, point]), np.array([when, when]), "decay", p4.copy(),
        parent=parent, hits=[], origin=origin,
    )  # fmt: skip
    event.tracks.append(track)
    event.duration = max(event.duration, when)
    return len(event.tracks) - 1


def _choose(name, entry, rng):
    """The daughters of the parent's decay: the entry's, or drawn from the listed fractions."""
    daughters = parse_daughters(entry.daughters)
    if daughters:
        return daughters
    modes = modes_of(name)
    weights = np.array([b for b, _ in modes])
    return tuple(modes[rng.choice(len(modes), p=weights / weights.sum())][1])


def _decay_into(event, index, daughters, field_strength, rng, quick):
    """Let track `index` (which ended by decaying) decay into `daughters`."""
    track = event.tracks[index]
    name = track.name
    particle = PARTICLES[name]
    where, when = track.points[-1], float(track.times[-1])
    # As in particle_events._produce: a resonance keeps the mass it was made
    # with, anything else has its own; the momentum is the one it has left.
    m_parent = float(mass(track.p4)) if particle.width > 0.0 else particle.mass
    parent_p4 = four_momentum(m_parent, track.p4_end[1:])
    if name == "H" and tuple(daughters) == HIGGS_ZZ:
        # H -> Z Z* -> four charged leptons, as in the "Higgs -> 4 leptons" scene.
        m1, m2 = z_pair_masses(rng, m_parent)
        for z in two_body_decay(parent_p4, m1, m2, rng):
            flavour = rng.choice(["mu", "e"])
            z_index = _instant(event, "Z", z, where, when, index, "decay")
            _decay_into(event, z_index, (f"{flavour}-", f"{flavour}+"), field_strength, rng, quick)
        return
    momenta = decay_momenta(name, parent_p4, daughters, rng)
    for d, q in zip(daughters, momenta, strict=True):
        if quick and PARTICLES[d].kind == "muon":
            # Only the masses are wanted: a muon's measurement uses its
            # momentum and starting point alone (particle_events.measured).
            q = np.array(q, dtype=float)
            point = np.array(where, dtype=float)
            event.tracks.append(
                Track(d, q, point[None, :], np.array([when]), "exit", q.copy(),
                      parent=index, hits=[], origin="decay")
            )  # fmt: skip
            continue
        _produce(event, d, q, where, when, field_strength, rng, index, 1, "decay")


def _place_parent(event, entry, momentum, field_strength, rng):
    """Put the parent of a "decay" entry into the event; returns its index."""
    name = entry.name
    particle = PARTICLES[name]
    m = particle.mass
    if particle.width > 0.0:
        # A mass from the line shape, above what the chosen daughters need.
        daughters = parse_daughters(entry.daughters)
        need = 0.0 if tuple(daughters) == HIGGS_ZZ else sum(PARTICLES[d].mass for d in daughters)
        for _ in range(100):
            m = float(breit_wigner_mass(particle.mass, particle.width, rng))
            if m > need:
                break
        else:
            m = particle.mass
    p4 = four_momentum(m, momentum)
    at_rest = not np.any(momentum)
    if particle.width > 0.0 or entry.where == "start" or at_rest or not particle.ctau:
        return _instant(event, name, p4, entry.start, 0.0)
    lifetime = rng.exponential(particle.ctau / C) * p4[0] / m  # time dilation: gamma = E / m
    track, _ = propagate(
        name, particle.kind, particle.charge, p4, entry.start, 0.0, field_strength, lifetime, rng
    )
    event.tracks.append(track)
    event.duration = max(event.duration, float(track.times[-1]))
    return len(event.tracks) - 1


def family_of(event, index):
    """Indices of track `index` and everything that came from it."""
    family = {index}
    for i, t in enumerate(event.tracks):  # parents always come before children
        if t.parent in family:
            family.add(i)
    return family


def measured_to_kink(track, field_strength, rng):
    """Three-momentum of a charged track that decayed in flight, as the tracker
    measures it over the segment from its start to the kink (its decay
    point), or None if that segment cannot be measured.

    As particle_events.measured (true momentum at the start, curvature
    smeared by the resolution), with the projected length L of the fit cut
    at the kink: the largest distance across the field from the start
    reached by the points of the path inside the tracker, and at most the
    lever arm of an undecayed track (particle_events.lever_arm). The rule
    for fewer than MIN_LAYERS layers is momentum_resolution's.
    """
    particle = PARTICLES[track.name]
    if particle.charge == 0 or track.end != "decay" or len(track.points) < 2:
        return None
    p = track.p4[1:]
    start = track.points[0]
    pts = track.points
    inside = (np.hypot(pts[:, 0], pts[:, 1]) < TRACKER_RADIUS) & (
        np.abs(pts[:, 2]) < TRACKER_HALF_LENGTH
    )
    if not inside.any():
        return None
    reach = float(np.max(np.hypot(pts[inside, 0] - start[0], pts[inside, 1] - start[1])))
    lever = min(lever_arm(start, p, particle.charge, field_strength), reach)
    pt, total = float(np.hypot(p[0], p[1])), float(np.linalg.norm(p))
    sigma = momentum_resolution(pt, total, particle.mass, lever, field_strength, particle.charge)
    if sigma is None:
        return None
    curvature = max(abs(1.0 + sigma * rng.normal()), 0.05)  # as in measured()
    return p / curvature


def _measure(event, index, daughters, field_strength, rng):
    """((label, mass), "") of what track `index` decayed into, as measured, or
    (None, why not). See the module docstring."""
    tracks = event.tracks
    parent = tracks[index]
    family = family_of(event, index) - {index}
    # A charged particle that decayed in flight with a measurable segment
    # before its kink counts as itself (measured_to_kink); what came of it
    # is then left out (`covered`), else it would be counted twice.
    final, kinked, covered = [], {}, set()
    for i in sorted(family):  # parents always come before children
        t = tracks[i]
        if t.parent in covered:
            covered.add(i)
            continue
        if t.end == "decay":
            q = measured_to_kink(t, field_strength, rng)
            if q is not None:
                kinked[i] = q
                final.append(i)
                covered.add(i)
            continue
        if PARTICLES[t.name].kind != "neutrino" and t.origin not in ("radiated", "conversion"):
            final.append(i)
    neutrinos = any(
        PARTICLES[tracks[i].name].kind == "neutrino" for i in family if i not in covered
    )
    if len(final) < 2:
        seen = f"only the {tracks[final[0]].name} is" if final else "nothing is"
        return None, f"No mass for the {parent.name}: {seen} visible of its decay"
    total = np.zeros(4)
    for i in final:
        t = tracks[i]
        particle = PARTICLES[t.name]
        if i in kinked:
            q4 = four_momentum(particle.mass, kinked[i])
        elif particle.kind == "electron":
            q4 = lepton_measured(event, i, field_strength, rng)
        elif particle.charge != 0:
            q = measured(t, field_strength, rng)
            q4 = None if q is None else four_momentum(particle.mass, q)
        elif particle.kind == "photon":
            # Direction exact, energy with the calorimeter's resolution (as
            # for electrons, particle_events.lepton_measured).
            energy = float(t.p4[0])
            sigma = np.sqrt((ECAL_A / np.sqrt(energy)) ** 2 + ECAL_C**2)
            q4 = t.p4 * max(1.0 + sigma * rng.normal(), 0.05)
        else:
            q4 = t.p4  # neutral hadrons: true momenta
        if q4 is None:
            return None, (
                f"No mass for the {parent.name}: its {t.name} could not be measured"
                " (it starts outside the tracker or too close to the beam line)"
            )
        total = total + q4
    if neutrinos:
        label = f"the visible products of the {parent.name}"
    elif parent.name == "H" and tuple(daughters) == HIGGS_ZZ:
        label = "the four leptons from the H"
    else:
        label = f"{' '.join(daughters)} from the {parent.name}"
    return (label, float(mass(total))), ""


def describe(entry):
    """An entry in a few words: "3 x K0S 5 GeV -> pi+ pi-"."""
    many = f"{entry.count} x " if entry.count != 1 else ""
    unit = {"momentum": "GeV", "energy": "GeV total", "kinetic": "GeV kinetic"}.get(
        entry.quantity, "GeV"
    )
    amount = (
        "at rest"
        if entry.quantity == "momentum" and entry.value == 0
        else f"{entry.value:.4g} {unit}"
    )
    text = f"{many}{entry.name} {amount}"
    if entry.action == "decay":
        mode = entry.daughters.strip() or "any listed decay"
        if entry.name == "H" and parse_daughters(entry.daughters) == HIGGS_ZZ:
            mode = "Z Z* -> 4 leptons"
        text += f" -> {mode}"
    if entry.collide:
        text += " (collides)"
    return text


def generate_custom(setup, field_strength, rng, quick=False):
    """One event of the custom `setup` in a solenoid of `field_strength` tesla.

    The user's inputs are fixed; what is drawn anew each time is the spread
    of the gun, the resonance masses, when and how things decay, and what
    the material does. quick: only the masses for the histogram are wanted;
    then only the entries that feed it are made.

    Raises ValueError (with the first reason) for a setup that problems()
    refuses.
    """
    refused = problems(setup, field_strength)
    if refused:
        raise ValueError(refused[0])
    event = Event(custom=True, beam_energy=0.0)
    event.entered = []
    histogram = _histogram_entry(setup)
    started = time.perf_counter()
    wanted = placed(setup)
    made, notes, stopped = 0, [], False
    # Pairs set to collide (problems() has checked them): made at the first
    # entry of the pair, skipped at the second.
    partner = {}
    for members in collision_groups(setup).values():
        partner[members[0]] = members[1]
    seconds = set(partner.values())
    for k, entry in enumerate(setup.entries):
        feeds = histogram is not None and entry.action == "decay" and entry.name == histogram.name
        if quick and not feeds:
            continue
        if k in seconds:
            continue
        if k in partner:
            if time.perf_counter() - started > TIME_BUDGET or len(event.tracks) > MAX_TRACKS:
                stopped = True
                break
            made += 2
            _collision(event, setup, k, partner[k], field_strength, rng, notes)
            continue
        kin = kinematics(entry, field_strength)
        particle = PARTICLES[entry.name]
        for _ in range(entry.count):
            if time.perf_counter() - started > TIME_BUDGET or len(event.tracks) > MAX_TRACKS:
                stopped = True
                break
            momentum = _draw(entry, kin, rng)
            made += 1
            if entry.action == "fire":
                p4 = four_momentum(particle.mass, momentum)
                _produce(event, entry.name, p4, np.array(entry.start, dtype=float), 0.0,
                         field_strength, rng, origin="placed")  # fmt: skip
                continue
            index = _place_parent(event, entry, momentum, field_strength, rng)
            event.tracks[index].origin = "placed"
            if event.tracks[index].end != "decay":
                notes.append(
                    f"The {entry.name} did not decay inside the detector"
                    f" ({_ENDED.get(event.tracks[index].end, 'it was lost')})"
                )
                continue
            daughters = _choose(entry.name, entry, rng)
            _decay_into(event, index, daughters, field_strength, rng, quick)
            result, why = _measure(event, index, daughters, field_strength, rng)
            if result is None:
                notes.append(why)
                continue
            event.masses.append(result)
            if feeds:
                event.entered.append(result[1])
        if stopped:
            break
    if stopped:
        notes.insert(
            0,
            f"Only {made} of the {wanted} placed particles were followed"
            " (limit on the time and size of one event)",
        )
    shown = [describe(e) for e in setup.entries[:3]]
    more = f"; and {len(setup.entries) - 3} more entries" if len(setup.entries) > 3 else ""
    if event.collisions:
        what = "; ".join(
            f"entries {r['entries'][0] + 1} and {r['entries'][1] + 1}: "
            + collision_caption(r)[0].lower() + collision_caption(r)[1:]
            for r in event.collisions
        )
        start = f"Custom (placed by hand; {what}): "
    else:
        start = "Custom (placed by hand, not a collision): "
    event.headline = start + "; ".join(shown) + more
    unique = list(dict.fromkeys(notes))
    event.note = ". ".join(unique[:2]) + (
        f" (and {len(unique) - 2} more)" if len(unique) > 2 else ""
    )
    return event


def _collision(event, setup, i, j, field_strength, rng, notes):
    """Entries i and j, set to collide: the resonance they form and its decay,
    or the two fired as usual if it is not formed (module docstring,
    "Collisions"). Appends the record to event.collisions."""
    a, b = setup.entries[i], setup.entries[j]
    p4 = []
    for entry in (a, b):
        momentum = _draw(entry, kinematics(entry, field_strength), rng)
        p4.append(four_momentum(PARTICLES[entry.name].mass, momentum))
    resonance = FORMATION[frozenset((a.name, b.name))]
    record = collide(p4[0], p4[1], resonance, rng)
    record.update(entries=(i, j), names=(a.name, b.name))
    event.collisions.append(record)
    met = meeting(a.start, p4[0], b.start, p4[1]) if record["formed"] else None
    if met is None:  # not formed: they pass through each other, as without the option
        record["formed"] = False
        for entry, q in zip((a, b), p4, strict=True):
            _produce(event, entry.name, q, np.array(entry.start, dtype=float), 0.0,
                     field_strength, rng, origin="placed")  # fmt: skip
        return
    point, when = met
    record["point"], record["time"] = tuple(float(v) for v in point), float(when)
    # The resonance takes the pair's whole four-momentum (its mass is sqrt(s));
    # the two fired particles are absorbed and leave no track.
    index = _instant(event, resonance, p4[0] + p4[1], point, when, origin="collision")
    record["index"] = index
    daughters = _choose(resonance, Entry("decay", resonance, daughters=""), rng)
    _decay_into(event, index, daughters, field_strength, rng, False)
    result, why = _measure(event, index, daughters, field_strength, rng)
    if result is None:
        notes.append(why)
    else:
        event.masses.append(result)


_ENDED = {
    "shower": "it showered first",
    "exit": "it left first",
    "stopped": "it stopped",
    "lost": "it kept curling in the magnet",
    "conversion": "it converted",
}


# ------------------------------------------------------ presets, current


def _presets():
    return {
        "Muon gun 50 GeV": Setup([Entry("fire", "mu-", value=50.0)]),
        "Electron and photon 30 GeV": Setup(
            [
                Entry("fire", "e-", value=30.0, phi=20.0),
                Entry("fire", "gamma", value=30.0, phi=-20.0),
            ]
        ),
        "Pion gun with smearing": Setup(
            [Entry("fire", "pi+", value=5.0, count=12, spread_p=20.0, spread_angle=15.0)]
        ),
        "K0S at 5 GeV": Setup([Entry("decay", "K0S", value=5.0, daughters="pi+ pi-")]),
        "Lambda at 5 GeV": Setup([Entry("decay", "Lambda", value=5.0, daughters="p pi-")]),
        "Z at rest to e+ e-": Setup([Entry("decay", "Z", value=0.0, daughters="e+ e-")]),
        "J/psi at 10 GeV to mu+ mu-": Setup(
            [Entry("decay", "J/psi", value=10.0, daughters="mu+ mu-")]
        ),
        "tau at 45 GeV": Setup([Entry("decay", "tau-", value=45.0)]),
        "Higgs at rest to 4 leptons": Setup([Entry("decay", "H", value=0.0, daughters="Z Z")]),
    }


PRESETS = tuple(_presets())
DEFAULT_PRESET = PRESETS[0]


def preset(name):
    """A fresh copy of the preset `name`."""
    return _presets()[name]


# The setup the scene runs: kept at module level so that it survives the
# particle module being rebuilt (Restart). Never changed in place: set_current
# stores a copy, and the threads that generate events hold on to the object
# they started with.
_current = preset(DEFAULT_PRESET)


def current():
    return _current


def set_current(setup):
    """Make `setup` the one the Custom scene runs. Raises ValueError (with the
    reasons) and changes nothing if it cannot be generated."""
    global _current
    refused = problems(setup)
    if refused:
        raise ValueError("; ".join(refused))
    _current = deepcopy(setup)


def reset():
    """Back to the default setup (Full reset)."""
    global _current
    _current = preset(DEFAULT_PRESET)


# ---------------------------------------------------------- save and load


def to_dict(setup):
    """The setup as a JSON-ready dictionary (for scenario_files)."""
    entries = []
    for entry in setup.entries:
        item = asdict(entry)
        item["start"] = [float(v) for v in entry.start]
        entries.append(item)
    return {"format": FORMAT, "entries": entries}


def from_dict(data):
    """The Setup saved by to_dict. Raises ValueError with a readable reason for
    anything else: wrong types, unknown names, values out of range, a decay
    that is not allowed."""
    if not isinstance(data, dict):
        raise ValueError("a custom particle setup must be a dictionary")
    version = data.get("format", FORMAT)
    if isinstance(version, bool) or not isinstance(version, int) or version > FORMAT or version < 1:
        raise ValueError("the setup was saved in a format this version does not read")
    items = data.get("entries")
    if not isinstance(items, list):
        raise ValueError("the setup has no list of entries")
    if len(items) > MAX_ENTRIES:
        raise ValueError(f"at most {MAX_ENTRIES} entries (the file has {len(items)})")
    default = Entry()
    kinds = {
        "action": str, "name": str, "quantity": str, "angles": str, "daughters": str,
        "where": str, "collide": str, "typed": bool, "count": int,
        "value": float, "theta": float, "eta": float, "phi": float,
        "spread_p": float, "spread_angle": float,
    }  # fmt: skip
    entries = []
    for k, item in enumerate(items):
        what = f"entry {k + 1}"
        if not isinstance(item, dict):
            raise ValueError(f"{what} is not a dictionary")
        if "name" not in item:
            raise ValueError(f"{what} does not name a particle")
        values = {}
        for key, kind in kinds.items():
            value = item.get(key, getattr(default, key))
            if kind is float:
                if not _finite(value):
                    raise ValueError(f"{what}: '{key}' must be a number")
                value = float(value)
            elif kind is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{what}: '{key}' must be a whole number")
            elif not isinstance(value, kind):
                raise ValueError(f"{what}: '{key}' has the wrong type")
            values[key] = value
        start = item.get("start", [0.0, 0.0, 0.0])
        if not isinstance(start, list) or len(start) != 3 or not all(_finite(v) for v in start):
            raise ValueError(f"{what}: 'start' must be three numbers")
        entry = Entry(start=tuple(float(v) for v in start), **values)
        if entry.quantity not in QUANTITIES:
            raise ValueError(f"{what}: unknown quantity '{entry.quantity}'")
        if entry.angles not in ANGLES:
            raise ValueError(f"{what}: unknown way of giving the direction '{entry.angles}'")
        entries.append(entry)
    setup = Setup(entries)
    refused = problems(setup)
    if refused:
        raise ValueError(refused[0])
    return setup
