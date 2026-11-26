"""Events of the particle-physics module: what is produced, how it decays, what is measured.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
4). Review the physics.

What is computed from the physics: energy and momentum conservation in
every decay (particle_kinematics.py), with the particles' measured masses,
lifetimes and branching fractions (particle_data.py: every listed decay is
drawn with its PDG fraction); where each decay happens; how every charge
moves in the field (particle_tracks.py); and the invariant mass of the
momenta that the measurement model below gives.

What is an approximate model: how the momenta are shared out in most decays
to three or more bodies (phase space, see decay_momenta); what the material
does (mean energy loss, Gaussian scattering, parametrised showers:
particle_material.py); the detector's resolution (below).

What is NOT simulated: the collision itself. Which particles come out of a
proton-proton collision, how many and with what momenta is the business of
event generators such as Pythia; here those are drawn from simple
distributions matched to measured averages, not derived:
- a Z, J/psi or Higgs gets a transverse momentum from an exponential (mean
  PT_MEAN) and a rapidity from a Gaussian (width Y_WIDTH);
- the "soft" particles accompanying it have the measured density and mean
  transverse momentum (SOFT_DENSITY, SOFT_PT), exponential in transverse
  momentum and uniform in pseudorapidity within +-SOFT_ETA, with the mix
  of species in SOFT_MIX. Real ones come in correlated jets; these do not.
The sentences and the diagram about how a Z, J/psi or Higgs is produced
(MECHANISMS) are an illustration of the usual mechanism, nothing the module
computes; for recorded events no mechanism is stated at all, since only the
two muons were measured.
One piece of real dynamics is kept: a Z decays to a fermion pair with the
angular distribution 1 + cos^2(theta) about the beam axis in its rest frame
(spin 1; the small forward-backward asymmetry is left out).

Measurement. The momenta are NOT fitted from points along the paths. Each
charged particle's true momentum at its starting point is smeared by the
resolution that a tracker like this one would have, which follows the field
and the geometry (measured, momentum_resolution). A tracker measures the
curvature k = 1 / R = 0.3 B / p_T of the path's projection across the field;
its error has two parts (Gluckstern 1963, Nucl. Instrum. Methods 24, 381;
PDG, "Particle detectors at accelerators", eqs. 35.63 to 35.65):
    sigma(p_T) / p_T = sqrt(res^2 + ms^2),
    res = eps p_T / (0.3 B L^2) sqrt(720 / (N + 4))     (N points of error eps over a projected length L),
    ms  = 0.016 GeV / (0.3 B L beta) sqrt(L / (X0 cos(lambda)))   (multiple scattering in the tracker's material),
with p_T in GeV, B in T, lengths in m, lambda the angle of the path to the
plane across the field (cos(lambda) = p_T / p) and X0 the tracker's radiation
length. So the resolution improves as 1 / B, and the first part as 1 / L^2.
L is the distance, across the field, from the particle's starting point to
where a straight line along its initial direction leaves the tracker
(through the outer cylinder or an end), and at most the diameter of its
circle (lever_arm); N = TRACKER_LAYERS L / TRACKER_RADIUS layers of
POINT_RESOLUTION each. A particle with fewer than three layers (closer to
the beam line than |eta| = 3, or made near the tracker's edge) is not
measured, and an event that needs it gives no mass. With eps = 35 micrometres
and 13 layers a central muon in 3.8 T has 0.8% at low p_T, 1.0% at 45 GeV
and 1.6% at 100 GeV, close to CMS. The smearing is Gaussian in the
curvature; the direction is taken as exact. Electrons radiate in the
tracker, so their energy is taken from the calorimeter instead
(lepton_measured). Neutral particles, where used (the visible mass of
Z -> tau tau), are taken with their true momenta.

Scenes with a forced decay chain (added 2026-11-26 under the owner's
approval; review the physics). "B+ -> J/psi K+" and "D*+ -> D0 pi+" make
one particle at the collision point (transverse momentum and rapidity as
for the Z: PT_MEAN, Y_WIDTH) and force the first decays of the chain
(_produce, `forced`) instead of drawing them from the tables; everything
after is drawn from the tables as usual. Shown is only that chain; the
texts give its PDG fraction (forced_fraction). The B+ (c tau 491.1 um)
and the D0 (123.01 um) fly for their drawn lifetimes, so a millimetre or
less. The J/psi from B+ -> J/psi K+ is polarised (decay_momenta,
polarised_by). The D0's oscillation (particle_data.MIXING: it decays as a
D0bar with probability 3.3e-5) is not applied to its forced decay. Measured
mass (_measure): of everything visible that came from the B+ (mu+ mu- K+)
or the D0 (K- pi+), each particle with its own mass, charged ones by the
tracker model above. A charged particle that decayed in flight (a K or pi)
is measured over its path from its start to the kink if that segment has
the layers for it (measured_to_kink); else the event gives no mass.
"Higgs -> 2 photons": the Higgs as in "Higgs -> 4 leptons", decaying at
once to two photons, isotropic in its rest frame (spin 0). Each photon is
measured with its true direction and its true energy smeared by the
calorimeter's resolution (ECAL_A, ECAL_C, as for electrons); a photon that
converted in the tracker still counts as the photon (photon_measured).
Only this mode is shown: in the Standard Model 0.227% of Higgs bosons decay
to two photons (LHC Higgs Cross Section Working Group, CERN Yellow Report 4,
CERN-2017-002).

Collisions: "e+ e- -> Z (collider scan)" (adapted 2026-11-26 from the
deleted Custom scene, written 2026-11-23 under the owner's override; review
the physics). Here the collision IS computed, in the following sense.
- An electron and a positron fly at each other along the beam (z) axis
  with equal energies E and meet at the origin at t = 0 (meeting checks
  that their straight paths meet; along the axis the solenoid does not bend
  them). Each event sets sqrt(s) = 2 E anew, uniformly within M_Z -+
  SCAN_WIDTHS Gamma_Z (a scan, as if the collider were retuned for every
  event; no beam-energy spread). sqrt(s): s = (p1 + p2)^2, the invariant
  mass of the pair (PDG 2024, review "Kinematics"); head on with equal
  energies sqrt(s) = 2 E exactly.
- Formation (collide). No resonance is formed unless |sqrt(s) - M| <=
  WINDOW_WIDTHS widths. Within that window a Z is formed with probability
      w(sqrt(s)) = (Gamma/2)^2 / ((sqrt(s) - M)^2 + (Gamma/2)^2)
  (particle_kinematics.breit_wigner_formation; PDG 2024, review
  "Resonances", non-relativistic Breit-Wigner; Z: M = 91.1876 GeV, Gamma =
  2.4952 GeV from particle_data.py). So the sqrt(s) of the collisions that
  formed a Z traces the Breit-Wigner line shape; with sqrt(s) uniform in
  M -+ 2 Gamma the fraction formed is atan(4) / 4 = 0.3315. STATED
  APPROXIMATION: the absolute rate is a shape, not computed from a cross
  section: a pair exactly at the peak always forms the Z (probability 1),
  which a real pair would do only rarely (it would mostly pass by: the peak
  cross section is a few tens of nanobarns). Left out: initial-state
  radiation (which lowers the effective sqrt(s) and gives the measured line
  shape its tail above the peak), the photon exchange and its interference
  with the Z, and the forward-backward asymmetry.
- The Z's four-momentum is p1 + p2 exactly: at rest, with mass sqrt(s). The
  Breit-Wigner gives how likely each sqrt(s) is to form it, not a mass
  drawn apart from the pair (that would break energy-momentum conservation).
- It decays at once at the origin by its listed modes (particle_data.py:
  e+ e-, mu+ mu-, tau+ tau-; only the charged-lepton modes are listed, so
  Z -> nu nu and Z -> hadrons do not occur), with the angular distribution
  1 + cos^2(theta) about the beam axis (_lepton_pair, as in the other Z
  scenes).
- Not formed: nothing happens in this model (other processes, such as
  scattering through photon exchange, are not simulated); the two fly on
  along the beam line and leave. Nothing enters the histogram.
The histogram is of sqrt(s), which is set, not measured: of the collisions
that formed a Z.
"""

from dataclasses import dataclass, field

import numpy as np

from physengine.modules.particle_data import _ANTI, A1, ISOBARS, MIXING, PARTICLES, RESONANCES, RHO
from physengine.modules.particle_kinematics import (
    breit_wigner_formation,
    breit_wigner_mass,
    four_momentum,
    leptonic_decay,
    longitudinal_vector_decay,
    mass,
    phase_space_decay,
    semileptonic_decay,
    tau_decay_via_a1,
    tau_decay_via_rho,
    two_body_decay,
    vector_isobar_decay,
)
from physengine.modules.particle_material import TRACKER, TRACKER_HALF_LENGTH
from physengine.modules.particle_tracks import OUTER_Z, TRACKER_RADIUS, C, K, Track, propagate

# "B+" and "D*+" (added 2026-11-26): chosen display values, not
# measurements (high enough that most of their tracks reach across the tracker).
PT_MEAN = {"Z": 8.0, "J/psi": 6.0, "H": 20.0, "B+": 8.0, "D*+": 6.0}
Y_WIDTH = 1.2
# Soft particles: measured averages for proton-proton collisions at 13 TeV:
# about 6 charged particles per unit of pseudorapidity (CMS Collaboration
# 2015, Phys. Lett. B 751, 143) with a mean transverse momentum of about
# 0.55 GeV, and roughly half as many neutral ones.
SOFT_PT, SOFT_ETA = 0.40, 2.2  # exponential above 0.15 GeV: mean 0.55
SOFT_DENSITY = 9.0  # particles per unit of pseudorapidity, charged and neutral
M_HIGGS = 125.25  # GeV (its width, 4 MeV, is far below the resolution: ignored)
SOFT_MIX = (
    ("pi+", 0.27), ("pi-", 0.27), ("pi0", 0.16), ("K+", 0.05), ("K-", 0.05),
    ("K0S", 0.06), ("K0L", 0.04), ("p", 0.03), ("pbar", 0.03), ("Lambda", 0.04),
)  # fmt: skip
POINT_RESOLUTION = 35.0e-6  # m: error of one measured point across the field
TRACKER_LAYERS = 13  # measured points on a path that crosses the whole tracker radius
MIN_LAYERS = 3  # fewer points than this do not give a curvature
ECAL_A, ECAL_C = 0.028, 0.003
MAX_GENERATIONS = 6
Z_STAR_MIN = 12.0  # GeV: lowest Z* mass kept (as in the experiments' selections)
WINDOW_WIDTHS = 10.0  # a pair forms a resonance only within this many widths of its mass
SCAN_WIDTHS = 2.0  # the e+ e- scan draws sqrt(s) within M_Z -+ this many widths
BEAM_START = 0.75  # m: where the e+ e- beams are drawn from (C x 2.5 ns, the playback's lead-in)


@dataclass
class Event:
    tracks: list = field(default_factory=list)
    headline: str = ""  # what was made in the collision, in words
    # Particles that decay at the collision point and leave no track of
    # their own: dicts with name, p4, daughters (names) and what made them.
    bosons: list = field(default_factory=list)
    # How it was made, for the close-up diagram: (incoming a, incoming b,
    # what they form, text), see MECHANISMS.
    mechanism: str = ""
    # The two beam particles (names in particle_data.PARTICLES): the first
    # flies in from -z, the second from +z.
    beams: tuple = ("p", "p")
    beam_energy: float = 6.5  # TeV per beam particle
    # How event.masses were obtained, for the texts: "tracker" (the model
    # tracker, electrons from the calorimeter), "calorimeter" (photons), or
    # "beams" (sqrt(s), set by the beams, not measured).
    measurement: str = "tracker"
    # e+ e- -> Z only: the collision record of collide() (sqrt_s, within,
    # probability, formed); None for the other scenes.
    formation: dict | None = None
    real: bool = False  # a recorded event, not a generated one
    duration: float = 0.0  # ns until the last particle has stopped or left
    # (label, value) pairs measured in this event, e.g. ("mu+ mu-", 91.3)
    masses: list = field(default_factory=list)
    in_histogram: bool = True  # False: the masses are shown but not entered (same-sign pairs)
    note: str = ""  # a remark for the commentary (e.g. why no mass was measured)
    error: str = ""  # set when the event could not be made: the reason, for the display




def _decay_time(particle, p4, rng):
    """Lab-frame time [ns] after which the particle decays, or None if it is stable here."""
    if particle.ctau is None or not particle.decays:
        return None
    proper = rng.exponential(particle.ctau / C) if particle.ctau > 0.0 else 0.0
    return proper * p4[0] / particle.mass  # time dilation: gamma = E / m


def _produce(
    event,
    name,
    p4,
    vertex,
    t0,
    field_strength,
    rng,
    parent=-1,
    generation=0,
    origin="collision",
    quick=False,
    forced=None,
):
    """Add `name` to the event, follow it, and let it decay in turn.

    quick: for events made only for the histogram. A muon from the collision
    or from a decay is then not followed through the detector (its
    measurement uses only its momentum and starting point, see `measured`);
    it gets a track of one point.
    forced: {name: daughters}: the first particle of that name in this chain
    that decays does so into `daughters`, instead of a mode drawn from its
    table (the scenes with a forced chain, module docstring). Passed on to
    the daughters without the entry used.
    """
    particle = PARTICLES[name]
    if quick and particle.kind == "muon" and (parent < 0 or origin == "decay"):
        p4 = np.array(p4, dtype=float)
        start = np.array(vertex, dtype=float)
        event.tracks.append(
            Track(name, p4, start[None, :], np.array([t0]), "exit", p4.copy(),
                  parent=parent, hits=[], origin=origin)
        )  # fmt: skip
        return
    lifetime = _decay_time(particle, p4, rng)
    track, photons = propagate(
        name, particle.kind, particle.charge, p4, vertex, t0, field_strength, lifetime, rng
    )
    track.parent = parent
    track.origin = origin
    event.tracks.append(track)
    index = len(event.tracks) - 1
    event.duration = max(event.duration, track.times[-1])
    for photon, where, when in photons:
        # Radiated by an electron on its way through the tracker.
        _produce(
            event, "gamma", photon, where, when, field_strength, rng, index, generation, "radiated"
        )
    if track.end == "conversion" and generation < MAX_GENERATIONS:
        # The photon becomes an electron-positron pair (particle_material.py).
        energy = track.p4_end[0]
        direction = track.p4_end[1:] / np.linalg.norm(track.p4_end[1:])
        share = rng.uniform(0.02, 0.98)
        m_e = PARTICLES["e-"].mass
        for d, part in (("e-", share * energy), ("e+", (1.0 - share) * energy)):
            momentum = np.sqrt(max(part * part - m_e * m_e, 0.0)) * direction
            _produce(
                event, d, np.array([part, *momentum]), track.points[-1], track.times[-1],
                field_strength, rng, index, generation + 1, "conversion",
            )  # fmt: skip
        return
    if track.end != "decay" or generation >= MAX_GENERATIONS:
        return
    if forced and name in forced:
        # A forced decay: no draw from the table, and no oscillation.
        decays_as, daughters = name, tuple(forced[name])
        forced = {k: v for k, v in forced.items() if k != name}
    else:
        # A neutral B, Bs or D0 may have oscillated: it then decays by its
        # antiparticle's table (decay_flavour). The track keeps the name it was made with.
        proper_time = lifetime * particle.mass / p4[0]
        decays_as = decay_flavour(name, proper_time, rng)
        probabilities, modes = PARTICLES[decays_as].modes()
        daughters = modes[rng.choice(len(modes), p=probabilities)]
    # The parent's momentum where it decays (turned by the field, and
    # lowered by ionisation, on the way). A resonance keeps the mass it was
    # made with; anything else has its own.
    m_parent = float(mass(track.p4)) if particle.width > 0.0 else particle.mass
    parent_p4 = four_momentum(m_parent, track.p4_end[1:])
    momenta = decay_momenta(
        decays_as, parent_p4, daughters, rng, polarised_by=_b_to_jpsi_k(event, name, p4, parent)
    )
    for d, q in zip(daughters, momenta, strict=True):
        _produce(
            event,
            d,
            q,
            track.points[-1],
            track.times[-1],
            field_strength,
            rng,
            index,
            generation + 1,
            "decay",
            quick,
            forced,
        )


_PIONS = ("pi+", "pi-", "pi0")

# Form-factor models of the semileptonic decays P -> P' l nu
# (particle_kinematics.semileptonic_form_factors), by (parent, daughter meson);
# the charge conjugates are added below.
# - B -> D: CLN, rho^2 = 1.129 +- 0.024 +- 0.023, the HFLAV average (spring
#   2021, B -> D l nu, CLN fit, isospin assumed).
# - Bs -> Ds: CLN, rho^2(Ds-) = 1.27 +- 0.05, LHCb 2020, Phys. Rev. D 101,
#   072004 (arXiv:2001.03225), nominal CLN fit.
# - D -> K: BESIII 2015 (PRD 92, 072012) two-parameter series fit to
#   D0 -> K- e+ nu, r1 = -2.2286 +- 0.0864 +- 0.0573, pole at the Ds*+ mass
#   2112.2 MeV (PDG 2024). Used for D+ -> Kbar0 l nu as well (isospin).
# - D -> pi: the same fit to D0 -> pi- e+ nu, r1 = -2.0365 +- 0.0807 +- 0.0257,
#   P = 1. Used for D+ -> pi0 l nu as well (isospin).
# - Ds -> K0: no fit at hand; single pole at the D*(2010)+ mass, 2010.26 MeV
#   (PDG 2024), the nominal vector pole of c -> d.
# - B -> D*: CLN for a vector meson (particle_kinematics.vector_helicity_amplitudes),
#   rho^2 = 1.121 +- 0.014 +- 0.019, R1(1) = 1.269 +- 0.026, R2(1) = 0.853 +-
#   0.017: HFLAV spring 2021 average of B0 -> D*- l nu (CLN fit), with
#   eta_EW F(1) |V_cb| = (35.00 +- 0.11 +- 0.34)e-3 (hflav-eos.web.cern.ch,
#   semi/spring21/html/ExclusiveVcb/exclBtoDstar.html). Used for B+ -> D*0bar
#   l nu as well (isospin).
_CLN_BD = ("CLN", 1.129)
_CLN_BDSTAR = ("CLN-V", 1.121, 1.269, 0.853)
_D_TO_K = ("z2", -2.2286, 2.1122)
_D_TO_PI = ("z2", -2.0365, None)
SEMILEPTONIC_MODELS = {
    ("B+", "D0bar"): _CLN_BD,
    ("B0", "D-"): _CLN_BD,
    ("B+", "D*0bar"): _CLN_BDSTAR,
    ("B0", "D*-"): _CLN_BDSTAR,
    ("Bs", "Ds-"): ("CLN", 1.27),
    ("D0", "K-"): _D_TO_K,
    ("D+", "K0S"): _D_TO_K,
    ("D+", "K0L"): _D_TO_K,
    ("D0", "pi-"): _D_TO_PI,
    ("D+", "pi0"): _D_TO_PI,
    ("Ds+", "K0S"): ("pole", 2.01026),
    ("Ds+", "K0L"): ("pole", 2.01026),
}
SEMILEPTONIC_MODELS.update(
    {(_ANTI[a], _ANTI.get(b, b)): model for (a, b), model in list(SEMILEPTONIC_MODELS.items())}
)
_CHARGED_LEPTONS = ("e+", "e-", "mu+", "mu-", "tau+", "tau-")


def flip_probability(name, proper_time):
    """Probability that a `name` made at t = 0 decays as its antiparticle at
    `proper_time` [ns] (particle_data.MIXING); 0 for anything that does not mix."""
    if name not in MIXING:
        return 0.0
    x, y = MIXING[name]
    u = proper_time * C / PARTICLES[name].ctau  # Gamma t
    return 0.5 * (1.0 - np.cos(x * u) / np.cosh(y * u))


def decay_flavour(name, proper_time, rng):
    """The flavour a neutral meson made as `name` decays with: `name` or its
    antiparticle. Only flavour-specific modes notice: a CP eigenstate
    (J/psi K0S, K+ K-, Ds+ Ds-, ...) has the same fraction in both tables,
    which are mirror images (particle_data._conjugate)."""
    if rng.uniform() < flip_probability(name, proper_time):
        return _ANTI[name]
    return name


_SPIN0_KAONS = ("K+", "K-", "K0S", "K0L")
_B_MESONS = ("B+", "B-", "B0", "B0bar", "Bs", "Bsbar")


def _b_to_jpsi_k(event, name, p4, parent):
    """The B's four-momentum if `name` is a J/psi made in B -> J/psi K (a spin-0
    kaon and nothing else), else None. Told from the recoil: (B - J/psi)^2 is
    a kaon's mass squared only in the two-body decay (in J/psi K pi, J/psi
    K pi pi, J/psi K K, J/psi pi pi it takes a continuum of values). Works
    for the Bs of events."""
    if name != "J/psi" or parent < 0 or event.tracks[parent].name not in _B_MESONS:
        return None
    b = event.tracks[parent]
    b_p4 = four_momentum(PARTICLES[b.name].mass, b.p4_end[1:])
    recoil = float(mass(b_p4 - np.asarray(p4, dtype=float)))
    if any(abs(recoil - PARTICLES[k].mass) < 1e-6 for k in _SPIN0_KAONS):
        return b_p4
    return None


def daughter_mass(name, rng):
    """Mass a daughter `name` is made with: its own, or, for a resonance with a
    width, one drawn from its Breit-Wigner (see decay_momenta)."""
    particle = PARTICLES[name]
    if particle.width > 0.0:
        return float(breit_wigner_mass(particle.mass, particle.width, rng))
    return particle.mass


def decay_momenta(name, parent_p4, daughters, rng, polarised_by=None):
    """Four-momenta of the daughters of `name` -> `daughters`.

    - Two bodies: fixed by energy and momentum conservation, isotropic,
      except:
    - J/psi -> l+ l- with `polarised_by` (the four-momentum of the B, for a
      J/psi from B -> J/psi K): the J/psi has helicity 0, the l+ follows
      sin^2(theta) + (4 m_l^2 / M^2) cos^2(theta) about the J/psi's flight
      direction in the B's frame (particle_kinematics.longitudinal_vector_decay).
      J/psi from B -> J/psi K pi (through K*, whose polarisation needs the
      measured amplitudes) and other B modes stay isotropic.
    - l -> l' nu nu (the tau's leptonic decays): the weak interaction's
      energy sharing (particle_kinematics.leptonic_decay).
    - tau -> pi pi0 nu: through the rho(770) (tau_decay_via_rho).
    - tau -> three pions nu (pi- pi+ pi- and pi- pi0 pi0): through the
      a1(1260) -> rho pi (tau_decay_via_a1); the rho is the first two
      daughters.
    - B -> D* l nu (e, mu): the V - A helicity amplitudes with CLN form
      factors (particle_kinematics.vector_semileptonic_density).
    - P -> P' l nu of the D and B mesons (B -> D l nu, Bs -> Ds mu nu,
      D -> K l nu, D -> pi l nu, Ds -> K0 e nu): the V - A matrix element
      with a form-factor model (SEMILEPTONIC_MODELS,
      particle_kinematics.semileptonic_decay).
    - Ds+ -> K+ K- pi+, D0 -> K- pi+ pi0, D+ -> K- pi+ pi+ (and their
      conjugates): the isobar model of isobar_decay, through the phi, K*(892)
      and rho(770) with the PDG's sub-channel fractions
      (particle_data.ISOBARS); the remaining fraction is phase space.
    - Everything else with three or more bodies (K -> pi l nu, K -> 3 pi,
      pi0 -> e+ e- gamma, tau -> K pi nu, tau -> 4 pi nu, the stand-in
      modes, and the other decays of the D and B mesons, among them the
      semileptonic ones with two or more hadrons): phase space only, an
      approximation. It leaves out the resonances (K*, rho, D*) inside the
      multi-body final states and the angular distributions that spin gives.

    A daughter with a width (the D*+, a J/psi) gets a mass from its
    Breit-Wigner line shape (particle_kinematics.breit_wigner_mass, within
    10 widths of its mass), which it keeps when it decays in turn (_produce).
    """
    masses = [daughter_mass(d, rng) for d in daughters]
    if polarised_by is not None and name == "J/psi" and len(daughters) == 2:
        # Daughters are (l+, l-) in the table: the angle is that of the first.
        first, second = longitudinal_vector_decay(parent_p4, masses[0], polarised_by, rng)
        return [first, second] if daughters[0].endswith("+") else [second, first]
    if len(daughters) == 3 and daughters[1] == "nu" and daughters[2] == "nu":
        return leptonic_decay(parent_p4, masses[0], rng)
    if len(daughters) == 3 and daughters[2] == "nu" and daughters[1] in _CHARGED_LEPTONS:
        model = SEMILEPTONIC_MODELS.get((name, daughters[0]))
        if model is not None:
            return semileptonic_decay(parent_p4, masses[0], masses[1], model, rng)
    hadronic_tau = name in ("tau-", "tau+") and daughters[-1] == "nu"
    if hadronic_tau and all(d in _PIONS for d in daughters[:-1]):
        if len(daughters) == 3:
            return tau_decay_via_rho(parent_p4, masses[0], masses[1], RHO, rng)
        if len(daughters) == 4:
            return tau_decay_via_a1(parent_p4, *masses[:3], RHO, A1, rng)
    if (name, tuple(sorted(daughters))) in _ISOBARS:
        return isobar_decay(name, parent_p4, daughters, rng, masses)[0]
    return phase_space_decay(parent_p4, masses, rng)


# particle_data.ISOBARS by (parent, daughters sorted): a decay typed in another
# order finds its sub-channels too.
_ISOBARS = {(parent, tuple(sorted(d))): entry for (parent, d), entry in ISOBARS.items()}


def isobar_probabilities(name, daughters):
    """[(resonance, probability), ...] of the sub-channels of `name` -> `daughters`
    in the isobar model (particle_data.ISOBARS), [] if it has none. The
    probability of a sub-channel is B_i / B_mode; where the sub-channels add up
    to more than the mode (interference, which is left out), B_i / sum_j B_j.
    What is left, 1 - sum, is phase space."""
    entry = _ISOBARS.get((name, tuple(sorted(daughters))))
    if entry is None:
        return []
    total, channels = entry
    norm = max(total, sum(b for b, _, _ in channels))
    return [(resonance, b / norm) for b, resonance, _ in channels]


def isobar_decay(name, parent_p4, daughters, rng, masses=None):
    """(four-momenta of `daughters`, resonance or None) for `name` -> `daughters`
    in the isobar model.

    A sub-channel is drawn with its probability (isobar_probabilities), or
    none (phase space, resonance None) with what is left. Through a
    resonance V: its mass from its Breit-Wigner, parent -> V + spectator
    isotropic, V -> its two daughters with cos^2 of the helicity angle
    (particle_kinematics.vector_isobar_decay). Where the spectator is the
    same particle as one of V's daughters (the two pi+ of D+ -> K- pi+ pi+),
    which of the two comes from V is drawn at random, half and half.
    Stated approximation: the sub-channels are added as probabilities; the
    interference between their amplitudes is left out.
    """
    daughters = tuple(daughters)
    if masses is None:
        masses = [daughter_mass(d, rng) for d in daughters]
    _, channels = _ISOBARS[(name, tuple(sorted(daughters)))]
    probabilities = isobar_probabilities(name, daughters)
    u, chosen = rng.uniform(), None
    for (_, resonance, pair), (_, p) in zip(channels, probabilities, strict=True):
        if u < p:
            chosen = resonance, pair
            break
        u -= p
    if chosen is None:
        return phase_space_decay(parent_p4, masses, rng), None
    resonance, pair = chosen
    i = daughters.index(pair[0])
    candidates = [k for k, d in enumerate(daughters) if d == pair[1] and k != i]
    j = candidates[int(rng.integers(len(candidates)))]
    (k,) = [n for n in range(3) if n not in (i, j)]
    (a, b, c), _ = vector_isobar_decay(
        parent_p4, masses[i], masses[j], masses[k], RESONANCES[resonance], rng
    )
    out = [None] * 3
    out[i], out[j], out[k] = a, b, c
    return out, resonance


def _from_pt_y(m, pt, y, phi):
    """Four-momentum from mass, transverse momentum, rapidity and azimuth."""
    mt = np.sqrt(m * m + pt * pt)
    return np.array([mt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y)])


def _resonance(name, rng):
    """Four-momentum of a particle made at the collision point (a Z, J/psi, B+ or
    D*+): transverse momentum and rapidity as in the module docstring, and
    for one with a width a Breit-Wigner mass (breit_wigner_mass returns the
    mass itself for width 0)."""
    particle = PARTICLES[name]
    m = breit_wigner_mass(particle.mass, particle.width, rng)
    pt = rng.exponential(PT_MEAN[name])
    return _from_pt_y(m, pt, rng.normal(0.0, Y_WIDTH), rng.uniform(0.0, 2.0 * np.pi))


def z_pair_masses(rng, m_higgs=M_HIGGS):
    """Masses (heavier first) of the two Z bosons in H -> Z Z* -> four leptons.

    The Higgs (125 GeV) is lighter than two Z bosons (182 GeV), so at least
    one of them is far off its mass shell. The rate is (e.g. Djouadi 2008,
    Phys. Rep. 457, 1, eq. 2.31)
        dGamma / (dq1^2 dq2^2)  proportional to  BW(q1^2) BW(q2^2) sqrt(lam) (lam + 12 q1^2 q2^2 / M_H^4),
        BW(q^2) = M_Z Gamma_Z / ((q^2 - M_Z^2)^2 + M_Z^2 Gamma_Z^2),
        lam = (1 - q1^2 / M_H^2 - q2^2 / M_H^2)^2 - 4 q1^2 q2^2 / M_H^4,
    for q1 + q2 < M_H. Sampled by rejection in the two masses (so with the
    Jacobian 4 q1 q2), above Z_STAR_MIN: the usual pair is one Z near 91 GeV
    and one around 30 GeV.
    """
    z = PARTICLES["Z"]
    mz, gz = z.mass, z.width

    def density(m1, m2):
        q1, q2 = m1 * m1, m2 * m2
        h2 = m_higgs * m_higgs
        lam = (1.0 - q1 / h2 - q2 / h2) ** 2 - 4.0 * q1 * q2 / (h2 * h2)
        lam = np.where((m1 + m2 < m_higgs) & (lam > 0.0), lam, 0.0)
        bw = lambda q: mz * gz / ((q - mz * mz) ** 2 + mz * mz * gz * gz)  # noqa: E731
        return bw(q1) * bw(q2) * np.sqrt(lam) * (lam + 12.0 * q1 * q2 / (h2 * h2)) * 4.0 * m1 * m2

    high = m_higgs - Z_STAR_MIN
    grid = np.linspace(Z_STAR_MIN, high, 600)
    bound = 1.3 * density(grid[:, None], grid[None, :]).max()
    while True:
        m1 = rng.uniform(Z_STAR_MIN, high, 4000)
        m2 = rng.uniform(Z_STAR_MIN, high, 4000)
        ok = np.flatnonzero(rng.uniform(0.0, bound, 4000) < density(m1, m2))
        if len(ok):
            a, b = m1[ok[0]], m2[ok[0]]
            return (a, b) if a >= b else (b, a)


def _higgs(event, field_strength, rng, quick=False):
    """H -> Z Z* -> four charged leptons (electrons or muons)."""
    m1, m2 = z_pair_masses(rng)
    higgs = _from_pt_y(
        M_HIGGS,
        rng.exponential(PT_MEAN["H"]),
        rng.normal(0.0, Y_WIDTH),
        rng.uniform(0.0, 2.0 * np.pi),
    )
    first, second = two_body_decay(higgs, m1, m2, rng)
    event.bosons.append(
        {"name": "H", "p4": higgs, "daughters": ("Z", "Z"), "from": "the collision"}
    )
    flavours = [rng.choice(["mu", "e"]), rng.choice(["mu", "e"])]
    for z, flavour in zip((first, second), flavours, strict=True):
        # Each Z decays isotropically here: the correlations between the four
        # leptons' directions (which carry the Higgs's spin 0) are left out.
        _lepton_pair(event, z, (f"{flavour}-", f"{flavour}+"), field_strength, rng, False, quick)
        event.bosons.append(
            {
                "name": "Z",
                "p4": z,
                "daughters": (f"{flavour}-", f"{flavour}+"),
                "from": "the Higgs boson",
            }
        )
    event.headline = (
        f"Generated: a Higgs boson -> Z ({m1:.1f} GeV) + Z* ({m2:.1f} GeV)"
        f" -> {flavours[0]}- {flavours[0]}+ {flavours[1]}- {flavours[1]}+"
    )


def _lepton_pair(event, parent, daughters, field_strength, rng, polarised, quick=False):
    """parent -> two leptons at the origin; `polarised`: 1 + cos^2(theta) about the beam."""
    direction = None
    if polarised:
        while True:  # rejection sampling of 1 + cos^2
            cos_theta = rng.uniform(-1.0, 1.0)
            if rng.uniform(0.0, 2.0) < 1.0 + cos_theta**2:
                break
        phi = rng.uniform(0.0, 2.0 * np.pi)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        direction = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    first, second = two_body_decay(
        parent, PARTICLES[daughters[0]].mass, PARTICLES[daughters[1]].mass, rng, direction
    )
    for d, q in zip(daughters, (first, second), strict=True):
        _produce(event, d, q, np.zeros(3), 0.0, field_strength, rng, quick=quick)


def _soft_particles(event, field_strength, rng):
    """The many low-momentum particles that accompany any collision (see SOFT_DENSITY)."""
    names = [n for n, _ in SOFT_MIX]
    weights = np.array([w for _, w in SOFT_MIX])
    for _ in range(rng.poisson(SOFT_DENSITY * 2.0 * SOFT_ETA)):
        name = names[rng.choice(len(names), p=weights / weights.sum())]
        pt = 0.15 + rng.exponential(SOFT_PT)
        eta = rng.uniform(-SOFT_ETA, SOFT_ETA)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        momentum = np.array([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)])
        _produce(
            event,
            name,
            four_momentum(PARTICLES[name].mass, momentum),
            np.zeros(3),
            0.0,
            field_strength,
            rng,
        )


def _strange(event, field_strength, rng):
    """A handful of K0_S and Lambda with a few GeV, to watch them fly and decay."""
    for _ in range(rng.integers(3, 6)):
        name = "K0S" if rng.uniform() < 0.6 else "Lambda"
        pt = 0.6 + rng.exponential(1.5)
        eta = rng.uniform(-1.2, 1.2)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        momentum = np.array([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)])
        _produce(
            event,
            name,
            four_momentum(PARTICLES[name].mass, momentum),
            np.zeros(3),
            0.0,
            field_strength,
            rng,
        )


def forced_fraction(parent, daughters):
    """PDG branching fraction of `parent` -> `daughters` as listed in particle_data
    (in any order of the daughters), 0 if it is not listed."""
    wanted = sorted(daughters)
    return float(sum(b for b, d in PARTICLES[parent].decays if sorted(d) == wanted))


def _distance_text(metres):
    if metres >= 0.01:
        return f"{metres * 100:.3g} cm"
    if metres >= 1e-4:
        return f"{metres * 1000:.2g} mm"
    return f"{metres * 1e6:.2g} micrometres"


# The forced chains: (particle made at the collision point, its forced
# decays in order, the particle whose products are measured, their label).
CHAINS = {
    "B+ -> J/psi K+": ("B+", (("B+", ("J/psi", "K+")), ("J/psi", ("mu+", "mu-"))), "B+", "mu+ mu- K+"),
    "D*+ -> D0 pi+": ("D*+", (("D*+", ("D0", "pi+")), ("D0", ("K-", "pi+"))), "D0", "K- pi+"),
}  # fmt: skip

_ENDED = {
    "shower": "it interacted in the material first",
    "exit": "it left first",
    "stopped": "it stopped",
    "lost": "it kept curling in the magnet",
}


def _chain(scene, event, field_strength, rng, quick=False):
    """One event of a scene with a forced decay chain (CHAINS, module docstring)."""
    name, decays, measured_name, label = CHAINS[scene]
    _produce(
        event, name, _resonance(name, rng), np.zeros(3), 0.0, field_strength, rng,
        quick=quick, forced=dict(decays),
    )  # fmt: skip
    first = event.tracks[0]
    steps = [
        f"{parent} -> {' '.join(d)} ({100.0 * forced_fraction(parent, d):.4g}%)"
        for parent, d in decays
    ]
    shown = (
        "Only this chain is shown: " + ", then ".join(steps)
        + " of all decays (PDG); the decays after it are drawn from the tables"
    )  # fmt: skip
    index = next((i for i, t in enumerate(event.tracks) if t.name == measured_name), None)
    if index is None or event.tracks[index].end != "decay":
        track = event.tracks[0 if index is None else index]
        why = _ENDED.get(track.end, "it was lost")
        event.headline = f"Generated: a {name} that did not decay inside the detector"
        event.note = f"No mass: the {measured_name} did not decay inside the detector ({why}). {shown}"
        return
    flight = float(np.linalg.norm(event.tracks[index].points[-1] - event.tracks[index].points[0]))
    if name == measured_name:
        p = float(np.linalg.norm(first.p4[1:]))
        event.headline = (
            f"Generated: a {name} ({p:.3g} GeV) flies {_distance_text(flight)}, then decays"
            f" -> {' '.join(decays[0][1])}; {decays[1][0]} -> {' '.join(decays[1][1])}"
        )
    else:
        event.headline = (
            f"Generated: a {name} (mass {float(mass(first.p4)):.6g} GeV) decays at once"
            f" -> {' '.join(decays[0][1])}; the {decays[1][0]} flies {_distance_text(flight)},"
            f" then decays -> {' '.join(decays[1][1])}"
        )
    result, why = _measure(event, index, label, field_strength, rng)
    if result is None:
        event.note = f"{why}. {shown}"
    else:
        event.masses = [result]
        event.note = shown


def _higgs_photons(event, field_strength, rng, quick=False):
    """H -> gamma gamma (module docstring), measured by the calorimeter model."""
    higgs = _from_pt_y(
        M_HIGGS,
        rng.exponential(PT_MEAN["H"]),
        rng.normal(0.0, Y_WIDTH),
        rng.uniform(0.0, 2.0 * np.pi),
    )
    # Spin 0: isotropic in the Higgs's rest frame.
    for q in two_body_decay(higgs, 0.0, 0.0, rng):
        _produce(event, "gamma", q, np.zeros(3), 0.0, field_strength, rng, quick=quick)
    event.bosons.append(
        {"name": "H", "p4": higgs, "daughters": ("gamma", "gamma"), "from": "the collision"}
    )
    event.mechanism = "H"
    event.measurement = "calorimeter"
    event.headline = "Generated: a Higgs boson decays at once -> gamma gamma"
    shown = (
        "Only this mode is shown: in the Standard Model 0.227% of Higgs bosons decay to two"
        " photons (LHC Higgs Cross Section WG, CERN-2017-002)"
    )
    photons = [i for i, t in enumerate(event.tracks) if t.parent < 0 and t.name == "gamma"]
    momenta = [photon_measured(event, i, rng) for i in photons]
    if len(photons) == 2 and all(q is not None for q in momenta):
        event.masses = [("gamma gamma", float(mass(momenta[0] + momenta[1])))]
        event.note = shown
    else:
        event.note = f"No mass: a photon did not reach the calorimeter. {shown}"


def _z_scan(event, field_strength, rng, quick=False):
    """e+ e- -> Z at a sqrt(s) near the Z mass ("Collisions" in the module
    docstring). quick: the Z's decay products are not made (the histogram is
    of sqrt(s))."""
    z = PARTICLES["Z"]
    m_e = PARTICLES["e-"].mass
    root_s = rng.uniform(z.mass - SCAN_WIDTHS * z.width, z.mass + SCAN_WIDTHS * z.width)
    energy = 0.5 * root_s
    pz = np.sqrt(energy * energy - m_e * m_e)
    electron = np.array([energy, 0.0, 0.0, pz])  # flies in from -z
    positron = np.array([energy, 0.0, 0.0, -pz])  # from +z
    start_e, start_p = np.array([0.0, 0.0, -BEAM_START]), np.array([0.0, 0.0, BEAM_START])
    met = meeting(start_e, electron, start_p, positron)
    # Symmetric starts and equal speeds: they meet at the origin (the clock
    # is set so that this is t = 0).
    if met is None or not np.allclose(met[0], 0.0, atol=1e-12):
        raise RuntimeError("the e- and e+ beams do not meet at the origin")
    record = collide(electron, positron, "Z", rng)
    event.formation = record
    event.beams = ("e-", "e+")
    event.beam_energy = energy / 1000.0
    event.measurement = "beams"
    root_s = record["sqrt_s"]
    chance = f"probability {record['probability']:.3g} at this sqrt(s)"
    if not record["formed"]:
        # They fly on along the beam line, unbent (no transverse momentum).
        for name, p4 in (("e-", electron), ("e+", positron)):
            direction = np.sign(p4[3])
            z_end = np.linspace(0.0, direction * OUTER_Z, 12)
            points = np.stack([np.zeros(12), np.zeros(12), z_end], 1)
            times = np.abs(z_end) / (pz / energy * C)
            event.tracks.append(
                Track(name, p4.copy(), points, times, "exit", p4.copy(), hits=[], origin="beam")
            )
            event.duration = max(event.duration, float(times[-1]))
        event.headline = (
            f"Generated: e- e+ at sqrt(s) = {root_s:.4f} GeV: no Z formed ({chance});"
            " they fly on along the beam line"
        )
        event.note = (
            f"No Z was formed at sqrt(s) = {root_s:.4f} GeV ({chance}): nothing enters the"
            " histogram"
        )
        return
    total = electron + positron  # the Z: at rest, mass sqrt(s)
    probabilities, modes = z.modes()
    daughters = tuple(modes[rng.choice(len(modes), p=probabilities)])
    if not quick:
        _lepton_pair(event, total, daughters, field_strength, rng, True)
    event.bosons.append(
        {"name": "Z", "p4": total, "daughters": daughters, "from": "the e- e+ annihilation"}
    )
    event.mechanism = "e+ e- -> Z"
    event.masses = [("sqrt(s)", root_s)]
    event.headline = (
        f"Generated: e- e+ at sqrt(s) = {root_s:.4f} GeV form a Z ({chance}),"
        f" which decays at once -> {' '.join(daughters)}"
    )
    event.note = (
        "The rate is a shape, not a cross section (probability 1 at the peak); initial-state"
        " radiation, the photon exchange and the forward-backward asymmetry are left out;"
        " only the Z's charged-lepton modes are listed"
    )


def meeting(start_a, p4a, start_b, p4b):
    """(point [m], time [ns]) where two particles starting at start_a and
    start_b with four-momenta p4a and p4b meet on straight paths, or None if
    their paths do not meet: one start point, or both on one line parallel
    to the beam axis, each flying towards the other (or at rest)."""
    a, b = np.asarray(start_a, dtype=float), np.asarray(start_b, dtype=float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1.0e-9:
        return a.copy(), 0.0
    u = d / length
    if np.hypot(u[0], u[1]) > 1.0e-9:
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


@dataclass(frozen=True)
class Hist:
    """The histogram of a scene: what is measured and over what range."""

    label: str
    low: float
    high: float
    bins: int
    peaks: tuple = ()  # (known mass, name) drawn as lines
    log: bool = False  # bins equal in log(mass)


_Z_LINE = ((91.1876, "Z"),)
SCENES = {
    # name: (description, histogram)
    "Z -> mu+ mu-": ("A Z boson decays to two muons", Hist("mu+ mu- mass [GeV]", 60.0, 120.0, 60, _Z_LINE)),
    "Z -> e+ e-": ("A Z boson decays to an electron and a positron", Hist("e+ e- mass [GeV]", 60.0, 120.0, 60, _Z_LINE)),
    "Z -> tau+ tau-": ("A Z boson decays to two taus, which decay at once", Hist("visible mass [GeV]", 0.0, 120.0, 60)),
    "J/psi -> mu+ mu-": ("A J/psi meson decays to two muons", Hist("mu+ mu- mass [GeV]", 2.8, 3.4, 60, ((3.0969, "J/psi"),))),
    "Strange particles": ("K0_S and Lambda fly centimetres, then decay to two tracks", Hist("pi+ pi- mass [GeV]", 0.40, 0.60, 50, ((0.497611, "K0S"),))),
    "Busy collision": ("A Z -> mu+ mu- among the debris of a collision", Hist("mu+ mu- mass [GeV]", 60.0, 120.0, 60, _Z_LINE)),
    "Higgs -> 4 leptons": ("A Higgs boson decays to two Z bosons, each to two leptons", Hist("four-lepton mass [GeV]", 105.0, 145.0, 40, ((125.25, "Higgs"),))),
    # Added 2026-11-26 (module docstring: forced chains, H -> gamma gamma, Collisions).
    "B+ -> J/psi K+": ("A B+ meson flies a millimetre or so, then decays to a J/psi and a K+; the J/psi to two muons", Hist("mu+ mu- K+ mass [GeV]", 5.0, 5.6, 60, ((PARTICLES["B+"].mass, "B+"),))),
    "D*+ -> D0 pi+": ("A D*+ decays at once to a D0 and a slow pi+; the D0 flies a fraction of a millimetre, then decays to K- pi+", Hist("K- pi+ mass [GeV]", 1.70, 2.03, 66, ((PARTICLES["D0"].mass, "D0"),))),
    "Higgs -> 2 photons": ("A Higgs boson decays to two photons (0.227% of its decays in the Standard Model: only this mode is shown)", Hist("gamma gamma mass [GeV]", 110.0, 140.0, 60, ((M_HIGGS, "Higgs"),))),
    "e+ e- -> Z (collider scan)": (
        "An electron and a positron collide head on near the Z mass; a Z forms with the Breit-Wigner probability",
        Hist(
            "sqrt(s) of the collisions that formed a Z [GeV]",
            PARTICLES["Z"].mass - SCAN_WIDTHS * PARTICLES["Z"].width,
            PARTICLES["Z"].mass + SCAN_WIDTHS * PARTICLES["Z"].width,
            40, ((PARTICLES["Z"].mass, "Z"),),
        ),
    ),
    # Recorded by CMS in 2011 (particle_opendata.py). The histograms are of
    # opposite-charge pairs; same-charge pairs are shown but not entered.
    "Real data: Z -> mu+ mu-": ("A real collision recorded by CMS: two muons, selected as a Z candidate", Hist("mu+ mu- mass [GeV]", 60.0, 120.0, 60, _Z_LINE)),
    "Real data: J/psi -> mu+ mu-": ("A real collision recorded by CMS: two muons, selected as a J/psi candidate", Hist("mu+ mu- mass [GeV]", 2.8, 3.4, 60, ((3.0969, "J/psi"),))),
    "Real data: all muon pairs": (
        "A real collision recorded by CMS with two muons",
        Hist(
            "mu+ mu- mass [GeV], logarithmic", 2.0, 120.0, 120,
            ((3.0969, "J/psi"), (3.6861, "psi'"), (9.4603, "Upsilon"), (91.1876, "Z")), True,
        ),
    ),
}  # fmt: skip

# What usually happens between the protons when such a particle is made, for
# the close-up diagram and the text: (incoming, incoming, what they form,
# sentence). An illustration of the main known production mechanism: it is
# described, not simulated, and other mechanisms contribute too. It is shown
# for generated events only.
MECHANISMS = {
    "Z": ("quark", "antiquark", "Z", "Illustration (not simulated): a Z is mostly made when a quark from one proton and an antiquark from the other annihilate."),
    "J/psi": ("gluon", "gluon", "c cbar", "Illustration (not simulated): a J/psi is mostly made when two gluons, one from each proton, fuse into a charm quark and antiquark, which bind."),
    "H": ("gluon", "gluon", "H", "Illustration (not simulated): a Higgs boson is mostly made when two gluons, one from each proton, fuse through a loop of top quarks."),
    "soft": ("quark/gluon", "quark/gluon", "strings", "Illustration (not simulated): quarks and gluons knocked out of the protons pull new quark pairs out of the vacuum; these bind into hadrons, some of them strange."),
    "B+": ("gluon", "gluon", "b bbar", "Illustration (not simulated): bottom quarks are mostly made as a quark-antiquark pair from gluons, one from each proton; the anti-b binds with an up quark into the B+ (the b ends up in another hadron, not shown)."),
    "D*+": ("gluon", "gluon", "c cbar", "Illustration (not simulated): charm quarks are mostly made as a quark-antiquark pair from gluons, one from each proton; the c binds with a down antiquark into the D*+ (the anti-c ends up in another hadron, not shown)."),
    # Not an illustration: the formation is computed from sqrt(s) (_z_scan).
    "e+ e- -> Z": ("electron", "positron", "Z", "Computed: the electron and the positron annihilate into a Z, with the Breit-Wigner probability at this sqrt(s) (a shape, not a cross section; the photon exchange is left out)."),
}  # fmt: skip


def bin_index(hist, m):
    """Histogram bin of mass m, or -1 if outside."""
    if not hist.low <= m < hist.high:
        return -1
    if hist.log:
        return int(np.log(m / hist.low) / np.log(hist.high / hist.low) * hist.bins)
    return int((m - hist.low) / (hist.high - hist.low) * hist.bins)


class MassHistogram:
    """The masses measured so far, kept as counts only (nothing grows with the
    number of events): the bins of `hist` for the display, and FINE narrower
    ones over the same range for the median and the width, which are read off
    the cumulative counts and are therefore exact to one fine bin
    (range / FINE: 0.025 GeV for the Z's 60 GeV).
    """

    FINE = 2400

    def __init__(self, hist):
        self.hist = hist
        self.counts = np.zeros(hist.bins, dtype="f4")
        self.fine = np.zeros(self.FINE, dtype=np.int64)
        self.entries = 0  # all masses added, inside the range or not
        self._summary = None

    def _fraction(self, m):
        """Position of masses m in the range, 0 to 1 (on the histogram's own scale)."""
        hist = self.hist
        if hist.log:
            return np.log(m / hist.low) / np.log(hist.high / hist.low)
        return (m - hist.low) / (hist.high - hist.low)

    def _mass(self, fraction):
        hist = self.hist
        if hist.log:
            return hist.low * (hist.high / hist.low) ** fraction
        return hist.low + fraction * (hist.high - hist.low)

    def add(self, masses):
        m = np.asarray(masses, dtype=float).ravel()
        m = m[np.isfinite(m)]
        self.entries += len(m)
        m = m[(m >= self.hist.low) & (m < self.hist.high)]
        if not len(m):
            return
        f = self._fraction(m)
        coarse = np.minimum((f * self.hist.bins).astype(int), self.hist.bins - 1)
        fine = np.minimum((f * self.FINE).astype(int), self.FINE - 1)
        self.counts += np.bincount(coarse, minlength=self.hist.bins).astype("f4")
        self.fine += np.bincount(fine, minlength=self.FINE)
        self._summary = None

    def quantile(self, q):
        """The mass below which a fraction q of the entries inside the range lie."""
        cumulative = np.cumsum(self.fine)
        target = q * cumulative[-1]
        k = int(np.searchsorted(cumulative, target))
        below = cumulative[k - 1] if k > 0 else 0
        inside = (target - below) / max(self.fine[k], 1)  # linear within the fine bin
        return float(self._mass((k + inside) / self.FINE))

    def summary(self):
        """(entries inside the range, median, half the width of the central 68%),
        or None while the range is empty. Computed once per change."""
        if self._summary is None:
            inside = int(self.fine.sum())
            if inside == 0:
                return None
            half_width = 0.5 * (self.quantile(0.84) - self.quantile(0.16))
            self._summary = (inside, self.quantile(0.5), half_width)
        return self._summary


def quick_masses(scene, count, field_strength, rng):
    """Masses of `count` more events for the histogram, by the cheapest way that
    gives the same distribution as the events shown.

    Recorded events: their masses come straight from the data (no tracking);
    only opposite-charge pairs are entered, so fewer than `count` may come
    back. Generated events: muons from the collision are not followed
    through the detector and the soft particles of the busy collision are
    not made, since neither changes the measured mass (generate(quick=True));
    electrons, taus and strange particles are followed as usual, because
    their measurement depends on what happens to them on the way.
    """
    if scene.startswith("Real data"):
        from physengine.modules import particle_opendata as data

        momenta, charges = data.load(scene)
        if not len(momenta):
            raise ValueError(f"no usable events in {data.path(scene).name}")
        rows = rng.integers(len(momenta), size=count)
        rows = rows[data.opposite_sign(charges[rows])]
        return [float(m) for m in mass(momenta[rows, 0] + momenta[rows, 1])]
    out = []
    for _ in range(count):
        out += histogram_masses(generate(scene, field_strength, rng, quick=True))
    return out


def histogram_masses(event):
    """The masses of an event that go into the scene's histogram."""
    if not event.in_histogram:
        return []
    return [m for _, m in event.masses]


def _real_event(scene, event, field_strength, rng):
    """A recorded event (particle_opendata.py): its two muons, followed through the
    detector. Only what was measured is stated: the two momenta, the charges
    and the mass they combine to."""
    from physengine.modules import particle_opendata as data

    momenta, charges = data.load(scene)
    if not len(momenta):
        raise ValueError(f"no usable events in {data.path(scene).name}")
    k = int(rng.integers(len(momenta)))
    names = ["mu+" if q > 0 else "mu-" for q in charges[k]]
    for name, p4 in zip(names, momenta[k], strict=True):
        _produce(event, name, p4, np.zeros(3), 0.0, field_strength, rng)
    m = float(mass(momenta[k][0] + momenta[k][1]))
    event.real = True
    event.beam_energy = data.BEAM_ENERGY
    pair = " ".join(sorted(names))  # "mu+ mu-", "mu+ mu+" or "mu- mu-"
    event.masses = [(f"{pair}, as measured by CMS", m)]
    if names[0] == names[1]:
        event.in_histogram = False
        event.headline = (
            f"Recorded by CMS in 2011: two muons of the same charge ({pair}),"
            f" combined mass {m:.4g} GeV"
        )
        event.note = (
            "Two muons of the same charge cannot both come from one neutral particle:"
            " this pair is not entered in the histogram of mu+ mu- pairs"
        )
    else:
        event.headline = (
            f"Recorded by CMS in 2011: a mu+ and a mu- with a combined mass of {m:.4g} GeV"
        )


def generate(scene, field_strength, rng, quick=False):
    """One event of `scene` in a solenoid of `field_strength` tesla.

    quick: only the masses are wanted (see quick_masses); the event is then
    not complete enough to be displayed.
    """
    event = Event()
    if scene.startswith("Real data"):
        _real_event(scene, event, field_strength, rng)
        return event
    pairs = {
        "Z -> mu+ mu-": ("Z", ("mu-", "mu+")),
        "Z -> e+ e-": ("Z", ("e-", "e+")),
        "Z -> tau+ tau-": ("Z", ("tau-", "tau+")),
        "J/psi -> mu+ mu-": ("J/psi", ("mu-", "mu+")),
        "Busy collision": ("Z", ("mu-", "mu+")),
    }
    if scene in pairs:
        name, daughters = pairs[scene]
        parent = _resonance(name, rng)
        _lepton_pair(event, parent, daughters, field_strength, rng, name == "Z", quick)
        event.bosons.append(
            {"name": name, "p4": parent, "daughters": daughters, "from": "the collision"}
        )
        event.mechanism = name
        event.headline = (
            f"Generated: a {name} (mass {float(mass(parent)):.4g} GeV) decays at once"
            f" -> {daughters[0]} {daughters[1]}"
        )
        if scene == "Busy collision" and not quick:
            _soft_particles(event, field_strength, rng)
    elif scene == "Strange particles":
        _strange(event, field_strength, rng)
        made = [t.name for t in event.tracks if t.parent < 0]
        event.mechanism = "soft"
        event.headline = (
            f"Generated: {made.count('K0S')} K0S and {made.count('Lambda')} Lambda"
            " leave the collision point"
        )
    elif scene == "Higgs -> 4 leptons":
        _higgs(event, field_strength, rng, quick)
        event.mechanism = "H"
    elif scene in CHAINS:
        # These scenes measure their masses themselves (_measure), and set the note.
        _chain(scene, event, field_strength, rng, quick)
        event.mechanism = CHAINS[scene][0]
        return event
    elif scene == "Higgs -> 2 photons":
        _higgs_photons(event, field_strength, rng, quick)
        return event
    elif scene == "e+ e- -> Z (collider scan)":
        _z_scan(event, field_strength, rng, quick)
        return event
    else:
        raise ValueError(f"unknown scene {scene}")
    event.masses = reconstruct(scene, event, field_strength, rng)
    if not event.masses and scene != "Strange particles":
        event.note = (
            "No mass measured: a particle needed for it left too close to the beam line"
            " for the tracker"
        )
    return event


def lever_arm(start, momentum, charge, field_strength):
    """Projected length L [m] over which the tracker sees a charged particle that
    starts at `start` with three-momentum `momentum` (see the module docstring)."""
    start = np.asarray(start, dtype=float)
    pt = float(np.hypot(momentum[0], momentum[1]))
    r0_squared = float(start[0] ** 2 + start[1] ** 2)
    inside = r0_squared < TRACKER_RADIUS**2 and abs(start[2]) < TRACKER_HALF_LENGTH
    if not inside or pt <= 0.0 or charge == 0 or field_strength <= 0.0:
        return 0.0
    # Along the initial direction across the field, to the outer cylinder ...
    along = float(start[0] * momentum[0] + start[1] * momentum[1]) / pt
    length = -along + np.sqrt(along * along + TRACKER_RADIUS**2 - r0_squared)
    # ... or to the end of the tracker, whichever comes first.
    if momentum[2] != 0.0:
        to_end = (np.sign(momentum[2]) * TRACKER_HALF_LENGTH - start[2]) * pt / momentum[2]
        length = min(length, float(to_end))
    return float(min(length, 2.0 * pt / (K * abs(charge) * field_strength)))


def momentum_resolution(pt, p, m, lever, field_strength, charge=1):
    """Relative error sigma(p_T) / p_T of the tracker for transverse momentum pt
    [GeV] (total momentum p, mass m) measured over the projected length
    `lever` [m] in `field_strength` tesla, or None if it cannot be measured
    (see the module docstring)."""
    if lever <= 0.0 or field_strength <= 0.0 or pt <= 0.0:
        return None
    layers = int(TRACKER_LAYERS * lever / TRACKER_RADIUS + 1e-9)
    if layers < MIN_LAYERS:
        return None
    bend = K * abs(charge) * field_strength  # p_T = bend * R
    points = POINT_RESOLUTION * pt / (bend * lever**2) * np.sqrt(720.0 / (layers + 4))
    beta = p / np.hypot(p, m)
    thickness = lever / (TRACKER.x0 / 100.0) / (pt / p)  # path / X0 = L / (X0 cos(lambda))
    scattering = 0.016 * abs(charge) / (bend * lever * beta) * np.sqrt(thickness)
    return float(np.hypot(points, scattering))


def measured(track, field_strength, rng):
    """Three-momentum of a charged track as the tracker would measure it, or None
    if it cannot (see the module docstring): the true momentum where the
    track starts, with its curvature (1 / p_T) smeared by the resolution."""
    particle = PARTICLES[track.name]
    p = track.p4[1:]
    pt, total = float(np.hypot(p[0], p[1])), float(np.linalg.norm(p))
    lever = lever_arm(track.points[0], p, particle.charge, field_strength)
    sigma = momentum_resolution(pt, total, particle.mass, lever, field_strength, particle.charge)
    if sigma is None:
        return None
    # A curvature measured with the wrong sign (a huge error) keeps its size.
    curvature = max(abs(1.0 + sigma * rng.normal()), 0.05)
    return p / curvature


def vertex_pairs(event):
    """Pairs of opposite-sign tracks from the decay of one particle away from the
    collision point: (index of the parent, track, track)."""
    by_parent = {}
    for t in event.tracks:
        if t.origin == "decay" and PARTICLES[t.name].charge != 0:
            by_parent.setdefault(t.parent, []).append(t)
    return [
        (parent, pair[0], pair[1])
        for parent, pair in by_parent.items()
        if len(pair) == 2 and PARTICLES[pair[0].name].charge == -PARTICLES[pair[1].name].charge
    ]


def visible_tracks(event):
    """The particles a detector could see of an event's decays, each counted once:
    every particle that did not decay, except neutrinos, and except what a
    particle on this list turned into on its way through the material (the
    photons an electron radiated, the e+ e- pair a photon converted to,
    and whatever came of those)."""
    return [
        t for t in event.tracks
        if t.end != "decay"
        and PARTICLES[t.name].kind != "neutrino"
        and t.origin not in ("radiated", "conversion")
    ]  # fmt: skip


def reconstruct(scene, event, field_strength, rng):
    """Invariant masses measured in the event: (label, mass) pairs.

    Lepton pairs: every opposite-sign pair of muons (or electrons) from the
    collision point, each taken with its own mass. Strange particles: every
    pair of opposite-sign tracks from a common vertex away from the
    collision point, both taken as pions (so the pairs from Lambda -> p pi do
    not peak: they spread from 0.32 to 0.56 GeV, under and beside the K0_S
    peak). Z -> tau tau: the visible decay products of both taus
    (visible_tracks), which falls short of the Z mass by what the neutrinos
    carry off; charged ones as the tracker measures them, neutral ones with
    their true momenta. A particle the tracker cannot measure (see the
    module docstring) is left out of the visible mass, and gives no entry
    at all where it is one of a pair or of the four leptons.
    """
    tracks = event.tracks
    out = []
    if scene == "Strange particles":
        m_pi = PARTICLES["pi+"].mass
        for _, a, b in vertex_pairs(event):
            momenta = [measured(t, field_strength, rng) for t in (a, b)]
            if all(q is not None for q in momenta):
                total = sum(four_momentum(m_pi, q) for q in momenta)
                out.append(("pi+ pi-", float(mass(total))))
        return out
    if scene == "Z -> tau+ tau-":
        total = np.zeros(4)
        for t in visible_tracks(event):
            q = measured(t, field_strength, rng) if PARTICLES[t.name].charge else t.p4[1:]
            if q is not None:
                total = total + four_momentum(PARTICLES[t.name].mass, q)
        if total[0] > 0.0:
            out.append(("visible", float(mass(total))))
        return out
    if scene == "Higgs -> 4 leptons":
        leptons = [i for i, t in enumerate(tracks) if t.parent < 0]
        momenta = [lepton_measured(event, i, field_strength, rng) for i in leptons]
        if len(leptons) == 4 and all(q is not None for q in momenta):
            out.append(("4 leptons", float(mass(sum(momenta)))))
        return out
    flavour = "e" if scene == "Z -> e+ e-" else "mu"
    leptons = [
        (t, lepton_measured(event, i, field_strength, rng))
        for i, t in enumerate(tracks)
        if t.parent < 0 and t.name in (f"{flavour}-", f"{flavour}+")
    ]
    for k, (a, qa) in enumerate(leptons):
        for b, qb in leptons[k + 1 :]:
            if a.name != b.name and qa is not None and qb is not None:
                out.append((f"{flavour}+ {flavour}-", float(mass(qa + qb))))
    return out


def lepton_measured(event, index, field_strength, rng):
    """Measured four-momentum of the lepton that is track `index`, or None if it
    cannot be measured.

    A muon: from the tracker (measured()). An electron: its direction taken
    as exact, its energy from the calorimeter, which also catches the
    photons it radiated on the way: the sum of those showers, with the
    calorimeter's resolution sqrt((ECAL_A / sqrt(E))^2 + ECAL_C^2) (the form
    and size of the CMS crystal calorimeter's; the noise term is left out).
    This too is a smearing of true values, not a simulation of the
    calorimeter's signals.
    """
    track = event.tracks[index]
    particle = PARTICLES[track.name]

    def from_tracker():
        q = measured(track, field_strength, rng)
        return None if q is None else four_momentum(particle.mass, q)

    if particle.kind != "electron":
        return from_tracker()
    # The electron's own shower and those of everything it radiated: its
    # photons, and the pairs those converted to.
    family = {index}
    for i, t in enumerate(event.tracks):  # parents always come before children
        if t.parent in family and t.origin in ("radiated", "conversion"):
            family.add(i)
    energy = sum(
        event.tracks[i].shower.energy for i in family if event.tracks[i].shower is not None
    )
    if energy <= 0.0:
        return from_tracker()
    sigma = np.sqrt((ECAL_A / np.sqrt(energy)) ** 2 + ECAL_C**2)
    energy *= max(1.0 + sigma * rng.normal(), 0.05)
    direction = track.p4[1:] / np.linalg.norm(track.p4[1:])
    return np.array([energy, *(energy * direction)])


def family_of(event, index):
    """Indices of track `index` and everything that came from it."""
    family = {index}
    for i, t in enumerate(event.tracks):  # parents always come before children
        if t.parent in family:
            family.add(i)
    return family


def photon_measured(event, index, rng):
    """Measured four-momentum of the photon that is track `index`, or None if
    nothing of it reached the calorimeter (no shower in its family).

    Its true direction, and its true energy smeared by the calorimeter's
    resolution sqrt((ECAL_A / sqrt(E))^2 + ECAL_C^2) (as for electrons,
    lepton_measured). A photon that converted in the tracker still counts as
    the photon: its pair's showers are taken as its own.
    """
    track = event.tracks[index]
    if not any(event.tracks[i].shower is not None for i in family_of(event, index)):
        return None
    energy = float(track.p4[0])
    sigma = np.sqrt((ECAL_A / np.sqrt(energy)) ** 2 + ECAL_C**2)
    return track.p4 * max(1.0 + sigma * rng.normal(), 0.05)


def measured_to_kink(track, field_strength, rng):
    """Three-momentum of a charged track that decayed in flight, as the tracker
    measures it over the segment from its start to the kink (its decay
    point), or None if that segment cannot be measured.

    As measured() (true momentum at the start, curvature smeared by the
    resolution), with the projected length L of the fit cut at the kink:
    the largest distance across the field from the start reached by the
    points of the path inside the tracker, and at most the lever arm of an
    undecayed track (lever_arm). The rule for fewer than MIN_LAYERS layers
    is momentum_resolution's.
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


def _measure(event, index, label, field_strength, rng):
    """((label, mass), "") of everything visible that came from track `index`,
    as measured, or (None, why not).

    The visible products: all descendants that did not decay, except
    neutrinos and except what a product turned into in the material (as in
    visible_tracks), each with its own mass. Charged ones by the tracker
    model (measured), electrons by lepton_measured, photons by
    photon_measured, neutral hadrons with their true momenta. A charged
    particle that decayed in flight is measured itself over its path to the
    kink (measured_to_kink), and what came of it is left out; if that
    segment has too few layers, or any product cannot be measured, there is
    no mass. A neutral one that decayed (the J/psi) counts through its products.
    """
    tracks = event.tracks
    parent = tracks[index]
    family = family_of(event, index) - {index}
    final, kinked, covered = [], {}, set()
    for i in sorted(family):  # parents always come before children
        t = tracks[i]
        if t.parent in covered:
            covered.add(i)
            continue
        if t.end == "decay":
            if PARTICLES[t.name].charge == 0:
                continue  # counted through its products
            q = measured_to_kink(t, field_strength, rng)
            if q is None:
                return None, (
                    f"No mass for the {parent.name}: its {t.name} decayed in flight"
                    " before the tracker could measure it"
                )
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
        return None, f"No mass for the {parent.name}: too little of its decay is visible"
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
            q4 = photon_measured(event, i, rng)
        else:
            q4 = t.p4  # neutral hadrons: true momenta
        if q4 is None:
            return None, (
                f"No mass for the {parent.name}: its {t.name} could not be measured"
                " (it left too close to the beam line, or started outside the tracker)"
            )
        total = total + q4
    if neutrinos:
        label = f"the visible products of the {parent.name}"
    return (label, float(mass(total))), ""
