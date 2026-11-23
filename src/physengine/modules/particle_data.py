"""Particles of the particle-physics module: masses, charges, lifetimes, decays.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
4). Review the physics: the numbers are from the Particle Data Group's
Review of Particle Physics, rounded to the digits given here. Edition per block:
- leptons, photon, light hadrons (pi, K, K0S, K0L, p, n, Lambda), the tau,
  the J/psi, the Z, the Higgs and the RHO/A1 resonance parameters: PDG 2022
  (R. L. Workman et al., Prog. Theor. Exp. Phys. 2022, 083C01) and its 2023
  update;
- charm and bottom mesons (D+, D0, Ds+, B+, B0, Bs: masses, lifetimes,
  branching fractions) and the mixing parameters (MIXING): PDG 2024
  (S. Navas et al., Phys. Rev. D 110, 030001). One D0 fraction (K- pi0 e+ nu)
  is an isospin estimate from a PDG 2024 value, not the PDG's own (see there).

Units: masses and widths in GeV, lifetimes as c tau in metres, charges in
units of e.

Decay modes, two kinds of table.

- Complete tables (the tau, light hadrons): each listed mode is drawn with
  exactly its measured (PDG) branching fraction. Where the listed modes do
  not add up to one, the remainder goes to one stand-in mode,
  `Particle.rest`, which is an approximation: it lumps all the rare modes
  left out into a single final state. Its size is given in the comments
  (2.3% for the tau, 0.3% or less for the others). Tables whose listed
  fractions add up to one within the rounding of the quoted digits (2 in
  10^4) have no stand-in and are normalised by that rounding.
- Partial tables (`partial=True`: the D and B mesons, the J/psi, the Z):
  most of the PDG's modes have particles that are not in this table (D**,
  eta, phi, rho, quarks, ...). No stand-in is made up for them. A decay is drawn
  among the listed modes in proportion to their PDG fractions, renormalised:
  mode i with probability B_i / sum_j B_j. The ratios between listed modes
  are the measured ones; the absolute fractions are not (each is larger by
  1 / sum_j B_j). `Particle.covered` is the share sum_j B_j, given in the
  comments and shown by the Custom window.

Charge-conjugate particles get the mirrored table (`_conjugate`).

How the momenta of a decay are shared out (phase space, a matrix element, a
resonance) is in particle_events.decay_momenta.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Particle:
    name: str
    mass: float  # GeV
    charge: int
    ctau: float | None = None  # c tau [m]; None: stable here
    width: float = 0.0  # GeV: for resonances that decay at once (Z, J/psi)
    kind: str = "hadron"  # how the detector treats it, see particle_tracks
    color: tuple = (0.8, 0.8, 0.8)
    # ((branching fraction, (daughter names)), ...)
    decays: tuple = field(default_factory=tuple)
    # Daughters of the stand-in mode that takes the fraction not listed.
    rest: tuple = ()
    # True: only part of the PDG's modes are listed (module docstring).
    partial: bool = False

    def covered(self):
        """Share of all decays that the listed modes make up (their PDG fractions summed)."""
        return float(sum(b for b, _ in self.decays))

    def modes(self):
        """(probabilities, daughter-name tuples) to draw a decay from.

        Complete table: the listed modes keep their branching fractions; the
        remainder, if any, is the stand-in mode `rest` (last in the lists).
        Partial table: the listed fractions renormalised to one. See the
        module docstring.
        """
        listed = sum(b for b, _ in self.decays)
        probabilities = [b for b, _ in self.decays]
        daughters = [d for _, d in self.decays]
        if self.partial:
            if not self.decays or self.rest or listed > 1.0 + ROUNDING:
                raise ValueError(f"{self.name}: a partial table needs modes, no stand-in, sum <= 1")
            return [b / listed for b in probabilities], daughters
        if self.rest:
            if listed >= 1.0:
                raise ValueError(f"{self.name}: listed fractions exceed one")
            return probabilities + [1.0 - listed], daughters + [self.rest]
        if not self.decays or abs(1.0 - listed) > ROUNDING:
            raise ValueError(f"{self.name}: the decay table is not complete")
        return [b / listed for b in probabilities], daughters


ROUNDING = 2.0e-4  # largest |1 - sum| accepted as rounding of the quoted fractions

_ANTI = {"e-": "e+", "mu-": "mu+", "tau-": "tau+", "pi-": "pi+", "K-": "K+", "p": "pbar"}
_ANTI.update({"n": "nbar", "D+": "D-", "D0": "D0bar", "Ds+": "Ds-"})
_ANTI.update({"B+": "B-", "B0": "B0bar", "Bs": "Bsbar"})
_ANTI.update({"D*+": "D*-", "D*0": "D*0bar", "Ds*+": "Ds*-"})
_ANTI.update({v: k for k, v in _ANTI.items()})


def _conjugate(particle):
    """The antiparticle: opposite charge, every daughter replaced by its antiparticle."""
    swap = lambda names: tuple(_ANTI.get(n, n) for n in names)  # noqa: E731
    return replace(
        particle,
        name=_ANTI[particle.name],
        charge=-particle.charge,
        decays=tuple((b, swap(d)) for b, d in particle.decays),
        rest=swap(particle.rest),
    )


# Display colours (convention of event displays: muons red, electrons green,
# photons yellow, charged hadrons bluish white, neutral hadrons blue).
_MU, _E, _GAMMA = (1.0, 0.25, 0.2), (0.3, 1.0, 0.4), (1.0, 0.9, 0.3)
_PI, _K, _P, _N, _NU = (
    (0.6, 0.85, 1.0),
    (0.75, 0.6, 1.0),
    (1.0, 0.6, 0.8),
    (0.35, 0.5, 1.0),
    (0.5, 0.5, 0.5),
)
_HEAVY = (1.0, 0.8, 0.45)  # charged charm and bottom mesons

_LIST = [
    # --- leptons and the photon
    Particle("e-", 0.000510999, -1, kind="electron", color=_E),
    Particle("e+", 0.000510999, +1, kind="electron", color=_E),
    Particle("mu-", 0.1056584, -1, ctau=658.64, kind="muon", color=_MU),
    Particle("mu+", 0.1056584, +1, ctau=658.64, kind="muon", color=_MU),
    Particle("nu", 0.0, 0, kind="neutrino", color=_NU),  # any neutrino or antineutrino
    Particle("gamma", 0.0, 0, kind="photon", color=_GAMMA),
    # tau: c tau = 87.03 micrometres. PDG fit values (K0 excluded from the
    # pion modes; pi- K0bar is half K0_S, half K0_L). Listed: 97.7%. The
    # stand-in for the other 2.3% (modes with kaon pairs, K pi pi, eta, omega
    # pi pi0, five pions, ...) is one of the larger of them, pi- pi+ pi- 2 pi0 nu.
    Particle(
        "tau-",
        1.77686,
        -1,
        ctau=87.03e-6,
        kind="hadron",
        color=(1.0, 0.6, 0.2),
        decays=(
            (0.1739, ("mu-", "nu", "nu")),
            (0.1782, ("e-", "nu", "nu")),
            (0.1082, ("pi-", "nu")),
            (0.00696, ("K-", "nu")),
            (0.2549, ("pi-", "pi0", "nu")),
            (0.00433, ("K-", "pi0", "nu")),
            (0.0926, ("pi-", "pi0", "pi0", "nu")),
            (0.0104, ("pi-", "pi0", "pi0", "pi0", "nu")),
            (0.0931, ("pi-", "pi+", "pi-", "nu")),
            (0.0462, ("pi-", "pi+", "pi-", "pi0", "nu")),
            (0.00419, ("pi-", "K0S", "nu")),
            (0.00419, ("pi-", "K0L", "nu")),
        ),
        rest=("pi-", "pi+", "pi-", "pi0", "pi0", "nu"),
    ),
    # --- light hadrons
    # pi+-: c tau = 7.8045 m; mu nu 99.98770%, e nu 0.01230% (both including
    # the radiative decays).
    Particle(
        "pi-",
        0.13957039,
        -1,
        ctau=7.8045,
        color=_PI,
        decays=((0.999877, ("mu-", "nu")), (0.000123, ("e-", "nu"))),
    ),
    # pi0: mean life 8.43e-17 s, c tau = 25.3 nm (decays where it is made);
    # gamma gamma 98.823%, e+ e- gamma 1.174% (the Dalitz decay, drawn here
    # from phase space: the real one favours small e+ e- masses).
    Particle(
        "pi0",
        0.1349768,
        0,
        ctau=25.3e-9,
        kind="neutral",
        color=_N,
        decays=((0.98823, ("gamma", "gamma")), (0.01174, ("e+", "e-", "gamma"))),
    ),
    # K+-: c tau = 3.711 m. The six modes of the PDG fit: 99.995%.
    Particle(
        "K-",
        0.493677,
        -1,
        ctau=3.711,
        color=_K,
        decays=(
            (0.6356, ("mu-", "nu")),
            (0.2067, ("pi-", "pi0")),
            (0.05583, ("pi-", "pi-", "pi+")),
            (0.0507, ("pi0", "e-", "nu")),
            (0.03352, ("pi0", "mu-", "nu")),
            (0.01760, ("pi-", "pi0", "pi0")),
        ),
    ),
    # K0_S: c tau = 2.6844 cm. Listed: 99.89%. The stand-in for the other
    # 0.11% (pi e nu and pi mu nu, both charges) is pi+ e- nu.
    Particle(
        "K0S",
        0.497611,
        0,
        ctau=0.026844,
        kind="neutral",
        color=_N,
        decays=((0.6920, ("pi+", "pi-")), (0.3069, ("pi0", "pi0"))),
        rest=("pi+", "e-", "nu"),
    ),
    # K0_L: c tau = 15.34 m: reaches the calorimeter.
    Particle("K0L", 0.497611, 0, kind="neutral", color=_N),
    Particle("p", 0.93827209, +1, color=_P),
    Particle("pbar", 0.93827209, -1, color=_P),
    Particle("n", 0.93956542, 0, kind="neutral", color=_N),
    Particle("nbar", 0.93956542, 0, kind="neutral", color=_N),  # from Ds+ -> p nbar
    # Lambda: c tau = 7.89 cm. Listed: 99.7%. The stand-in for the other 0.3%
    # (n gamma, p pi- gamma, p e- nu) is the largest of them, n gamma.
    Particle(
        "Lambda",
        1.115683,
        0,
        ctau=0.0789,
        kind="neutral",
        color=_N,
        decays=((0.639, ("p", "pi-")), (0.358, ("n", "pi0"))),
        rest=("n", "gamma"),
    ),
    # --- charm and bottom mesons. PDG 2024 (S. Navas et al., Phys. Rev. D 110,
    # 030001): every mode whose particles are all in this table, with a
    # fraction of at least 1e-3, plus the leptonic modes (l nu, tau nu) and
    # the J/psi K modes of the B mesons down to 1e-4. A K0 (K0bar) is half
    # K0_S and half K0_L. These tables are partial (see the module
    # docstring). Width 0: they fly before they decay. Decays to three or
    # more bodies are phase space (particle_events.decay_momenta), except
    # the three-body modes with resonant sub-channels in ISOBARS.
    # D+: mass 1869.66 MeV, mean life 1033 fs, c tau = 309.8 micrometres.
    # Listed: 67.77% of all decays.
    Particle(
        "D+", 1.86966, +1, ctau=309.8e-6, color=_HEAVY,
        partial=True,
        decays=(
            (0.0436, ("K0S", "e+", "nu")),  # K0bar e+ nu 8.72%, half
            (0.0436, ("K0L", "e+", "nu")),  # K0bar e+ nu 8.72%, half
            (0.0438, ("K0S", "mu+", "nu")),  # K0bar mu+ nu 8.76%, half
            (0.0438, ("K0L", "mu+", "nu")),  # K0bar mu+ nu 8.76%, half
            (0.0402, ("K-", "pi+", "e+", "nu")),
            (0.0365, ("K-", "pi+", "mu+", "nu")),
            (0.00372, ("pi0", "e+", "nu")),
            (0.0035, ("pi0", "mu+", "nu")),
            (0.00249, ("pi-", "pi+", "e+", "nu")),
            (0.0012, ("tau+", "nu")),
            (0.000374, ("mu+", "nu")),
            (0.01562, ("K0S", "pi+")),
            (0.0146, ("K0L", "pi+")),
            (0.0938, ("K-", "pi+", "pi+")),
            (0.0736, ("K0S", "pi+", "pi0")),
            (0.0625, ("K-", "pi+", "pi+", "pi0")),
            (0.031, ("K0S", "pi+", "pi+", "pi-")),
            (0.0289, ("K0S", "pi+", "pi0", "pi0")),
            (0.0057, ("K-", "pi+", "pi+", "pi+", "pi-")),
            (0.0055, ("K0S", "pi+", "pi0", "pi0", "pi0")),
            (0.00495, ("K-", "pi+", "pi+", "pi0", "pi0")),
            (0.00254, ("K+", "K0S", "K0S")),
            (0.001247, ("pi+", "pi0")),
            (0.00327, ("pi+", "pi+", "pi-")),
            (0.01165, ("pi+", "pi+", "pi-", "pi0")),
            (0.00417, ("pi+", "pi0", "pi0", "pi0")),
            (0.0019, ("pi+", "pi0", "pi0", "pi0", "pi0")),
            (0.0107, ("pi+", "pi+", "pi-", "pi0", "pi0")),
            (0.00166, ("pi+", "pi+", "pi+", "pi-", "pi-")),
            (0.00342, ("pi+", "pi+", "pi-", "pi0", "pi0", "pi0")),
            (0.00234, ("pi+", "pi+", "pi+", "pi-", "pi-", "pi0")),
            (0.00304, ("K0S", "K+")),
            (0.00321, ("K0L", "K+")),
            (0.00507, ("K0S", "K+", "pi0")),
            (0.00968, ("K+", "K-", "pi+")),
            (0.00662, ("K+", "K-", "pi+", "pi0")),
            (0.0027, ("K0S", "K0S", "pi+")),
            (0.00134, ("K0S", "K0S", "pi+", "pi0")),
            (0.00189, ("K+", "K0S", "pi+", "pi-")),
            (0.00227, ("K0S", "K-", "pi+", "pi+")),
        ),
    ),
    # D0: mass 1864.84 MeV, mean life 410.3 fs, c tau = 123.01 micrometres.
    # Listed: 71.98%. D0-D0bar mixing: see MIXING.
    Particle(
        "D0", 1.86484, 0, ctau=123.01e-6, kind="neutral", color=_N,
        partial=True,
        decays=(
            (0.03549, ("K-", "e+", "nu")),
            (0.0341, ("K-", "mu+", "nu")),
            # K- pi0 e+ nu: an isospin estimate, not a measurement. The K pi
            # pair (c -> s e+ nu, the s quark joining the ubar spectator: the
            # K*(892)- and the K pi S-wave) is an I = 1/2 state in the isospin
            # limit; the Clebsch-Gordan coefficients of I = 1/2, I3 = -1/2 ->
            # K pi give |K- pi0|^2 = 1/3, |Kbar0 pi-|^2 = 2/3, so
            #   B(K- pi0 e+ nu) = 1/2 B(Kbar0 pi- e+ nu) = 1/2 x 1.44% = 0.72%,
            # with B(Kbar0 pi- e+ nu) = (1.44 +- 0.04)% from PDG 2024
            # (scratchpad pdg/charm.txt:704, "K 0 pi- e+ nue (1.44 +- 0.04) %").
            # The PDG's own value, (1.6 +1.3 -0.5)% (charm.txt:702-703), was listed here
            # before; it is a single old measurement whose error is far larger
            # than the isospin relation's, and its central value breaks the
            # relation with the precise Kbar0 pi- e+ nu value.
            (0.0072, ("K-", "pi0", "e+", "nu")),
            (0.0072, ("K0S", "pi-", "e+", "nu")),  # K0bar pi- e+ nu 1.44%, half
            (0.0072, ("K0L", "pi-", "e+", "nu")),  # K0bar pi- e+ nu 1.44%, half
            (0.00291, ("pi-", "e+", "nu")),
            (0.00267, ("pi-", "mu+", "nu")),
            (0.00145, ("pi-", "pi0", "e+", "nu")),
            (0.03947, ("K-", "pi+")),
            (0.0124, ("K0S", "pi0")),
            (0.00976, ("K0L", "pi0")),
            (0.028, ("K0S", "pi+", "pi-")),
            (0.144, ("K-", "pi+", "pi0")),
            (0.0091, ("K0S", "pi0", "pi0")),
            (0.0126, ("K0L", "pi0", "pi0")),
            (0.0076, ("K0S", "pi0", "pi0", "pi0")),
            (0.0822, ("K-", "pi+", "pi+", "pi-")),
            (0.052, ("K0S", "pi+", "pi-", "pi0")),
            (0.0886, ("K-", "pi+", "pi0", "pi0")),
            (0.0095, ("K-", "pi+", "pi0", "pi0", "pi0")),
            (0.0127, ("K0S", "pi+", "pi-", "pi0", "pi0")),
            (0.043, ("K-", "pi+", "pi+", "pi-", "pi0")),
            (0.00266, ("K0S", "pi+", "pi+", "pi-", "pi-")),
            (0.00442, ("K0S", "K+", "K-")),
            (0.001454, ("pi+", "pi-")),
            (0.0149, ("pi+", "pi-", "pi0")),
            (0.00756, ("pi+", "pi+", "pi-", "pi-")),
            (0.01002, ("pi+", "pi-", "pi0", "pi0")),
            (0.00346, ("pi+", "pi+", "pi-", "pi-", "pi0")),
            (0.00153, ("pi+", "pi-", "pi0", "pi0", "pi0")),
            (0.0048, ("pi+", "pi+", "pi-", "pi-", "pi0", "pi0")),
            (0.00408, ("K+", "K-")),
            (0.0033, ("K0S", "K-", "pi+")),
            (0.00217, ("K0S", "K+", "pi-")),
            (0.00342, ("K+", "K-", "pi0")),
            (0.00247, ("K+", "K-", "pi+", "pi-")),
            (0.00132, ("K0S", "K-", "pi+", "pi0")),
            (0.0031, ("K+", "K-", "pi+", "pi-", "pi0")),
        ),
    ),
    # Ds+: mass 1968.35 MeV, mean life 501.2 fs, c tau = 150.3 micrometres.
    # Listed: 35.28% (most of the rest has an eta, eta' or phi).
    Particle(
        "Ds+", 1.96835, +1, ctau=150.3e-6, color=_HEAVY,
        partial=True,
        decays=(
            (0.0536, ("tau+", "nu")),
            (0.00535, ("mu+", "nu")),
            (0.0017, ("K0S", "e+", "nu")),  # K0bar e+ nu 3.4e-3, half
            (0.0017, ("K0L", "e+", "nu")),  # K0bar e+ nu 3.4e-3, half
            (0.0145, ("K+", "K0S")),
            (0.0149, ("K+", "K0L")),
            (0.0537, ("K+", "K-", "pi+")),
            (0.0147, ("K+", "K0S", "pi0")),
            (0.0071, ("K0S", "K0S", "pi+")),
            (0.0153, ("K0S", "K-", "pi+", "pi+")),
            (0.0095, ("K+", "K0S", "pi+", "pi-")),
            (0.0066, ("K+", "K-", "pi+", "pi+", "pi-")),
            (0.055, ("K+", "K-", "pi+", "pi0")),
            (0.0108, ("pi+", "pi+", "pi-")),
            (0.0052, ("pi+", "pi0", "pi0")),
            (0.0078, ("pi+", "pi+", "pi+", "pi-", "pi-")),
            (0.049, ("pi+", "pi+", "pi+", "pi-", "pi-", "pi0")),
            (0.00109, ("K0S", "pi+")),
            (0.0062, ("K+", "pi+", "pi-")),
            (0.00538, ("K0S", "pi+", "pi0")),
            (0.0028, ("K0S", "pi+", "pi+", "pi-")),
            (0.0097, ("K+", "pi+", "pi-", "pi0")),
            (0.00122, ("p", "nbar")),
        ),
    ),
    # B+: mass 5279.41 MeV, mean life 1.638 ps, c tau = 491.1 micrometres.
    # Listed: 33.79% (8.93% before the D* modes; most of the rest has a D**,
    # an eta, omega or rho, or many light hadrons).
    # The PDG's l nu modes are each lepton's own fraction (e or mu, not
    # their sum): entered once with e, once with mu.
    Particle(
        "B+", 5.27941, +1, ctau=491.1e-6, color=_HEAVY,
        partial=True,
        decays=(
            (0.0221, ("D0bar", "e+", "nu")),
            (0.0221, ("D0bar", "mu+", "nu")),
            (0.0077, ("D0bar", "tau+", "nu")),
            (0.00382, ("D-", "pi+", "e+", "nu")),
            (0.00382, ("D-", "pi+", "mu+", "nu")),
            (0.00173, ("D0bar", "pi+", "pi-", "e+", "nu")),
            (0.00173, ("D0bar", "pi+", "pi-", "mu+", "nu")),
            (0.000109, ("tau+", "nu")),
            (0.00461, ("D0bar", "pi+")),
            (0.0055, ("D0bar", "pi+", "pi+", "pi-")),
            (0.00107, ("D-", "pi+", "pi+")),
            (0.009, ("D0bar", "Ds+")),
            (0.000775, ("D0bar", "D+", "K0S")),  # D0bar D+ K0 1.55e-3, half
            (0.000775, ("D0bar", "D+", "K0L")),  # D0bar D+ K0 1.55e-3, half
            (0.00145, ("D0bar", "D0", "K+")),
            (0.00102, ("J/psi", "K+")),
            (0.00057, ("J/psi", "K0S", "pi+")),  # J/psi K0 pi+ 1.14e-3, half
            (0.00057, ("J/psi", "K0L", "pi+")),  # J/psi K0 pi+ 1.14e-3, half
            (0.00081, ("J/psi", "K+", "pi+", "pi-")),
            # Through a D* (added 2026-11-23). PDG 2024, scratchpad pdg/b.txt,
            # line of each value in the comment; same selection as above.
            (0.0553, ("D*0bar", "e+", "nu")),  # b.txt:347
            (0.0553, ("D*0bar", "mu+", "nu")),  # b.txt:347
            (0.00542, ("D*-", "pi+", "e+", "nu")),  # b.txt:366
            (0.00542, ("D*-", "pi+", "mu+", "nu")),  # b.txt:366
            (0.00517, ("D*0bar", "pi+")),  # b.txt:518
            (0.0103, ("D*0bar", "pi+", "pi+", "pi-")),  # b.txt:537
            (0.018, ("D*0bar", "pi-", "pi+", "pi+", "pi0")),  # b.txt:539
            (0.0057, ("D*0bar", "pi+", "pi+", "pi+", "pi-", "pi-")),  # b.txt:547
            (0.00135, ("D*-", "pi+", "pi+")),  # b.txt:490
            (0.015, ("D*-", "pi+", "pi+", "pi0")),  # b.txt:550
            (0.0026, ("D*-", "pi+", "pi+", "pi+", "pi-")),  # b.txt:551
            (0.0076, ("D0bar", "Ds*+")),  # b.txt:689-690
            (0.0082, ("D*0bar", "Ds+")),  # b.txt:691-692
            (0.0171, ("D*0bar", "Ds*+")),  # b.txt:693-694
            (0.00105, ("D*0bar", "D+", "K0S")),  # D*0bar D+ K0 2.1e-3, half (b.txt:705)
            (0.00105, ("D*0bar", "D+", "K0L")),
            (0.0019, ("D0bar", "D*+", "K0S")),  # D0bar D*+ K0 3.8e-3, half (b.txt:706)
            (0.0019, ("D0bar", "D*+", "K0L")),
            (0.0046, ("D*0bar", "D*+", "K0S")),  # D*0bar D*+ K0 9.2e-3, half (b.txt:707)
            (0.0046, ("D*0bar", "D*+", "K0L")),
            (0.00226, ("D*0bar", "D0", "K+")),  # b.txt:709
            (0.0063, ("D0bar", "D*0", "K+")),  # b.txt:710
            (0.0112, ("D*0bar", "D*0", "K+")),  # b.txt:711
            (0.00132, ("D*-", "D*+", "K+")),  # b.txt:722
        ),
    ),
    # B0: mass 5279.72 MeV, mean life 1.517 ps, c tau = 454.8 micrometres.
    # Listed: 30.07% (8.18% before the D* modes); l nu as for the B+.
    # B0-B0bar oscillation: see MIXING.
    Particle(
        "B0", 5.27972, 0, ctau=454.8e-6, kind="neutral", color=_N,
        partial=True,
        decays=(
            (0.0212, ("D-", "e+", "nu")),
            (0.0212, ("D-", "mu+", "nu")),
            (0.0099, ("D-", "tau+", "nu")),
            (0.00364, ("D0bar", "pi-", "e+", "nu")),
            (0.00364, ("D0bar", "pi-", "mu+", "nu")),
            (0.00145, ("D-", "pi+", "pi-", "e+", "nu")),
            (0.00145, ("D-", "pi+", "pi-", "mu+", "nu")),
            (0.00251, ("D-", "pi+")),
            (0.006, ("D-", "pi+", "pi+", "pi-")),
            (0.0072, ("D-", "Ds+")),
            (0.00107, ("D-", "D0", "K+")),
            (0.0004455, ("J/psi", "K0S")),  # J/psi K0 8.91e-4, half
            (0.0004455, ("J/psi", "K0L")),  # J/psi K0 8.91e-4, half
            (0.00115, ("J/psi", "K+", "pi-")),
            (0.000225, ("J/psi", "K0S", "pi+", "pi-")),  # J/psi K0 pi+ pi- 4.5e-4, half
            (0.000225, ("J/psi", "K0L", "pi+", "pi-")),  # J/psi K0 pi+ pi- 4.5e-4, half
            # Through a D* (added 2026-11-23). PDG 2024, scratchpad pdg/b.txt.
            (0.049, ("D*-", "e+", "nu")),  # b.txt:1804
            (0.049, ("D*-", "mu+", "nu")),  # b.txt:1804
            (0.00544, ("D*0bar", "pi-", "e+", "nu")),  # b.txt:1818
            (0.00544, ("D*0bar", "pi-", "mu+", "nu")),  # b.txt:1818
            (0.00266, ("D*-", "pi+")),  # b.txt:1877
            (0.015, ("D*-", "pi+", "pi0")),  # b.txt:1883-1884
            (0.00721, ("D*-", "pi+", "pi+", "pi-")),  # b.txt:1892
            (0.0176, ("D*-", "pi+", "pi+", "pi-", "pi0")),  # b.txt:1901
            (0.0047, ("D*-", "pi+", "pi+", "pi+", "pi-", "pi-")),  # b.txt:1902
            (0.0027, ("D*0bar", "pi+", "pi+", "pi-", "pi-")),  # b.txt:2160
            (0.008, ("D*-", "Ds+")),  # b.txt:1953
            (0.0074, ("D-", "Ds*+")),  # b.txt:1954
            (0.0177, ("D*-", "Ds*+")),  # b.txt:1955
            (0.0035, ("D-", "D*0", "K+")),  # b.txt:2166
            (0.00247, ("D*-", "D0", "K+")),  # b.txt:2167
            (0.0106, ("D*-", "D*0", "K+")),  # b.txt:2168
            (0.00405, ("D*-", "D*+", "K0S")),  # D*- D*+ K0 8.1e-3, half (b.txt:2172-2173)
            (0.00405, ("D*-", "D*+", "K0L")),
            (0.0012, ("D*0bar", "D*0", "K0S")),  # D*0bar D*0 K0 2.4e-3, half (b.txt:2180-2182)
            (0.0012, ("D*0bar", "D*0", "K0L")),
        ),
    ),
    # Bs (B_s0): mass 5366.93 MeV, mean life 1.520 ps, c tau = 455.7
    # micrometres. Listed: 3.86% (the PDG gives Ds- mu+ nu, not Ds- e+ nu).
    # Bs-Bsbar oscillation: see MIXING.
    Particle(
        "Bs", 5.36693, 0, ctau=455.7e-6, kind="neutral", color=_N,
        partial=True,
        decays=(
            (0.0231, ("Ds-", "mu+", "nu")),
            (0.00298, ("Ds-", "pi+")),
            (0.0061, ("Ds-", "pi+", "pi+", "pi-")),
            (0.0044, ("Ds+", "Ds-")),
            (0.00104, ("D0bar", "K-", "pi+")),
            (0.00079, ("J/psi", "K+", "K-")),
            (0.000202, ("J/psi", "pi+", "pi-")),
        ),
    ),
    # --- the vector charm mesons D*: strong or electromagnetic decays, at once
    # (c tau 0). Complete tables. PDG 2024, scratchpad pdg/ files:
    # - D*(2010)+: mass 2010.26 +- 0.05 MeV (charm.txt:1342), width 83.4 +- 1.8
    #   keV (charm.txt:1345): a mass from the Breit-Wigner of that width
    #   wherever a D*+ is made (particle_events.decay_momenta). D0 pi+ 67.7%,
    #   D+ pi0 30.7%, D+ gamma 1.6% (charm.txt:1351-1353); sum 100.0%.
    # - D*(2007)0: mass 2006.85 +- 0.05 MeV (charm.txt:1324). Its width is not
    #   measured: Gamma < 2.1 MeV at 90% CL (charm.txt:1326). Width 0 here (the
    #   nominal mass). D0 pi0 64.7%, D0 gamma 35.3% (charm.txt:1332-1333); sum
    #   100.0%. D0 e+ e- (3.91e-3, charm.txt:1334) is quoted outside that sum
    #   and is left out.
    # - Ds*+: mass 2112.2 +- 0.4 MeV (ds.txt:444); width not measured,
    #   Gamma < 1.9 MeV at 90% CL (ds.txt:447): width 0. Ds+ gamma 93.6%, Ds+
    #   pi0 5.77% (ds.txt:454-456). The other 0.63% is the stand-in Ds+ e+ e-,
    #   measured (6.7 +- 1.6)e-3 (ds.txt:457-458), from phase space.
    # Two-body decays are isotropic: the D*'s polarisation is not followed.
    Particle(
        "D*+", 2.01026, +1, ctau=0.0, width=83.4e-6, color=_HEAVY,
        decays=((0.677, ("D0", "pi+")), (0.307, ("D+", "pi0")), (0.016, ("D+", "gamma"))),
    ),
    Particle(
        "D*0", 2.00685, 0, ctau=0.0, kind="neutral", color=_N,
        decays=((0.647, ("D0", "pi0")), (0.353, ("D0", "gamma"))),
    ),
    Particle(
        "Ds*+", 2.1122, +1, ctau=0.0, color=_HEAVY,
        decays=((0.936, ("Ds+", "gamma")), (0.0577, ("Ds+", "pi0"))),
        rest=("Ds+", "e+", "e-"),
    ),
    # --- resonances: decay at once, mass drawn from a Breit-Wigner of this width
    # J/psi: e e 5.971%, mu mu 5.961%. Partial (listed: 11.93%; the rest is
    # hadrons through gluons): a J/psi from a B decays to one of the two,
    # half and half. The J/psi scenes choose mu mu themselves.
    Particle(
        "J/psi",
        3.0969,
        0,
        ctau=0.0,
        width=92.6e-6,
        kind="neutral",
        color=_N,
        partial=True,
        decays=((0.05961, ("mu+", "mu-")), (0.05971, ("e+", "e-"))),
    ),
    # Z: mass 91.1876, width 2.4952. Charged leptons: mu mu 3.3662%, e e
    # 3.3632%, tau tau 3.3696%. Partial
    # (listed: 10.10%; quarks and neutrinos are not listed).
    Particle(
        "Z",
        91.1876,
        0,
        ctau=0.0,
        width=2.4952,
        kind="neutral",
        color=_N,
        partial=True,
        decays=((0.033662, ("mu+", "mu-")), (0.033632, ("e+", "e-")), (0.033696, ("tau+", "tau-"))),
    ),
]

# The Higgs boson: mass 125.25 GeV, width 4 MeV (it decays at once). Only the
# decay shown by the module is listed.
_LIST.append(
    Particle("H", 125.25, 0, ctau=0.0, width=0.0041, kind="neutral", color=(1.0, 1.0, 1.0))
)

_CONJUGATED = ("tau-", "pi-", "K-", "D+", "D0", "Ds+", "B+", "B0", "Bs", "D*+", "D*0", "Ds*+")
_LIST += [_conjugate(p) for p in _LIST if p.name in _CONJUGATED]

PARTICLES = {p.name: p for p in _LIST}

# Flavour oscillation of the neutral mesons (particle_events.decay_flavour):
# name -> (x, y), x = Delta m / Gamma, y = Delta Gamma / (2 Gamma). No CP
# violation in mixing (|q/p| = 1). A meson made as P decays as its
# antiparticle with probability, at proper time t (u = Gamma t),
#     P(flip | t) = [cosh(y u) - cos(x u)] / (2 cosh(y u)),
# and time-integrated chi = (x^2 + y^2) / (2 (1 + x^2)). PDG 2024 (scratchpad
# pdg/ files):
# - B0: x_d = 0.7697 +- 0.0035 (b.txt:1470); y_d = 0 (Delta Gamma_d is
#   consistent with zero, not taken from the file). Gives chi_d = 0.1860,
#   the PDG's chi_d = 0.1860 +- 0.0011 (b.txt:1466).
# - Bs: x_s = 26.99 +- 0.09 (bs.txt:30); y_s = Delta Gamma_s tau / 2 =
#   0.083e12 s^-1 x 1.520e-12 s / 2 = 0.0631 (bs.txt:23, 21). Gives chi_s =
#   0.49932, the PDG's chi_s = 0.499318 +- 0.000005 (bs.txt:32).
# - D0: x = Delta m tau = 0.997e10 s^-1 x 410.3e-15 s = 0.00409 (charm.txt:462,
#   459), y = 1.394e-2 / 2 = 0.00697 (charm.txt:464): chi_D = 3.3e-5.
MIXING = {
    "B0": (0.7697, 0.0),
    "Bs": (26.99, 0.083 * 1.520 / 2.0),
    "D0": (0.997e10 * 410.3e-15, 1.394e-2 / 2.0),
}
MIXING.update({_ANTI[k]: v for k, v in list(MIXING.items())})

# Hadronic resonances the tau's decays go through (particle_events.decay_momenta):
# name -> (mass, width) [GeV]. rho(770): the charged one, as seen in tau
# decays. a1(1260): the PDG's estimates; its width is poorly known (250 to
# 600 MeV).
RHO = (0.77511, 0.1491)
A1 = (1.230, 0.420)

# Vector resonances inside the three-body decays of the D mesons (ISOBARS):
# name -> (Breit-Wigner mass, full width) [GeV]. They are not particles of
# the table (no tracks, no decay tables of their own): only their line shapes
# are used, in particle_events.isobar_decay. PDG 2024 summary tables (S. Navas
# et al., Phys. Rev. D 110, 030001), as text in scratchpad pdg/light.txt and
# pdg/strange.txt:
# - phi(1020): m = 1019.461 +- 0.016 MeV, Gamma = 4.249 +- 0.013 MeV
#   (light.txt:351-352).
# - K*(892)+-: hadroproduced, m = 891.67 +- 0.26 MeV, Gamma = 51.4 +- 0.8 MeV
#   (strange.txt:494, 497). The values from tau decays (895.5, 46.2 MeV,
#   strange.txt:495, 498) are not used: the D decays are hadronic.
# - K*(892)0: m = 895.55 +- 0.20 MeV, Gamma = 47.3 +- 0.5 MeV (strange.txt:496, 499).
# - rho(770)+-: the summary table gives only the rho0 (Breit-Wigner) values,
#   m = 775.26 +- 0.23 MeV, Gamma = 147.4 +- 0.8 MeV (light.txt:187-188); they
#   are used for the charged rho too (isospin). (RHO above is the older
#   PDG 2022 charged-rho value of the tau decays; it is left as it was.)
_PHI = (1.019461, 0.004249)
_KSTAR_CHARGED = (0.89167, 0.0514)
_KSTAR_NEUTRAL = (0.89555, 0.0473)
_RHO_770 = (0.77526, 0.1474)
RESONANCES = {
    "phi": _PHI,
    "K*+": _KSTAR_CHARGED, "K*-": _KSTAR_CHARGED,
    "K*0": _KSTAR_NEUTRAL, "K*0bar": _KSTAR_NEUTRAL,
    "rho+": _RHO_770, "rho-": _RHO_770, "rho0": _RHO_770,
}  # fmt: skip

# Isobar model of three-body decays (particle_events.isobar_decay): (parent,
# its daughters as listed in its table) -> (branching fraction of that mode,
# sub-channels). A sub-channel is (branching fraction of the parent's decay
# through it, resonance, the two daughters the resonance decays to); the
# third daughter is the spectator. The sub-channel fractions are the PDG's
# with the resonance's own decay to those two daughters included. PDG 2024,
# scratchpad pdg/ files:
# - Ds+ -> K+ K- pi+, (5.37 +- 0.10)% (ds.txt:161):
#   phi pi+, phi -> K+ K-, (2.21 +- 0.06)% (ds.txt:163);
#   K+ K*(892)0bar, K*0bar -> K- pi+, (2.58 +- 0.06)% (ds.txt:166-167).
#   The rest, 0.58% (f0(980) pi+ 1.11%, ds.txt:171, and others that
#   overlap through interference), is phase space.
# - D0 -> K- pi+ pi0, (14.4 +- 0.6)% (charm.txt:791):
#   K- rho+, (11.2 +- 0.7)% (charm.txt:792; rho+ -> pi+ pi0 is ~100%,
#   light.txt:192);
#   K*(892)- pi+, K*- -> K- pi0, (2.31 +0.40 -0.20)% (charm.txt:794-796);
#   K*(892)0bar pi0, K*0bar -> K- pi+, (1.95 +- 0.25)% (charm.txt:797-799).
#   These three add up to 15.46%, more than the mode itself: the PDG notes
#   (charm.txt:1451-1453, note [c]) that sub-fractions may not add up to the
#   mode because of interference. Without interference (see below) they
#   cannot all be kept; they are renormalised to the mode, keeping their
#   ratios: each is drawn with B_i / 15.46% of the K- pi+ pi0 decays, and
#   nothing is left for phase space (the non-resonant 1.15%, charm.txt:813,
#   and the K*(1430), K*(1680) and rho(1700) channels are not modelled).
# - D+ -> K- pi+ pi+, (9.38 +- 0.16)% (charm.txt:183):
#   K*(892)0bar pi+, K*0bar -> K- pi+, (1.04 +- 0.12)% (charm.txt:190-193).
#   The rest, 89%, is phase space: mostly the (K- pi+) S-wave (7.52%,
#   charm.txt:184), which is not a Breit-Wigner resonance.
# Stated approximation: the sub-channels are added as probabilities, not as
# amplitudes. The interference between them (which the PDG's own analyses
# include, and which makes the fractions not add up) is left out.
ISOBARS = {
    ("Ds+", ("K+", "K-", "pi+")): (
        0.0537, ((0.0221, "phi", ("K+", "K-")), (0.0258, "K*0bar", ("K-", "pi+"))),
    ),
    ("D0", ("K-", "pi+", "pi0")): (
        0.144,
        (
            (0.112, "rho+", ("pi+", "pi0")),
            (0.0231, "K*-", ("K-", "pi0")),
            (0.0195, "K*0bar", ("K-", "pi+")),
        ),
    ),
    ("D+", ("K-", "pi+", "pi+")): (0.0938, ((0.0104, "K*0bar", ("K-", "pi+")),)),
}  # fmt: skip
_ANTI_RESONANCE = {"phi": "phi", "K*-": "K*+", "K*0bar": "K*0", "rho+": "rho-"}
_ANTI_RESONANCE.update({v: k for k, v in _ANTI_RESONANCE.items()})
ISOBARS.update({
    (_ANTI[parent], tuple(_ANTI.get(d, d) for d in daughters)): (
        total,
        tuple(
            (b, _ANTI_RESONANCE[r], tuple(_ANTI.get(d, d) for d in pair))
            for b, r, pair in channels
        ),
    )
    for (parent, daughters), (total, channels) in list(ISOBARS.items())
})  # fmt: skip

# For the information panel: name in words, and what the particle is.
ABOUT = {
    "e-": ("electron", "The lightest charged particle; stable. Orbits atomic nuclei."),
    "e+": ("positron", "The electron's antiparticle: same mass, opposite charge."),
    "mu-": (
        "muon",
        "A heavy copy of the electron (207 times its mass). Lives 2.2 microseconds, long enough to cross the whole detector.",
    ),
    "mu+": ("antimuon", "The muon's antiparticle."),
    "nu": (
        "neutrino",
        "Nearly massless and uncharged; feels only the weak force, so it leaves the detector unseen. Its presence shows as missing energy.",
    ),
    "gamma": (
        "photon",
        "The particle of light, massless. Invisible to the tracker; showers in the electromagnetic calorimeter.",
    ),
    "tau-": (
        "tau lepton",
        "The heaviest copy of the electron (3477 times its mass). Lives 0.29 picoseconds: it decays within a millimetre.",
    ),
    "tau+": ("antitau", "The tau's antiparticle."),
    "pi+": (
        "charged pion",
        "The lightest charged hadron: an up quark and a down antiquark. The most common particle made in collisions.",
    ),
    "pi-": (
        "charged pion",
        "The lightest charged hadron: a down quark and an up antiquark. The most common particle made in collisions.",
    ),
    "pi0": ("neutral pion", "The lightest hadron. Decays to two photons almost where it is made (after 25 nanometres)."),
    "K+": ("charged kaon", "A hadron with a strange antiquark (and an up quark)."),
    "K-": ("charged kaon", "A hadron with a strange quark (and an up antiquark)."),
    "K0S": (
        "K-short",
        "A neutral kaon (down and strange quarks). Flies a few centimetres unseen, then decays to two pions: a 'V' in the tracker.",
    ),
    "K0L": ("K-long", "The long-lived neutral kaon: reaches the hadron calorimeter."),
    "p": ("proton", "Two up quarks and a down quark; stable. What the LHC collides."),
    "pbar": ("antiproton", "The proton's antiparticle."),
    "nbar": ("antineutron", "The neutron's antiparticle."),
    "D+": (
        "charged D meson",
        "A charm quark with a down antiquark. Lives 1.0 picoseconds (c tau 0.31 mm), then decays, mostly to kaons and pions.",
    ),
    "D-": ("charged D meson", "The D+'s antiparticle: a charm antiquark with a down quark."),
    "D0": (
        "neutral D meson",
        "A charm quark with an up antiquark. Lives 0.41 picoseconds (c tau 0.12 mm).",
    ),
    "D0bar": ("anti-D0 meson", "The D0's antiparticle: a charm antiquark with an up quark."),
    "Ds+": (
        "D_s meson",
        "A charm quark with a strange antiquark. Lives 0.50 picoseconds (c tau 0.15 mm).",
    ),
    "Ds-": ("D_s meson", "The Ds+'s antiparticle: a charm antiquark with a strange quark."),
    "B+": (
        "charged B meson",
        "A bottom antiquark with an up quark. Lives 1.6 picoseconds (c tau 0.49 mm): its decay point is set apart from the collision, which is how B mesons are found.",
    ),
    "B-": ("charged B meson", "The B+'s antiparticle: a bottom quark with an up antiquark."),
    "B0": (
        "neutral B meson",
        "A bottom antiquark with a down quark. Lives 1.5 picoseconds (c tau 0.45 mm). Oscillates into its antiparticle: 18.6% of B0 decay as a B0bar.",
    ),
    "B0bar": ("anti-B0 meson", "The B0's antiparticle: a bottom quark with a down antiquark."),
    "Bs": (
        "B_s meson",
        "A bottom antiquark with a strange quark. Lives 1.5 picoseconds (c tau 0.46 mm). Oscillates into its antiparticle about 4 times per lifetime: half of them decay as a Bsbar.",
    ),
    "Bsbar": ("anti-B_s meson", "The Bs's antiparticle: a bottom quark with a strange antiquark."),
    "D*+": (
        "charged D* meson",
        "The spin-1 partner of the D+ (charm quark, down antiquark). Decays at once,"
        " mostly to D0 pi+ with a slow pion.",
    ),
    "D*-": ("charged D* meson", "The D*+'s antiparticle. Decays at once, mostly to D0bar pi-."),
    "D*0": (
        "neutral D* meson",
        "The spin-1 partner of the D0. Decays at once to D0 pi0 or D0 gamma.",
    ),
    "D*0bar": ("anti-D*0 meson", "The D*0's antiparticle. Decays at once to D0bar pi0 or D0bar gamma."),
    "Ds*+": (
        "D_s* meson",
        "The spin-1 partner of the Ds+. Decays at once, mostly to Ds+ and a photon.",
    ),
    "Ds*-": ("D_s* meson", "The Ds*+'s antiparticle. Decays at once, mostly to Ds- and a photon."),
    "n": (
        "neutron",
        "Two down quarks and an up quark. Uncharged: seen only by its shower in the hadron calorimeter.",
    ),
    "Lambda": (
        "Lambda baryon",
        "A neutron-like particle with a strange quark (up, down, strange). Flies centimetres, then decays to a proton and a pion.",
    ),
    "J/psi": (
        "J/psi meson",
        "A charm quark bound to a charm antiquark. Its discovery in 1974 established the charm quark.",
    ),
    "Z": (
        "Z boson",
        "One of the carriers of the weak force, 97 times the proton's mass. Discovered at CERN in 1983. Decays at once.",
    ),
    "H": (
        "Higgs boson",
        "The particle of the Higgs field, which gives the W and Z bosons and the quarks and leptons their mass. Discovered at CERN in 2012.",
    ),
}
