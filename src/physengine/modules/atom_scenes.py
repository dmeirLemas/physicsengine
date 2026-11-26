"""The scenes of the atom module: what each one simulates and what it gives the renderer.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, atom
module). Review the physics.

NumPy only: no GL, no ImGui. A scene owns the state of one demonstration,
advances it in step(), and describes what to draw:

    balls()    (rows, tags): rows (n, 8) = x, y, z, radius, r, g, b, glow.
               glow = 0: a solid shaded ball; glow > 0: a soft light of that
               brightness; glow < 0: a solid ball with a halo of brightness
               |glow|. tags (n,) index info_for(), or -1.
    cloud()    (n, 7) = x, y, z, r, g, b, size: static soft points (an
               electron cloud); re-read when cloud_version changes.
    streaks()  (n, 10) = start xyz, end xyz, r, g, b, width in pixels.
    labels()   [(xyz, text, rgb)]
    log        commentary, newest last.

Lengths: each scene has its own world unit (`unit`: 'a0' for electron
clouds, 'fm' for nuclei, others stated). Orbitals' z axis is drawn upward
(world y).

The physics is in atom_orbitals, atom_nucleus, atom_decay and
atom_reactions. What the scenes add is animation: where nucleons sit, how
fast things are drawn to move, how a nucleus stretches before it splits.
Those are illustrations and each scene's `disclaimer` says so.
"""

import colorsys
import time

import numpy as np

from physengine.modules import atom_decay, atom_nucleus, atom_orbitals, atom_reactions
from physengine.modules.atom_data import (
    BOHR_RADIUS_FM,
    C_LIGHT,
    NAMES,
    SYMBOLS,
    format_energy,
    format_time,
    label,
    parse_label,
)

# ------------------------------------------------------------------- settings
# What the user has chosen in each scene. Kept here (not in the scene
# objects) so that Restart, which rebuilds the module, returns to the same
# choices; Full reset puts the defaults back (reset_settings).
DEFAULTS = {
    "orbital": {"n": 3, "l": 2, "m": 0, "Z": 1, "real": True},
    "atom": {"Z": 18, "magnify": True},
    "nucleus": {"Z": 26, "A": 56},
    "decay": {"nuclide": "U-238", "continuous": True},
    "fission": {"system": "U-235", "continuous": True},
    "chainreaction": {"R": 2.4, "p": 0.6, "start": 30},
    "fusion": {"reaction": "D + T -> He-4 + n", "log_temperature": 8.2, "continuous": True},
}
SETTINGS = {key: dict(value) for key, value in DEFAULTS.items()}


def reset_settings():
    for key, value in DEFAULTS.items():
        SETTINGS[key] = dict(value)


# --------------------------------------------------------------------- colours
PROTON = (0.96, 0.30, 0.22)
NEUTRON = (0.42, 0.56, 0.86)
ELECTRON = (0.30, 0.95, 0.72)
POSITRON = (1.0, 0.45, 0.78)
NEUTRINO = (0.72, 0.72, 0.92)
PHOTON = (1.0, 0.88, 0.35)
POSITIVE = (1.0, 0.55, 0.18)  # psi > 0 in a real orbital
NEGATIVE = (0.22, 0.58, 1.0)  # psi < 0
SHELL_COLORS = (
    (1.0, 0.86, 0.40),  # n = 1 (K)
    (1.0, 0.50, 0.22),  # 2 (L)
    (0.95, 0.30, 0.52),  # 3 (M)
    (0.66, 0.42, 1.0),  # 4 (N)
    (0.30, 0.62, 1.0),  # 5 (O)
    (0.25, 0.90, 0.78),  # 6 (P)
    (0.55, 0.95, 0.35),  # 7 (Q)
)
KIND_COLOR = {
    "electron": ELECTRON, "positron": POSITRON, "neutrino": NEUTRINO, "photon": PHOTON,
    "neutron": NEUTRON, "proton": PROTON,
}  # fmt: skip

LOG_LINES = 40

ABOUT = {
    "proton": "Proton: charge +e, mass 938.272 MeV, two up quarks and one down quark.",
    "neutron": "Neutron: no charge, mass 939.565 MeV, one up quark and two down quarks."
    " Free, it decays in about 15 minutes (half-life 610 s); bound in a stable nucleus it"
    " does not.",
    "electron": "Electron: charge -e, mass 0.511 MeV.",
    "positron": "Positron: the electron's antiparticle, charge +e. It soon meets an electron"
    " and the pair turns into two 511 keV gamma rays.",
    "neutrino": "Neutrino: no charge, almost no mass, passes through matter practically"
    " unhindered. It carries off part of the energy of every beta decay.",
    "photon": "Photon: a quantum of light (here a gamma ray). No mass, no charge.",
    "alpha": "Alpha particle: a helium-4 nucleus, two protons and two neutrons, charge +2e.",
}


def phase_color(phase):
    """RGB for a wave-function phase [rad]: hue = phase / 2 pi (red at 0, cyan at pi)."""
    hue = (np.asarray(phase) / (2.0 * np.pi)) % 1.0
    return np.array([colorsys.hsv_to_rgb(h, 0.85, 1.0) for h in np.atleast_1d(hue)])


_PHASE_TABLE = phase_color(np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False))


def phase_colors(phase):
    index = (np.asarray(phase) / (2.0 * np.pi) * 256).astype(int) % 256
    return _PHASE_TABLE[index]


def to_world(xyz):
    """Physics axes (x, y, z) to world axes: z is drawn up (world y)."""
    xyz = np.asarray(xyz, dtype=float)
    return np.stack([xyz[..., 1], xyz[..., 2], xyz[..., 0]], axis=-1)


def speed_text(beta):
    if beta < 0.01:
        return f"{beta * C_LIGHT / 1000.0:.3g} km/s"
    return f"{beta * 100.0:.3g}% of the speed of light"


def beta_of(kinetic, mass):
    """v / c of a body with kinetic energy T and mass m."""
    if mass == 0.0:
        return 1.0
    gamma = 1.0 + kinetic / mass
    return float(np.sqrt(1.0 - 1.0 / gamma**2))


def _perpendicular(direction):
    other = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(direction, other)
    return side / np.linalg.norm(side)


def wave_streaks(head, direction, length, amplitude, rgb, width=2.0, waves=5.0, phase=0.0):
    """A wavy line (a photon) ending at `head`, as streak rows."""
    side = _perpendicular(direction)
    s = np.linspace(0.0, 1.0, 41)
    points = (
        head[None, :]
        - direction[None, :] * (length * s)[:, None]
        + side[None, :]
        * (amplitude * np.sin(2.0 * np.pi * waves * s + phase) * np.sin(np.pi * s))[:, None]
    )
    rows = np.zeros((len(s) - 1, 10))
    rows[:, 0:3], rows[:, 3:6] = points[:-1], points[1:]
    rows[:, 6:9] = np.asarray(rgb)[None, :] * (1.0 - 0.6 * s[:-1, None])
    rows[:, 9] = width
    return rows


def straight_streak(start, end, rgb, width=2.0):
    return np.array([[*start, *end, *rgb, width]])


EMPTY_BALLS = np.zeros((0, 8))
EMPTY_STREAKS = np.zeros((0, 10))


class Scene:
    """What every scene provides (see the module docstring)."""

    title = ""
    unit = "fm"
    unit_metres = 1e-15
    view = (20.0, 0.5, 0.25)  # camera distance, yaw, pitch
    disclaimer = ""
    cloud_size_limit = 48.0  # pixels: the largest a cloud point is drawn

    def __init__(self, table, rng):
        self.table = table
        self.rng = rng
        self.log = []
        self.cloud_version = 0
        self.view_version = 0
        self.infos = []

    def say(self, text):
        self.log.append(text)
        del self.log[:-LOG_LINES]

    def step(self, dt):
        return

    def work(self, budget=0.004):
        """Called every frame, paused or not: do a little of any bulk work
        asked for (events added to a histogram), for at most `budget` seconds."""
        return

    def balls(self):
        return EMPTY_BALLS, np.zeros(0, dtype=int)

    def cloud(self):
        return None

    def streaks(self):
        return EMPTY_STREAKS

    def labels(self):
        return []

    def info_for(self, tag):
        """(title, [lines]) about the thing with this tag, or None."""
        if 0 <= tag < len(self.infos):
            return self.infos[tag]
        return None


def nucleon_rows(points, protons, radius=atom_nucleus.NUCLEON_DRAW_RADIUS, scale=1.0):
    """Ball rows for nucleons at `points`: protons red, neutrons blue."""
    rows = np.zeros((len(points), 8))
    rows[:, 0:3] = points * scale
    rows[:, 3] = radius * scale
    rows[:, 4:7] = np.where(np.asarray(protons)[:, None], PROTON, NEUTRON)
    return rows


NUCLEON_TAGS = [
    ("Proton", [ABOUT["proton"]]),
    ("Neutron", [ABOUT["neutron"]]),
]


# ============================================================ electron clouds


class OrbitalScene(Scene):
    """One electron around a nucleus of charge Z: the orbital psi_nlm as a cloud."""

    title = "Hydrogen-like orbital"
    unit = "a0"
    unit_metres = 5.29177210903e-11
    disclaimer = (
        "Computed: the exact orbital of one electron (Schroedinger equation, point nucleus);"
        " each dot is a place where the electron could be found, drawn with probability"
        " |psi|^2 and coloured by the phase of psi. Level energies and line wavelengths from"
        " the Rydberg formula with the reduced mass; no fine structure. Illustration: the"
        " photon's picture, the instant jump."
    )
    POINTS = 90000

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.s = SETTINGS["orbital"]
        self.photon = None  # dict: direction, t, rgb, inward
        self.last_line = None
        self.rebuild()
        self.say(self.describe())

    # -- state

    @property
    def nuclear_mass(self):
        Z = self.s["Z"]
        return self.table.nuclear_mass(Z, self.table.default_isotope(Z))

    def energy(self, n=None):
        return atom_orbitals.energy(n or self.s["n"], self.s["Z"], self.nuclear_mass)

    def describe(self):
        s = self.s
        name = atom_orbitals.orbital_name(s["n"], s["l"], s["m"], s["real"])
        nodes = s["n"] - s["l"] - 1
        return (
            f"{self.species()}: orbital {name}, energy {self.energy():.4f} eV,"
            f" {nodes} radial node{'s' if nodes != 1 else ''},"
            f" {s['l']} angular; mean radius"
            f" {atom_orbitals.mean_radius(s['n'], s['l'], s['Z']):.3g} a0"
        )

    def species(self):
        Z = self.s["Z"]
        if Z == 1:
            return "Hydrogen"
        return f"{SYMBOLS[Z]}{Z - 1}+ ion (one electron)"

    def clamp(self):
        s = self.s
        s["n"] = int(np.clip(s["n"], 1, atom_orbitals.N_MAX))
        s["l"] = int(np.clip(s["l"], 0, s["n"] - 1))
        s["m"] = int(np.clip(s["m"], -s["l"], s["l"]))

    def rebuild(self):
        self.clamp()
        s = self.s
        xyz, phase = atom_orbitals.sample_orbital(
            s["n"], s["l"], s["m"], s["Z"], self.POINTS, s["real"], self.rng
        )
        self.extent = float(np.quantile(np.linalg.norm(xyz, axis=1), 0.97))
        data = np.zeros((len(xyz), 7))
        data[:, 0:3] = to_world(xyz)
        if s["real"]:
            data[:, 3:6] = np.where((phase > 1.0)[:, None], NEGATIVE, POSITIVE)
        else:
            data[:, 3:6] = phase_colors(phase)
        data[:, 6] = 0.016 * self.extent
        self._cloud = data
        self.cloud_version += 1
        self.view = (3.0 * self.extent, 0.6, 0.3)
        self.view_version += 1

    def set_orbital(self, n, l, m):  # noqa: E741
        self.s.update(n=n, l=l, m=m)
        self.rebuild()
        self.say(self.describe())

    # -- transitions

    def jump(self, n2, l2):
        """Go to level (n2, l2) by emitting or absorbing one photon, if the
        electric-dipole rules allow it. Returns whether it happened."""
        s = self.s
        n1, l1, m1 = s["n"], s["l"], s["m"]
        if (n2, l2) == (n1, l1):
            return False
        if n2 == n1:
            self.say(
                f"{n1}{atom_orbitals.L_LETTERS[l1]} and {n2}{atom_orbitals.L_LETTERS[l2]} have"
                " the same energy here (no fine structure): no photon."
            )
            return False
        if abs(l2 - l1) != 1:
            self.say(
                f"{n1}{atom_orbitals.L_LETTERS[l1]} -> {n2}{atom_orbitals.L_LETTERS[l2]} is"
                f" forbidden for electric-dipole light: l must change by exactly 1"
                f" (here by {l2 - l1:+d})."
            )
            return False
        m2 = int(np.clip(m1, -l2, l2))  # delta m = 0 where possible, else +-1
        upper, lower = max(n1, n2), min(n1, n2)
        photon, vacuum = atom_orbitals.transition(upper, lower, s["Z"], self.nuclear_mass)
        emitted = n2 < n1
        band = atom_orbitals.wavelength_band(vacuum)
        series = atom_orbitals.SERIES.get(lower)
        series_text = f", {series} series" if series else ""
        self.last_line = {
            "upper": upper, "lower": lower, "energy": photon, "vacuum": vacuum,
            "air": atom_orbitals.air_wavelength(vacuum) if vacuum > 200.0 else None,
            "band": band, "emitted": emitted,
        }  # fmt: skip
        verb = "emits" if emitted else "absorbs"
        self.say(
            f"{n1}{atom_orbitals.L_LETTERS[l1]} -> {n2}{atom_orbitals.L_LETTERS[l2]}: the atom"
            f" {verb} a photon of {photon:.4g} eV, wavelength {vacuum:.5g} nm"
            f" ({band}{series_text})."
        )
        self.photon = {
            "direction": to_world(atom_decay.isotropic(self.rng)),
            "t": 0.0,
            "rgb": atom_orbitals.wavelength_rgb(vacuum),
            "inward": not emitted,
        }
        s.update(n=n2, l=l2, m=m2)
        extent_before = self.extent
        self.rebuild()
        self.photon["extent"] = max(extent_before, self.extent)
        return True

    PHOTON_SECONDS = 2.2

    def step(self, dt):
        if self.photon is not None:
            self.photon["t"] += dt
            if self.photon["t"] > self.PHOTON_SECONDS:
                self.photon = None

    def _photon_head(self):
        p = self.photon
        f = p["t"] / self.PHOTON_SECONDS
        if p["inward"]:
            f = 1.0 - f
        return p["direction"] * (0.1 + 2.4 * f) * p["extent"]

    def balls(self):
        # The nucleus: far too small to see at this scale; a marker.
        rows = [[0.0, 0.0, 0.0, 0.012 * self.extent, 1.0, 0.95, 0.85, 1.6]]
        tags = [0]
        self.infos = [
            (
                "Nucleus",
                [
                    f"Charge +{self.s['Z']}e. At this scale it would be about"
                    f" {4.54e-5 * self.table.default_isotope(self.s['Z']) ** (1 / 3):.1g} a0"
                    " across: a hundred-thousandth of the cloud. The dot only marks the place."
                ],
            )
        ]
        if self.photon is not None:
            rows.append(
                [*self._photon_head(), 0.02 * self.photon["extent"], *self.photon["rgb"], 2.5]
            )
            tags.append(-1)
        return np.array(rows), np.array(tags)

    def streaks(self):
        if self.photon is None:
            return EMPTY_STREAKS
        p = self.photon
        direction = -p["direction"] if p["inward"] else p["direction"]
        return wave_streaks(
            self._photon_head(), direction, 0.7 * p["extent"], 0.05 * p["extent"], p["rgb"],
            width=2.5, phase=-14.0 * p["t"],
        )  # fmt: skip

    def labels(self):
        if self.photon is None or self.last_line is None:
            return []
        line = self.last_line
        return [(self._photon_head(), f"photon {line['vacuum']:.4g} nm", self.photon["rgb"])]

    def cloud(self):
        return self._cloud


class AtomScene(Scene):
    """A whole neutral atom: every occupied subshell as a cloud, the nucleus at the centre."""

    title = "Atom: electron shells and nucleus"
    unit = "a0"
    unit_metres = 5.29177210903e-11
    disclaimer = (
        "An illustration of shell structure, not a solution of the many-electron atom:"
        " ground-state configuration by the aufbau order (with the known exceptions), each"
        " subshell drawn as a hydrogen-like orbital with Slater's effective charge, averaged"
        " over directions. Sizes and energies are rough (tens of percent). The nucleus is"
        " drawn to scale unless 'magnified' is on; nucleon positions are schematic."
    )
    POINTS = 110000
    MAGNIFIED_FRACTION = 0.035  # magnified nucleus radius, as a fraction of the atom's

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.s = SETTINGS["atom"]
        self.rebuild()

    def rebuild(self):
        s = self.s
        s["Z"] = int(np.clip(s["Z"], 1, atom_orbitals.Z_MAX_CONFIGURATION))
        Z = s["Z"]
        self.A = self.table.default_isotope(Z)
        self.rows = atom_orbitals.subshells(Z)
        chunks = []
        for row in self.rows:
            count = max(int(self.POINTS * row["electrons"] / Z), 1500)
            xyz = atom_orbitals.sample_shell(row["n"], row["l"], row["z_eff"], count, self.rng)
            data = np.zeros((count, 7))
            data[:, 0:3] = xyz
            data[:, 3:6] = SHELL_COLORS[row["n"] - 1]
            # Brightness per electron equal: dimmer points where a subshell got extra ones.
            data[:, 3:6] *= (self.POINTS * row["electrons"] / Z) / count
            data[:, 6] = 0.03 * row["radius"]
            chunks.append(data)
        self._cloud = np.vstack(chunks)
        self.cloud_version += 1
        self.extent = max(row["radius"] for row in self.rows) * 1.9
        self.view = (2.4 * self.extent, 0.6, 0.3)
        self.view_version += 1
        self.points, self.protons = atom_nucleus.pack(Z, self.A)
        self.true_radius = float(atom_nucleus.radius(self.A)) / BOHR_RADIUS_FM  # a0
        self.log.clear()
        self.say(
            f"{NAMES[Z].capitalize()} ({label(Z, self.A)}): {Z} electrons,"
            f" {atom_orbitals.configuration_text(Z)}."
        )
        self.say(
            f"Nucleus radius {atom_nucleus.radius(self.A):.2f} fm = {self.true_radius:.2g} a0:"
            f" {self.extent / 1.9 / self.true_radius:,.0f} times smaller than the outer shell."
        )

    @property
    def magnification(self):
        if not self.s["magnify"]:
            return 1.0
        return self.MAGNIFIED_FRACTION * self.extent / self.true_radius

    def balls(self):
        scale = self.magnification / BOHR_RADIUS_FM
        rows = nucleon_rows(self.points, self.protons, scale=scale)
        self.infos = NUCLEON_TAGS
        return rows, np.where(self.protons, 0, 1)

    def labels(self):
        if not self.s["magnify"]:
            return [((0.0, 0.0, 0.0), "nucleus (true size)", (0.8, 0.8, 0.85))]
        text = f"nucleus, magnified x{self.magnification:,.0f} (display only)"
        return [((0.0, self.MAGNIFIED_FRACTION * self.extent, 0.0), text, (1.0, 0.75, 0.5))]

    def cloud(self):
        return self._cloud


# =================================================================== the nucleus


class NucleusScene(Scene):
    """A nucleus of Z protons and N neutrons, with its mass and binding energy."""

    title = "Nucleus and binding energy"
    unit = "fm"
    disclaimer = (
        "Computed or measured: radius R = 1.2 fm A^(1/3); masses and binding energies from"
        " the AME2020 table where it has them, else the semi-empirical formula; half-lives"
        " and decay modes from NUBASE2020. Illustration: nucleons as balls at fixed places"
        " (a nucleus is a quantum liquid, not a cluster of marbles)."
    )

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.s = SETTINGS["nucleus"]
        self.t = 0.0
        self.rebuild()

    def rebuild(self):
        s = self.s
        s["Z"] = int(np.clip(s["Z"], 1, len(SYMBOLS) - 1))
        s["A"] = int(np.clip(s["A"], s["Z"], 300))
        Z, A = s["Z"], s["A"]
        self.points, self.protons = atom_nucleus.pack(Z, A)
        self.wobble = self.rng.uniform(0.0, 2.0 * np.pi, (A, 3))
        self.R = float(atom_nucleus.radius(A))
        self.view = (max(3.4 * self.R, 9.0), 0.6, 0.25)
        self.view_version += 1
        self.log.clear()
        binding = self.table.binding(Z, A)
        nuclide = self.table.get(Z, A)
        life = ""
        if nuclide is not None:
            life = " Stable." if nuclide.stable else f" Half-life {format_time(nuclide.half_life)}."
        self.say(
            f"{label(Z, A)}: {Z} protons, {A - Z} neutrons, radius {self.R:.2f} fm."
            f" Binding energy {binding:.2f} MeV ({binding / A:.3f} per nucleon;"
            f" {self.table.source(Z, A)}).{life}"
        )

    def step(self, dt):
        self.t += dt

    def balls(self):
        jitter = 0.05 * np.sin(1.7 * self.t + self.wobble)
        self.infos = NUCLEON_TAGS
        return nucleon_rows(self.points + jitter, self.protons), np.where(self.protons, 0, 1)

    def labels(self):
        return [((0.0, self.R + 1.2, 0.0), label(self.s["Z"], self.s["A"]), (0.85, 0.9, 1.0))]


# ============================================================== flying particles

# How fast each kind is drawn to move [world units per second]. Display
# only: real speeds differ by factors of thousands (given in the text).
DRAW_SPEED = {
    "alpha": 7.0, "electron": 13.0, "positron": 13.0, "neutrino": 18.0, "photon": 18.0,
    "neutron": 9.0, "proton": 9.0, "nucleus": 0.45, "fragment": 3.6,
}  # fmt: skip


class Flight:
    """Something that leaves: a particle or a small nucleus, drawn from t0 on."""

    def __init__(self, kind, name, text, origin, direction, t0, speed=None, info=None,
                 cluster=None, size=0.45):  # fmt: skip
        self.kind, self.name, self.text = kind, name, text
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.t0 = t0
        self.speed = DRAW_SPEED[kind] if speed is None else speed
        self.info = info
        self.cluster = cluster  # (points, protons) for nuclei
        self.size = size
        self.phase = 0.0

    def position(self, t):
        return self.origin + self.direction * self.speed * max(t - self.t0, 0.0)

    def visible(self, t):
        return t >= self.t0

    def rows(self, t, tag):
        where = self.position(t)
        if self.cluster is not None:
            rows = nucleon_rows(self.cluster[0] + where, self.cluster[1])
            return rows, np.full(len(rows), tag)
        rgb = KIND_COLOR[self.kind]
        if self.kind in ("neutron", "proton"):
            row = [*where, atom_nucleus.NUCLEON_DRAW_RADIUS, *rgb, -0.8]
        elif self.kind == "neutrino":
            row = [*where, self.size * 0.7, *rgb, 0.7]
        else:
            row = [*where, self.size, *rgb, 2.2]
        return np.array([row]), np.array([tag])

    def streaks(self, t):
        where = self.position(t)
        travelled = np.linalg.norm(where - self.origin)
        if travelled <= 0.0:
            return EMPTY_STREAKS
        if self.kind == "photon":
            return wave_streaks(
                where, self.direction, min(travelled, 9.0), 0.55, PHOTON, 2.2, 6.0, -16.0 * t
            )
        if self.kind in ("electron", "positron", "neutrino", "neutron", "proton", "alpha"):
            rgb = KIND_COLOR.get(self.kind, (1.0, 0.8, 0.6))
            tail = where - self.direction * min(travelled, 5.0)
            width = 1.2 if self.kind == "neutrino" else 2.2
            dim = 0.35 if self.kind == "neutrino" else 0.7
            return straight_streak(tail, where, tuple(dim * c for c in rgb), width)
        return EMPTY_STREAKS


def _extreme(points, mask, direction, count):
    """Indices of the `count` points among `mask` furthest along `direction`."""
    candidates = np.flatnonzero(mask)
    order = np.argsort(points[candidates] @ direction)[::-1]
    return candidates[order[:count]]


class EventScene(Scene):
    """Shared by the scenes that play one event after another."""

    HOLD = 1.6  # seconds the finished event stays before the next one

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.t = 0.0
        self.duration = 6.0
        self.flights = []
        self.events = 0

    @property
    def continuous(self):
        return self.s.get("continuous", True)

    def step(self, dt):
        self.t += dt
        if self.t > self.duration + self.HOLD and self.continuous:
            self.new_event()

    def new_event(self):
        raise NotImplementedError

    def flight_draw(self, rows, tags, base_tag):
        for k, flight in enumerate(self.flights):
            if flight.visible(self.t):
                r, t = flight.rows(self.t, base_tag + k)
                rows.append(r)
                tags.append(t)

    def streaks(self):
        parts = [f.streaks(self.t) for f in self.flights if f.visible(self.t)]
        parts = [p for p in parts if len(p)]
        return np.vstack(parts) if parts else EMPTY_STREAKS

    def flight_labels(self):
        return [
            (
                f.position(self.t) + np.array([0.0, 1.3, 0.0]),
                f.text,
                KIND_COLOR.get(f.kind, (1.0, 0.9, 0.8)),
            )
            for f in self.flights
            if f.visible(self.t) and f.text
        ]


# ================================================================ a single decay

DECAY_PRESETS = (
    "U-238", "Ra-226", "Po-210", "Am-241", "C-14", "H-3", "n", "Co-60", "Cs-137", "Sr-90",
    "I-131", "K-40", "Na-22", "F-18", "Be-7", "Ba-137m", "Tc-99m", "Bi-212", "Cf-252",
)  # fmt: skip


class DecayScene(EventScene):
    """One unstable nucleus decays; then another of the same kind; and so on."""

    title = "Decay of one nucleus"
    unit = "fm"
    disclaimer = (
        "Computed: which decay happens (measured branching, NUBASE2020), the energy released"
        " (measured masses, AME2020) and how the products share it (energy and momentum"
        " conservation; beta energies from the allowed spectrum with a simple Coulomb factor)."
        " All decays go to the daughter's ground state: gamma rays from excited daughters are"
        " not shown. Illustration: the nucleons' positions, all speeds on screen, and the"
        " moment of decay (a real nucleus waits one half-life on average, not two seconds)."
    )
    T_DECAY = 1.8

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.s = SETTINGS["decay"]
        # branch mode -> energies [MeV] of the light particle, for the histogram
        self.energies = {}
        self.to_fill = 0
        self.set_nuclide(self.s["nuclide"])

    def set_nuclide(self, name):
        parsed = parse_label(name)
        if parsed is None or not self.table.exists(parsed[0], parsed[1]):
            self.say(f"'{name}' is not a nuclide.")
            return False
        self.s["nuclide"] = label(*parsed)
        self.key = parsed
        Z, A, isomer = parsed
        self.nuclide = self.table.get(Z, A, isomer)
        self.options = atom_decay.branches(self.table, Z, A, isomer)
        self.energies = {}
        self.events = 0
        self.to_fill = 0
        self.log.clear()
        self.R = float(atom_nucleus.radius(A))
        self.view = (max(4.2 * self.R, 16.0), 0.5, 0.22)
        self.view_version += 1
        if self.nuclide is None:
            self.say(f"{label(*parsed)} is not in the data: nothing is known here about its decay.")
        elif self.nuclide.stable:
            self.say(f"{label(*parsed)} is stable: it does not decay.")
        elif not self.options:
            self.say(
                f"{label(*parsed)} is unstable (half-life {format_time(self.nuclide.half_life)}),"
                " but its decay modes are not among those this module follows."
            )
        else:
            self.say(
                f"{label(*parsed)}: half-life {format_time(self.nuclide.half_life)}."
                " Each nucleus decays at a random moment; the scene shows one after another."
            )
        self.new_event()
        return True

    def fill(self, count):
        """Ask for `count` more decays in the energy histogram (made by work())."""
        if self.options:
            self.to_fill += count

    def work(self, budget=0.004):
        end = time.perf_counter() + budget
        while self.to_fill > 0 and time.perf_counter() < end:
            self.to_fill -= 1
            event = self._generate()
            if event is not None:
                self._record(event)

    def _generate(self):
        Z, A, isomer = self.key
        branch = self.options[
            self.rng.choice(len(self.options), p=[b.fraction for b in self.options])
        ]
        if branch.mode == "SF":
            return None
        return atom_decay.decay(self.table, Z, A, isomer, self.rng, branch)

    def _record(self, event):
        light = event.products[0]
        self.energies.setdefault(event.branch.mode, []).append(light.kinetic)
        self.events += 1

    def new_event(self):
        self.t = 0.0
        self.flights = []
        self.event = None
        self.fission = None
        Z, A, isomer = self.key
        self.points, self.protons = atom_nucleus.pack(Z, A)
        self.changed = None  # (nucleon index, becomes proton?) at T_DECAY
        self.removed = np.zeros(A, dtype=bool)  # nucleons that leave at T_DECAY
        self.recoil = np.zeros(3)
        self.daughter_text = ""
        self.duration = 6.5
        if not self.options:
            return
        branch = self.options[
            self.rng.choice(len(self.options), p=[b.fraction for b in self.options])
        ]
        self.branch = branch
        if branch.mode == "SF":
            self._spontaneous_fission()
            return
        self._set_view(max(4.2 * self.R, 16.0))
        event = atom_decay.decay(self.table, Z, A, isomer, self.rng, branch)
        self.event = event
        self._record(event)
        products = event.products
        daughter = products[-1]
        light = products[0]
        direction = light.p4[1:] / np.linalg.norm(light.p4[1:])
        recoil_p = np.linalg.norm(daughter.p4[1:])
        if recoil_p > 0.0:
            self.recoil = daughter.p4[1:] / recoil_p
        t0 = self.T_DECAY
        mode = branch.mode
        say = f"{label(Z, A, isomer)} -> {daughter.name}: {atom_decay.MODE_TEXT[mode]}"
        share = f" ({branch.fraction * 100:.3g}% of decays)" if branch.fraction < 0.9995 else ""
        self.say(f"{say}{share}, {format_energy(branch.q)} released.")

        def info(product, extra=""):
            beta = beta_of(product.kinetic, product.mass)
            lines = [
                ABOUT.get(product.kind, ""),
                f"Kinetic energy {format_energy(product.kinetic)}"
                + (f", speed {speed_text(beta)}." if product.mass > 0.0 else "."),
            ]
            if extra:
                lines.append(extra)
            return (product.name, [line for line in lines if line])

        if mode == "alpha":
            chosen = np.concatenate(
                [
                    _extreme(self.points, self.protons, direction, 2),
                    _extreme(self.points, ~self.protons, direction, 2),
                ]
            )
            self.removed[chosen] = True
            cluster = atom_nucleus.pack(2, 4)
            origin = self.points[chosen].mean(axis=0)
            self.flights.append(
                Flight("alpha", "alpha", f"alpha  {format_energy(light.kinetic)}", origin,
                       direction, t0, info=info(light), cluster=cluster)
            )  # fmt: skip
            self.say(
                f"The alpha particle takes {format_energy(light.kinetic)}"
                f" ({speed_text(beta_of(light.kinetic, light.mass))}); {daughter.name} recoils"
                f" with {format_energy(daughter.kinetic)}. Two bodies: the split is fixed."
            )
        elif mode in ("n", "p"):
            mask = self.protons if mode == "p" else ~self.protons
            chosen = _extreme(self.points, mask, direction, 1)
            self.removed[chosen] = True
            self.flights.append(
                Flight(light.kind, light.name, f"{light.name}  {format_energy(light.kinetic)}",
                       self.points[chosen[0]], direction, t0, info=info(light))
            )  # fmt: skip
        elif mode in ("beta-", "beta+"):
            neutrino = products[1]
            nu_direction = neutrino.p4[1:] / max(np.linalg.norm(neutrino.p4[1:]), 1e-12)
            mask = ~self.protons if mode == "beta-" else self.protons
            chosen = _extreme(self.points, mask, direction, 1)[0]
            self.changed = (chosen, mode == "beta-")
            origin = self.points[chosen]
            self.flights.append(
                Flight(light.kind, light.name, f"{light.name}  {format_energy(light.kinetic)}",
                       origin, direction, t0, info=info(light))
            )  # fmt: skip
            self.flights.append(
                Flight("neutrino", neutrino.name,
                       f"{neutrino.name}  {format_energy(neutrino.kinetic)}",
                       origin, nu_direction, t0, info=info(neutrino))
            )  # fmt: skip
            what = (
                "neutron turns into a proton" if mode == "beta-" else "proton turns into a neutron"
            )
            self.say(
                f"A {what}. The {light.name} gets {format_energy(light.kinetic)}, the"
                f" {neutrino.name} {format_energy(neutrino.kinetic)}: three bodies share the"
                " energy, differently every time."
            )
        elif mode == "EC":
            chosen = _extreme(self.points, self.protons, -direction, 1)[0]
            self.changed = (chosen, False)
            origin = self.points[chosen]
            start = origin - direction * 13.0 * t0
            self.flights.append(
                Flight("electron", "electron", "electron of the atom's inner shell", start,
                       direction, 0.0, speed=13.0,
                       info=("Captured electron", ["One of the atom's own innermost electrons."
                                                   " The nucleus absorbs it."]))
            )  # fmt: skip
            self.flights.append(
                Flight("neutrino", light.name, f"neutrino  {format_energy(light.kinetic)}", origin,
                       direction, t0, info=info(light))
            )  # fmt: skip
            self.say(
                "The nucleus captures one of the atom's inner electrons; a proton turns into a"
                f" neutron and a neutrino leaves with {format_energy(light.kinetic)} (always the"
                " same: two bodies)."
            )
        elif mode == "IT":
            self.flights.append(
                Flight("photon", "gamma", f"gamma  {format_energy(light.kinetic)}", np.zeros(3),
                       direction, t0, info=info(light))
            )  # fmt: skip
            self.say(
                f"The excited nucleus drops to its ground state and emits a gamma ray of"
                f" {format_energy(light.kinetic)}. Nothing else changes."
            )
        self.daughter_text = f"{daughter.name}  recoil {format_energy(daughter.kinetic)}"
        self.daughter_info = info(daughter, "Its speed on screen is exaggerated.")

    def _spontaneous_fission(self):
        Z, A, _ = self.key
        name = label(Z, A)
        system = None if name in atom_reactions.FISSILE else atom_reactions.generic_system(Z, A)
        self.fission = FissionAnimation(
            self.table,
            self.rng,
            name,
            system,
            self.points,
            self.protons,
            t_split=self.T_DECAY + 0.8,
        )
        self.duration = self.fission.duration
        self.flights = self.fission.flights
        for line in self.fission.lines:
            self.say(line)
        self._set_view(56.0)

    def _set_view(self, distance):
        """Fit the view to the event (a fission needs a much wider one)."""
        if self.view[0] != distance:
            self.view = (distance, self.view[1], self.view[2])
            self.view_version += 1

    def balls(self):
        self.infos = list(NUCLEON_TAGS)
        if self.fission is not None:
            return self.fission.balls(self.t, self.infos)
        t = self.t
        after = t >= self.T_DECAY and self.event is not None
        protons = self.protons.copy()
        keep = np.ones(len(protons), dtype=bool)
        shift = np.zeros(3)
        points = self.points
        if self.event is not None and self.branch.mode == "IT" and not after:
            # Excited: the nucleus is drawn pulsing.
            points = points * (1.0 + 0.04 * np.sin(14.0 * t))
        if after:
            keep = ~self.removed
            if self.changed is not None:
                protons[self.changed[0]] = self.changed[1]
            shift = self.recoil * DRAW_SPEED["nucleus"] * (t - self.T_DECAY)
        rows = nucleon_rows(points[keep] + shift, protons[keep])
        tags = np.where(protons[keep], 0, 1)
        if self.changed is not None and 0.0 <= t - self.T_DECAY < 0.9:
            index = int(np.sum(keep[: self.changed[0]]))
            rows[index, 7] = -2.5 * (1.0 - (t - self.T_DECAY) / 0.9)
        rows, tags = [rows], [tags]
        if after:
            self.infos.append(self.daughter_info)
        for flight in self.flights:
            if flight.kind == "electron" and self.branch.mode == "EC" and t >= self.T_DECAY:
                continue  # absorbed
            if flight.visible(t):
                self.infos.append(flight.info)
                r, tg = flight.rows(t, len(self.infos) - 1)
                rows.append(r)
                tags.append(tg)
        return np.vstack(rows), np.concatenate(tags)

    def streaks(self):
        if self.fission is not None:
            return self.fission.streaks(self.t)
        parts = []
        for f in self.flights:
            if f.kind == "electron" and self.event is not None and self.branch.mode == "EC":
                if self.t >= self.T_DECAY:
                    continue
            if f.visible(self.t):
                parts.append(f.streaks(self.t))
        parts = [p for p in parts if len(p)]
        return np.vstack(parts) if parts else EMPTY_STREAKS

    def labels(self):
        if self.fission is not None:
            return self.fission.labels(self.t)
        Z, A, isomer = self.key
        out = []
        if self.event is None or self.t < self.T_DECAY:
            out.append(((0.0, self.R + 1.4, 0.0), label(Z, A, isomer), (0.85, 0.9, 1.0)))
        else:
            shift = self.recoil * DRAW_SPEED["nucleus"] * (self.t - self.T_DECAY)
            out.append(
                (shift + np.array([0.0, self.R + 1.4, 0.0]), self.daughter_text, (0.85, 0.9, 1.0))
            )
            for f in self.flights:
                if f.kind == "electron" and self.branch.mode == "EC":
                    continue
                if f.visible(self.t):
                    out.append((f.position(self.t) + np.array([0.0, 1.3, 0.0]), f.text,
                                KIND_COLOR.get(f.kind, (1.0, 0.85, 0.7))))  # fmt: skip
        return out


# ======================================================================= fission


def _match_sorted(source, target, axis):
    """Pair each source point with a target point by their order along `axis`."""
    order_s = np.argsort(source @ axis)
    order_t = np.argsort(target @ axis)
    matched = np.zeros_like(source)
    matched[order_s] = target[order_t]
    return matched


class FissionAnimation:
    """One fission event, drawn: the nucleus stretches, necks and splits (an
    illustration), and the fragments, neutrons and gamma rays leave with the
    energies atom_reactions.fission computed."""

    def __init__(self, table, rng, system_name, system, points, protons, t_split=3.0):
        self.event = event = atom_reactions.fission(table, system_name, rng, system)
        self.t_split = t_split
        self.duration = t_split + 4.5
        self.points, self.protons = points, protons
        (zl, al, ml, pl), (zh, ah, mh, ph) = event.fragments
        axis = pl[1:] / np.linalg.norm(pl[1:])
        self.axis = axis
        along = points @ axis
        # Share the nucleons out by their place along the axis: the far end
        # to the light fragment, the middle neutrons become the free ones.
        owner = np.full(len(points), 1)  # 0 light, 1 heavy, 2 free neutron
        proton_order = np.flatnonzero(protons)[np.argsort(along[protons])[::-1]]
        owner[proton_order[:zl]] = 0
        neutron_order = np.flatnonzero(~protons)[np.argsort(along[~protons])[::-1]]
        nl = al - zl
        nu = len(event.neutrons)
        owner[neutron_order[:nl]] = 0
        owner[neutron_order[nl : nl + nu]] = 2
        self.owner = owner
        self.split_at = 0.5 * (along[owner == 0].min() + along[owner == 1].max())
        # Where each fragment's nucleons end up: a round packed nucleus.
        self.targets = np.zeros_like(points)
        self.centres = []
        for which, (z, a) in enumerate(((zl, al), (zh, ah))):
            mask = owner == which
            packed, packed_protons = atom_nucleus.pack(z, a)
            for flag in (True, False):
                sel = mask & (protons == flag)
                self.targets[sel] = _match_sorted(points[sel], packed[packed_protons == flag], axis)
            self.centres.append(points[mask].mean(axis=0))
        self.velocity = [axis * DRAW_SPEED["fragment"], -axis * DRAW_SPEED["fragment"] * al / ah]
        self.names = [label(zl, al), label(zh, ah)]
        kin = [float(atom_decay.kinetic(np.linalg.norm(p[1:]), m)) for p, m in ((pl, ml), (ph, mh))]
        self.fragment_text = [f"{n}  {e:.1f} MeV" for n, e in zip(self.names, kin, strict=True)]
        self.fragment_info = [
            (
                n,
                [
                    f"Fission fragment: {z} protons, {a - z} neutrons.",
                    f"Kinetic energy {e:.1f} MeV, speed {speed_text(beta_of(e, m))}.",
                    "Far too many neutrons to be stable: it beta-decays several times"
                    f" and ends as {label(*final)}.",
                ],
            )
            for n, z, a, e, m, final in zip(
                self.names, (zl, zh), (al, ah), kin, (ml, mh), event.final, strict=True
            )
        ]
        self.flights = []
        free = np.flatnonzero(owner == 2)
        m_n = event.neutron_mass
        for index, p4 in zip(free, event.neutrons, strict=True):
            T = float(p4[0] - m_n)
            self.flights.append(
                Flight("neutron", "neutron", f"n  {T:.2f} MeV", points[index] * 0.6,
                       p4[1:] / np.linalg.norm(p4[1:]), t_split + 0.15,
                       info=("Prompt neutron", [ABOUT["neutron"],
                             f"Kinetic energy {T:.2f} MeV, speed {speed_text(beta_of(T, m_n))}.",
                             "It can split another nucleus: the basis of a chain reaction."]))
            )  # fmt: skip
        for k, p4 in enumerate(event.gammas):
            self.flights.append(
                Flight("photon", "gamma", f"gamma {p4[0]:.1f} MeV" if k < 3 else "",
                       self.centres[k % 2], p4[1:] / np.linalg.norm(p4[1:]),
                       t_split + 0.5 + 0.12 * k,
                       info=("Prompt gamma ray", [ABOUT["photon"], f"Energy {p4[0]:.2f} MeV."]))
            )  # fmt: skip
        what = "absorbs a slow neutron and splits" if event.induced else "splits by itself"
        self.lines = [
            f"{system_name} {what}: {self.names[0]} + {self.names[1]} + {nu} neutron"
            f"{'s' if nu != 1 else ''}. {event.q:.1f} MeV released at once:"
            f" fragments {event.fragment_energy:.1f}, neutrons {event.neutron_energy:.1f},"
            f" gamma rays {event.gamma_energy:.1f} MeV.",
            f"Later the fragments' beta decays release another {event.later:.1f} MeV"
            f" (ending as {label(*event.final[0])} and {label(*event.final[1])}).",
        ]

    def _positions(self, t):
        """Nucleon positions at time t (illustration of the shape)."""
        points, axis = self.points, self.axis
        along = points @ axis
        across = points - along[:, None] * axis[None, :]
        before = self.t_split - t
        if before > 0.0:
            # Stretching, with a wobble, and a neck forming at the end.
            progress = np.clip(1.0 - before / 1.6, 0.0, 1.0)
            stretch = 0.55 * progress**2 + 0.05 * np.sin(11.0 * t) * (0.3 + progress)
            neck = 0.75 * progress**3
        else:
            stretch, neck = 0.55, 0.75
        s = along * (1.0 + stretch)
        pinch = 1.0 - neck * np.exp(-(((along - self.split_at) / 2.2) ** 2))
        shaped = s[:, None] * axis[None, :] + across * (pinch / np.sqrt(1.0 + stretch))[:, None]
        if before > 0.0:
            return shaped
        # After the split each fragment rounds off and flies away.
        elapsed = -before
        settle = 1.0 - np.exp(-elapsed / 0.35)
        out = shaped.copy()
        for which in (0, 1):
            mask = self.owner == which
            centre = shaped[mask].mean(axis=0)
            rounded = centre + self.targets[mask]
            out[mask] = (
                shaped[mask] * (1.0 - settle) + rounded * settle + self.velocity[which] * elapsed
            )
        return out

    def fragment_centre(self, which, t):
        return self._positions(t)[self.owner == which].mean(axis=0)

    def balls(self, t, infos):
        positions = self._positions(t)
        split = t >= self.t_split
        mask = self.owner != 2 if split else np.ones(len(positions), dtype=bool)
        rows = nucleon_rows(positions[mask], self.protons[mask])
        tags = np.where(self.protons[mask], 0, 1)
        if split:
            infos.extend(self.fragment_info)
        rows, tags = [rows], [tags]
        for flight in self.flights:
            if flight.visible(t):
                infos.append(flight.info)
                r, tg = flight.rows(t, len(infos) - 1)
                rows.append(r)
                tags.append(tg)
        return np.vstack(rows), np.concatenate(tags)

    def streaks(self, t):
        parts = [f.streaks(t) for f in self.flights if f.visible(t)]
        parts = [p for p in parts if len(p)]
        return np.vstack(parts) if parts else EMPTY_STREAKS

    def labels(self, t):
        out = []
        if t >= self.t_split:
            for which in (0, 1):
                where = self.fragment_centre(which, t) + np.array([0.0, 6.5, 0.0])
                out.append((where, self.fragment_text[which], (0.85, 0.9, 1.0)))
            for f in self.flights:
                if f.visible(t) and f.text:
                    out.append(
                        (f.position(t) + np.array([0.0, 1.3, 0.0]), f.text, KIND_COLOR[f.kind])
                    )
        return out


class FissionScene(EventScene):
    """A slow neutron splits a uranium or plutonium nucleus."""

    title = "Fission of one nucleus"
    unit = "fm"
    view = (56.0, 0.35, 0.2)
    disclaimer = (
        "Computed: the energy released by each split (measured masses), exact energy and"
        " momentum conservation, the later beta-decay energy. Parametrised from measurements"
        " (approximate): which fragments form, how many neutrons, their energies, the"
        " fragments' kinetic energy. Illustration: the stretching and splitting shape, where"
        " nucleons sit, all speeds on screen."
    )
    T_ARRIVE = 1.3

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.s = SETTINGS["fission"]
        self.light_masses = []  # fragment mass numbers before neutron emission, for the yield plot
        self.heavy_masses = []
        self.totals = []  # (q, fragments, neutrons, gammas, later, number of neutrons)
        self.to_fill = 0
        self.new_event()

    def set_system(self, name):
        self.s["system"] = name
        self.light_masses, self.heavy_masses, self.totals = [], [], []
        self.events = 0
        self.to_fill = 0
        self.log.clear()
        self.new_event()

    def _record(self, event):
        self.light_masses.append(event.primary_masses[0])
        self.heavy_masses.append(event.primary_masses[1])
        self.totals.append(
            (event.q, event.fragment_energy, event.neutron_energy, event.gamma_energy,
             event.later, len(event.neutrons))
        )  # fmt: skip
        self.events += 1

    def fill(self, count):
        """Ask for `count` more fissions in the statistics (made by work())."""
        self.to_fill += count

    def work(self, budget=0.004):
        end = time.perf_counter() + budget
        while self.to_fill > 0 and time.perf_counter() < end:
            self.to_fill -= 1
            self._record(atom_reactions.fission(self.table, self.s["system"], self.rng))

    def new_event(self):
        self.t = 0.0
        system = atom_reactions.FISSILE[self.s["system"]]
        Z, A = system["target"]
        self.induced = system["induced"]
        compound_a = A + (1 if self.induced else 0)
        points, protons = atom_nucleus.pack(Z, compound_a)
        self.incoming = None
        if self.induced:
            # The absorbed neutron: the nucleon furthest to the -x side flies in.
            self.incoming = int(np.flatnonzero(~protons)[np.argmin(points[~protons, 0])])
        self.animation = FissionAnimation(
            self.table, self.rng, self.s["system"], None, points, protons,
            t_split=self.T_ARRIVE + 2.0,
        )  # fmt: skip
        self.duration = self.animation.duration
        self.flights = self.animation.flights
        self._record(self.animation.event)
        for line in self.animation.lines:
            self.say(line)
        self.target_name = label(Z, A)
        self.compound_name = label(Z, compound_a)
        self.R = float(atom_nucleus.radius(compound_a))

    def balls(self):
        self.infos = list(NUCLEON_TAGS)
        t = self.t
        if self.induced and t < self.T_ARRIVE:
            anim = self.animation
            rows = nucleon_rows(anim.points, anim.protons)
            tags = np.where(anim.protons, 0, 1)
            # The neutron on its way in (slowly: a thermal neutron).
            start = anim.points[self.incoming] + np.array([-34.0, 0.0, 0.0])
            f = t / self.T_ARRIVE
            rows[self.incoming, 0:3] = start + (anim.points[self.incoming] - start) * f
            rows[self.incoming, 7] = -1.2
            return rows, tags
        return self.animation.balls(max(t, self.T_ARRIVE), self.infos)

    def streaks(self):
        return self.animation.streaks(self.t)

    def labels(self):
        t = self.t
        if self.induced and t < self.T_ARRIVE:
            anim = self.animation
            start = anim.points[self.incoming] + np.array([-34.0, 0.0, 0.0])
            where = start + (anim.points[self.incoming] - start) * (t / self.T_ARRIVE)
            return [
                (where + np.array([0.0, 1.5, 0.0]), "slow neutron", NEUTRON),
                ((0.0, self.R + 1.5, 0.0), self.target_name, (0.85, 0.9, 1.0)),
            ]
        if t < self.animation.t_split:
            text = f"{self.compound_name}, excited" if self.induced else self.compound_name
            return [((0.0, self.R + 3.0, 0.0), text, (1.0, 0.85, 0.6))]
        return self.animation.labels(t)


# ================================================================ chain reaction


class ChainReactionScene(Scene):
    """Neutrons multiplying (or dying out) in a sphere of fissile material."""

    title = "Chain reaction"
    unit = "mean free paths"
    unit_metres = None
    disclaimer = (
        "A schematic model, not a reactor or weapon calculation: a bare sphere, neutrons of"
        " one speed flying straight, each collision a fission (probability p) or a capture."
        " No scattering, no energies, no delayed neutrons, no real material. Lengths in mean"
        " free paths. Only the neutron numbers per fission are U-235's. The grey dots just"
        " mark the material."
    )
    TIME_RATE = 0.9  # mean free flight times per second of animation

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.s = SETTINGS["chainreaction"]
        self.multiplicity = atom_reactions.multiplicity_distribution(
            atom_reactions.FISSILE["U-235"]["nu_bar"]
        )
        self.restart()

    def restart(self):
        s = self.s
        self.sim = atom_reactions.ChainReaction(
            s["R"], s["p"], self.multiplicity, self.rng, max_neutrons=12000
        )
        self.sim.inject(int(s["start"]))
        self.flashes = []  # [position, age]
        R = s["R"]
        n = 2600
        direction = self.rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        dots = np.zeros((n, 7))
        dots[:, 0:3] = direction * (R * self.rng.random(n) ** (1.0 / 3.0))[:, None]
        dots[:, 3:6] = (0.30, 0.36, 0.5)
        dots[:, 6] = 0.035 * R
        self._cloud = dots
        self.cloud_version += 1
        # The sphere's outline: three great circles.
        angle = np.linspace(0.0, 2.0 * np.pi, 73)
        circle = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], 1) * R
        rings = []
        for axes in ((0, 1, 2), (0, 2, 1), (2, 0, 1)):
            ring = circle[:, axes]
            rows = np.zeros((len(ring) - 1, 10))
            rows[:, 0:3], rows[:, 3:6] = ring[:-1], ring[1:]
            rows[:, 6:9] = (0.10, 0.14, 0.24)
            rows[:, 9] = 1.2
            rings.append(rows)
        self.outline = np.vstack(rings)
        self.view = (3.3 * R, 0.5, 0.25)
        self.view_version += 1
        self.log.clear()
        k = self.sim.k_infinity
        self.say(
            f"Sphere of radius {R:.2f} mean free paths; a collision causes fission with"
            f" probability {s['p']:.2f}. In an endless block each neutron would be followed by"
            f" k = {self.sim.nu_bar:.2f} x {s['p']:.2f} = {k:.2f} others; here some escape."
        )
        self.told = False

    def inject(self, n):
        self.sim.inject(n)

    def step(self, dt):
        self.sim.step(dt * self.TIME_RATE)
        self.flashes = [[p, age + dt] for p, age in self.flashes if age + dt < 0.5]
        self.flashes.extend([p, 0.0] for p in self.sim.recent_fissions[:150])
        if not self.told:
            if self.sim.saturated:
                self.say(
                    f"{self.sim.count} neutrons in flight and still growing: supercritical."
                    " The run stops here."
                )
                self.told = True
            elif self.sim.count == 0 and self.sim.time > 0.0:
                k = self.sim.measured_k()
                tail = f" (measured k = {k:.2f})" if k else ""
                self.say(f"The last neutron is gone: the chain died out{tail}. Subcritical.")
                self.told = True

    def balls(self):
        sim = self.sim
        R = self.s["R"]
        n = min(sim.count, 6000)
        rows = np.zeros((n + len(self.flashes), 8))
        rows[:n, 0:3] = sim.pos[:n]
        rows[:n, 3] = 0.022 * R + 0.012
        rows[:n, 4:7] = (0.55, 0.85, 1.0)
        rows[:n, 7] = 1.3
        for k, (where, age) in enumerate(self.flashes):
            rows[n + k] = [*where, (0.05 + 0.35 * age) * (0.5 + 0.2 * R), 1.0, 0.75, 0.3,
                           3.0 * (1.0 - age / 0.5)]  # fmt: skip
        self.infos = [
            ("Neutron", [ABOUT["neutron"], "In this model: one speed, straight flight."]),
            ("Fission", ["A neutron has just split a nucleus here; new neutrons leave."]),
        ]
        tags = np.concatenate([np.zeros(n, dtype=int), np.ones(len(self.flashes), dtype=int)])
        return rows, tags

    def streaks(self):
        sim = self.sim
        n = min(sim.count, 2500)
        rows = np.zeros((n, 10))
        rows[:, 3:6] = sim.pos[:n]
        rows[:, 0:3] = sim.pos[:n] - 0.16 * sim.dir[:n]
        rows[:, 6:9] = (0.2, 0.4, 0.6)
        rows[:, 9] = 1.4
        return np.vstack([self.outline, rows])

    def cloud(self):
        return self._cloud


# ======================================================================== fusion

FUSION_CHOICES = (*atom_reactions.REACTIONS, "Proton-proton chain (all three steps)")


class FusionScene(EventScene):
    """Two light nuclei meet and fuse; the products carry off the energy."""

    title = "Fusion"
    unit = "fm"
    view = (30.0, 0.0, 0.18)
    disclaimer = (
        "Computed: the energy released (measured masses), its split between the products"
        " (energy and momentum conservation, reactants at rest), the Coulomb barrier and the"
        " tunnelling factor. Not computed: how often the reaction happens (no cross-sections)."
        " Shown is a pair that does fuse; at any real temperature almost all approaches end"
        " with the nuclei bouncing apart. Illustration: nucleon positions, all speeds on screen."
    )
    T_MEET = 1.7
    T_OUT = 2.2

    def __init__(self, table, rng):
        super().__init__(table, rng)
        self.s = SETTINGS["fusion"]
        self.chain_step = 0
        self.set_choice(self.s["reaction"])

    def set_choice(self, name):
        self.s["reaction"] = name
        self.chain_step = 0
        self.log.clear()
        self.new_event()

    @property
    def is_chain(self):
        return self.s["reaction"] not in atom_reactions.REACTIONS

    @property
    def reaction(self):
        if self.is_chain:
            return atom_reactions.REACTIONS[atom_reactions.PP_CHAIN[self.chain_step % 3]]
        return atom_reactions.REACTIONS[self.s["reaction"]]

    @property
    def temperature(self):
        return 10.0 ** self.s["log_temperature"]

    def new_event(self):
        self.t = 0.0
        if self.is_chain and self.events:
            self.chain_step += 1
        reaction = self.reaction
        event = atom_reactions.react(self.table, reaction, self.rng)
        self.event = event
        self.events += 1
        self.clusters = [atom_nucleus.pack(z, a) for z, a in reaction.reactants]
        self.reactant_names = [atom_reactions.particle_name(r) for r in reaction.reactants]
        total_a = sum(a for _, a in reaction.reactants)
        total_z = sum(z for z, _ in reaction.reactants)
        self.compound = atom_nucleus.pack(total_z, total_a)
        self.flights = []
        energies = event.kinetic_energies()
        for item, (name, mass, p4), (_, T) in zip(
            reaction.products, event.products, energies, strict=True
        ):
            direction = p4[1:] / max(np.linalg.norm(p4[1:]), 1e-12)
            text = f"{name}  {format_energy(T)}"
            beta = beta_of(T, mass)
            lines = [f"Kinetic energy {format_energy(T)}"
                     + (f", speed {speed_text(beta)}." if mass > 0 else ".")]  # fmt: skip
            if isinstance(item, str):
                kind = item
                lines.insert(0, ABOUT[kind])
                self.flights.append(
                    Flight(kind, name, text, np.zeros(3), direction, self.T_OUT, info=(name, lines))
                )
            else:
                z, a = item
                kind = "neutron" if (z, a) == (0, 1) else "proton" if (z, a) == (1, 1) else "alpha"
                cluster = None if a == 1 else atom_nucleus.pack(z, a)
                if a == 1:
                    lines.insert(0, ABOUT[kind])
                self.flights.append(
                    Flight(kind, name, text, np.zeros(3), direction, self.T_OUT,
                           speed=3.0 + 9.0 * np.sqrt(beta / 0.2), info=(name, lines),
                           cluster=cluster)
                )  # fmt: skip
        parts = ", ".join(f"{n} {format_energy(T)}" for n, T in energies)
        prefix = f"Step {self.chain_step % 3 + 1} of 3. " if self.is_chain else ""
        self.say(f"{prefix}{reaction.name}: {format_energy(event.q)} released. {parts}.")
        extra = atom_reactions.annihilation_energy(reaction)
        if extra:
            self.say(f"The positron then meets an electron: two gamma rays, {extra:.3f} MeV more.")

    def _approach(self, t):
        """Distance of each reactant from the centre while they come together (display)."""
        f = np.clip(t / self.T_MEET, 0.0, 1.0)
        return 13.0 * (1.0 - f) ** 0.6

    def balls(self):
        self.infos = list(NUCLEON_TAGS)
        t = self.t
        rows, tags = [], []
        if t < self.T_MEET:
            d = self._approach(t)
            for sign, (points, protons) in zip((-1.0, 1.0), self.clusters, strict=True):
                offset = np.array([sign * (d + 0.9), 0.0, 0.0])
                rows.append(nucleon_rows(points + offset, protons))
                tags.append(np.where(protons, 0, 1))
        elif t < self.T_OUT:
            points, protons = self.compound
            r = nucleon_rows(points * (1.0 + 0.08 * np.sin(40.0 * t)), protons)
            r[:, 7] = -1.5
            rows.append(r)
            tags.append(np.where(protons, 0, 1))
        for flight in self.flights:
            if flight.visible(t):
                self.infos.append(flight.info)
                r, tg = flight.rows(t, len(self.infos) - 1)
                rows.append(r)
                tags.append(tg)
        if not rows:
            return EMPTY_BALLS, np.zeros(0, dtype=int)
        return np.vstack(rows), np.concatenate(tags)

    def labels(self):
        t = self.t
        if t < self.T_MEET:
            d = self._approach(t)
            return [
                ((sign * (d + 0.9), 2.6, 0.0), name, (0.85, 0.9, 1.0))
                for sign, name in zip((-1.0, 1.0), self.reactant_names, strict=True)
            ]
        if t < self.T_OUT:
            return [((0.0, 3.0, 0.0), "fused", (1.0, 0.85, 0.6))]
        return self.flight_labels()


SCENES = {
    "Hydrogen-like orbital": lambda table, rng: OrbitalScene(table, rng),
    "Atom: shells and nucleus": lambda table, rng: AtomScene(table, rng),
    "Nucleus and binding energy": lambda table, rng: NucleusScene(table, rng),
    "Decay of one nucleus": lambda table, rng: DecayScene(table, rng),
    "Fission of one nucleus": lambda table, rng: FissionScene(table, rng),
    "Chain reaction": lambda table, rng: ChainReactionScene(table, rng),
    "Fusion": lambda table, rng: FusionScene(table, rng),
}
DEFAULT_SCENE = "Hydrogen-like orbital"
