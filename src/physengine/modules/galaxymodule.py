"""Galaxy collisions: self-gravitating N-body galaxies.

Skeleton: lifecycle, display settings and rendering are wired up. The
physics is yours to fill in at the places marked YOUR STEP:
    1. load():  initial conditions (positions, masses, component labels,
                then velocities)
    2. step():  forces (Barnes-Hut, softening) and integration

Model units: G = 1 with 1 kpc, 1e10 solar masses, 207.4 km/s, 4.715 Myr.
"""

import time
from functools import partial

import numpy as np
from imgui_bundle import imgui

from physengine.core.module import BaseModule
from physengine.core.scalebar import KPC_M, draw_scale_bar
from physengine.core.spec import Choice, SliderFloat
from physengine.modules import scenario_files
from physengine.modules.galacy_ics import (
    disc_velocities,
    ics,
    isotropic_velocities,
    midplane_rotation_curve,
    sample_disc,
)
from physengine.modules.galaxy_components import BULGE, DISC, HALO, GalaxyModel
from physengine.modules.galaxy_editor import (
    SetupEditor,
    clear_edits,
    has_edits,
    save_edits,
    scene_for,
    scene_from_dict,
)
from physengine.modules.galaxy_encounters import (
    SCENES,
    SINGLE,  # noqa: F401 (tests import it from here)
    place_pair,
)
from physengine.modules.galaxy_gpu import GpuGravity
from physengine.modules.galaxy_gravity import bh_acc_grouped
from physengine.modules.galaxy_orbit import GalaxyPair
from physengine.modules.galaxy_renderer import GalaxyRenderer
from physengine.modules.nbodystate import Integrator, NBodyState

DEFAULT_SCENE = "Prograde"

# Gravity and integration (Barnes-Hut + leapfrog; chosen 2026-11-20).
G = 1.0
# Tree opening angle. With the group walk, 0.7 gives the same force error as
# the per-particle walk at 0.5: ~0.3% median, ~1% 99th percentile.
THETA = 0.7
EPS = 0.1  # Plummer softening length [kpc]
TIME_UNIT_MYR = 4.715
SEED = 42  # fixed, so Restart rebuilds the same galaxy

# The galaxy models (components, masses, scale lengths, particle counts,
# Toomre Q) are in galaxy_components.GalaxyModel; the scenes in
# galaxy_encounters.SCENES.

# Massless tracer stars per galaxy, by component. They move in the gravity
# of the massive particles but exert none, so they only add detail to the
# picture; their forces use a looser tree (TRACER_THETA, ~1% error).
TRACERS = {DISC: 10000, BULGE: 2000}
TRACER_THETA = 1.0
# Choices for the number of tracer stars per galaxy (scene setting). They are
# split between disc and bulge in the proportions of TRACERS.
TRACER_LEVELS = ("0", "3000", "6000", "12000", "24000")
DEFAULT_TRACERS = "0"
# Opening angles for the GPU walk (one thread per particle, per-particle test).
# Chosen for speed (2026-11-20): at 1.0 the force error of massive particles
# is 1.3% median, 4-5% at the 99th percentile (at 0.6: 0.4% / 1.2%), and the
# walk costs about 9 ms instead of 14. Together with dt = 0.16 (see
# GalaxyModule.dt) a prograde merger over 905 Myr keeps the centre
# separations, tidal-tail fractions and pictures of a reference run
# (theta 0.3, dt 0.04) as well as 0.6 / 0.08 did; the energy error grows from
# 0.34% to 1.5%. Quadrupole moments were tried and gave at most ~20% at equal
# median accuracy with worse tails, so not used.
GPU_THETA = 1.0
GPU_TRACER_THETA = 1.0
# Rebuild the tree every this many steps (GpuGravity can refresh only the cell
# masses in between). Measured over 236 Myr of an isolated galaxy: reusing the
# tree for 4 steps of 0.04 gave an energy error of 7e-4 against 1.4e-4 for a
# fresh tree every step of 0.08, so a fresh tree with the longer step is both
# faster per simulated time and more accurate.
GPU_REBUILD_EVERY = 1


def _components(model):
    """(label, component) of a GalaxyModel, in the order particles are laid out."""
    return [(HALO, model.halo), (BULGE, model.bulge), (DISC, model.disc)]


def _sample_component(label, comp, n, rng):
    """Positions and masses of n particles of one component, centred on the origin.

    The particles share the component's mass (tracers get theirs zeroed later).
    """
    if label == DISC:
        p, m = sample_disc(n, comp.mass, comp.R_d, comp.z0, comp.R_max, rng)
    else:
        p, m = ics(n, comp.mass, comp.a, comp.r_max, rng)
    # Sampling noise puts the halo's centre of mass a kpc or two off the
    # origin; recentre each component so all share one centre.
    p -= p.mean(axis=0)
    return p, m


def tracer_counts(total):
    """Split `total` tracer stars over the components in the proportions of TRACERS."""
    default = sum(TRACERS.values())
    return {label: int(round(total * n / default)) for label, n in TRACERS.items()}


def make_galaxy(model: GalaxyModel, rng, n_tracers=sum(TRACERS.values())):
    """One galaxy of `model` in equilibrium, centred and at rest: (pos, vel, mass, kind).

    Massive particles first, then n_tracers tracers (mass 0).
    """
    comps = dict(_components(model))
    # (label, count, is_tracer)
    parts = [(label, comp.n, False) for label, comp in comps.items()]
    parts += [(label, n, True) for label, n in tracer_counts(n_tracers).items() if n > 0]
    pos, mass, kind, tracer = [], [], [], []
    for label, n, is_tracer in parts:
        p, m = _sample_component(label, comps[label], n, rng)
        pos.append(p)
        mass.append(np.zeros(n) if is_tracer else m)
        kind.append(np.full(n, label))
        tracer.append(np.full(n, is_tracer))
    pos, mass, kind, tracer = map(np.concatenate, (pos, mass, kind, tracer))
    vel = np.zeros_like(pos)
    heavy = ~tracer
    hp, hm = pos[heavy], mass[heavy]

    # Disc: rotation plus random motions (Toomre Q, sech^2 thickness). The
    # rotation curve is measured from the actual gravity in the disc plane,
    # with the simulation's own softening.
    d = model.disc
    curve = midplane_rotation_curve(hp, hm, d.R_max, EPS, G=G)
    sel = kind == DISC
    vel[sel] = disc_velocities(
        pos[sel], d.mass, d.R_d, d.z0, d.R_max, hp, hm, rng, Q_ref=d.Q, G=G, rotation_curve=curve
    )

    # Halo and bulge: isotropic random motions from the Jeans equation.
    for label, comp in ((HALO, model.halo), (BULGE, model.bulge)):
        sel = kind == label
        vel[sel] = isotropic_velocities(pos[sel], comp.a, comp.r_max, hp, hm, rng)

    # Random sampling leaves a small net momentum; remove it so the galaxy
    # does not drift. (Positions are already centred.)
    vel -= hm @ vel[heavy] / hm.sum()
    return pos, vel, mass, kind


class GalaxyModule(BaseModule):
    # Model time units (4.715 Myr) per leapfrog step. Chosen for speed
    # (2026-11-20, with GPU_THETA = 1.0, see there): 754 000 years per step.
    # Against a reference run (theta 0.3, dt 0.04) of the prograde merger
    # over 905 Myr: separations within 2 kpc, tail fractions within 0.01,
    # energy error 1.5%; the bulge core (radius of the inner 10%) ends at
    # 0.70 kpc against 0.61. Smaller steps (0.02-0.08) agree with each other
    # more closely: use 0.08 with theta 0.6 for accuracy over speed.
    dt = 0.16

    def __init__(self) -> None:
        # Which scene: one isolated galaxy, or a pair on a collision course.
        # Changing it rebuilds the module (Restart keeps it).
        self.scene_name = DEFAULT_SCENE
        # Tracer stars per galaxy: more makes a fuller picture, fewer runs faster.
        self.tracers = DEFAULT_TRACERS
        # Display settings.
        self.star_size = 2.0
        self.exposure = 0.05
        self.bloom = 0.0
        self.halo = "Hidden"
        self.specs = (
            Choice(
                attr_name="scene_name",
                label="Scene",
                default=DEFAULT_SCENE,
                options=tuple(SCENES),
                need_reset=True,
            ),
            Choice(
                attr_name="tracers",
                label="Tracer stars",
                default=DEFAULT_TRACERS,
                options=TRACER_LEVELS,
                need_reset=True,
            ),
            SliderFloat(attr_name="star_size", label="Star size", default=2.0, min=1.0, max=6.0),
            SliderFloat(attr_name="exposure", label="Exposure", default=0.05, min=0.05, max=8.0),
            SliderFloat(attr_name="bloom", label="Glow", default=0.0, min=0.0, max=3.0),
            Choice(
                attr_name="halo", label="Dark halo", default="Hidden", options=("Hidden", "Shown")
            ),
        )

        self.renderer = GalaxyRenderer()
        self.pending_view_distance = None  # set by load()
        self.camera = None  # set in render(), for the scale bar
        self.integrator = Integrator(
            partial(bh_acc_grouped, G=G, eps=EPS, theta=THETA, tracer_theta=TRACER_THETA)
        )
        self.time = 0.0  # simulated time, model units
        self.step_ms = 0.0  # wall-clock cost of the last step
        self.gpu = None  # GpuGravity once setup() has a GL context
        self.scene = None  # GalaxyScene, set by load()
        self.editor = SetupEditor(self)
        self.scenario_message = ""  # why a saved setup was not used, for the setup window

    def load(self):
        """Build the initial state.

        Sets self.scene (the GalaxyScene: the preset of self.scene_name, or
        its edited version from the Galaxy setup window), self.nb
        (NBodyState: pos, vel, mass), self.kind (component label per
        particle: HALO, BULGE or DISC from galaxy_components) and self.galaxy
        (which galaxy each particle belongs to, 0, 1, ...). Tracer stars are
        in the state too, with mass 0.
        """
        self.scene = scene_for(self.scene_name)
        n_tracers = int(self.tracers)
        first = make_galaxy(self.scene.galaxies[0], np.random.default_rng(SEED), n_tracers)
        if self.scene.encounter is None:
            pos, vel, mass, kind = first
            galaxy = np.zeros(len(mass), dtype=int)
        else:
            # The second galaxy is sampled independently, even with the same model.
            second = make_galaxy(self.scene.galaxies[1], np.random.default_rng(SEED + 1), n_tracers)
            pair = GalaxyPair(*self.scene.galaxies, G=G)
            pos, vel, mass, kind, galaxy = place_pair(first, second, self.scene.encounter, pair)

        self.nb = NBodyState(len(pos))
        self.nb.pos[:] = pos
        self.nb.vel[:] = vel
        self.nb.mass[:] = mass
        self.kind = kind
        self.galaxy = galaxy
        self.pending_view_distance = self._view_distance()

    def _view_distance(self):
        """Camera distance [kpc] when the scene starts: the whole setup in view."""
        if self.scene.encounter is None:
            return max(80.0, 4.0 * self.scene.galaxies[0].disc.R_max)
        return max(200.0, 2.0 * self.scene.encounter.start_distance)

    def setup(self, ctx):
        # Forces on the GPU when compute shaders are available; otherwise the
        # CPU tree from __init__ stays in use.
        try:
            self.gpu = GpuGravity(ctx, rebuild_every=GPU_REBUILD_EVERY)
            self.integrator = Integrator(
                partial(self.gpu.acc, G=G, eps=EPS, theta=GPU_THETA, tracer_theta=GPU_TRACER_THETA)
            )
        except Exception as error:  # e.g. no OpenGL 4.3 compute shaders
            print(f"Galaxy forces: GPU unavailable ({error}); using the CPU tree")
            self.gpu = None
        pending = scenario_files.take_request("galaxy")  # opened from the home screen
        if pending is not None:
            try:
                self.prepare_scenario(pending)
            except ValueError as error:
                self.scenario_message = f"Saved scenario not opened ({error}); showing the preset"
        self._load_or_fall_back()
        self.renderer.setup(ctx, self.kind, self.galaxy)

    def _load_or_fall_back(self):
        """load(); if an edited or saved setup cannot be built, the preset instead.

        Plumbing: a saved scenario can pass the checks of its form and still
        not build. That must leave a working module and a message, not an
        exception in the app's module switch.
        """
        if not has_edits(self.scene_name):
            self.load()
            return
        try:
            self.load()
        except Exception as error:  # noqa: BLE001 (any failure of an edited setup)
            clear_edits()
            self.scenario_message = f"This setup could not be built ({error}); showing the preset"
            self.load()

    def step(self, dt):
        start = time.perf_counter()
        self.integrator.step(self.nb, dt, 1)
        self.time += dt
        self.step_ms = 1000.0 * (time.perf_counter() - start)

    def render(self, ctx, camera):
        self.camera = camera
        if self.pending_view_distance is not None:
            camera.distance = self.pending_view_distance
            self.pending_view_distance = None

        self.renderer.render(
            ctx,
            camera,
            self.nb.pos,
            show_halo=self.halo == "Shown",
            point_size=self.star_size,
            exposure=self.exposure,
            bloom=self.bloom,
        )

    def diagnostics(self):
        heavy = self.nb.mass > 0.0
        counts = {
            name: int(np.sum(heavy & (self.kind == k)))
            for name, k in (("halo", HALO), ("bulge", BULGE), ("disc", DISC))
        }
        return (
            f" Particles: {int(heavy.sum())} (disc {counts['disc']}, bulge {counts['bulge']}, halo {counts['halo']})\n"
            f" Tracer stars: {int((~heavy).sum())}\n"
            f" Total mass: {self.nb.mass.sum():.4g} x 1e10 M_sun\n"
            f" Time: {self.time * TIME_UNIT_MYR:.1f} Myr\n"
            f" Step cost: {self.step_ms:.1f} ms ({'GPU' if self.gpu else 'CPU'} forces)"
            + self._separation_line()
        )

    def _separation_line(self):
        """Distance between the galaxy centres (median position of each bulge)."""
        if self.galaxy.max(initial=0) < 1:
            return ""
        heavy_bulge = (self.kind == BULGE) & (self.nb.mass > 0.0)
        c0 = np.median(self.nb.pos[heavy_bulge & (self.galaxy == 0)], axis=0)
        c1 = np.median(self.nb.pos[heavy_bulge & (self.galaxy == 1)], axis=0)
        return f"\n Separation: {np.linalg.norm(c1 - c0):.1f} kpc"

    def draw_ui(self):
        self.editor.update()
        if self.camera is not None:
            W, H = imgui.get_io().display_size
            draw_scale_bar(self.camera, W, H, KPC_M, "kpc")

    def prepare_scenario(self, data):
        """Take a saved setup: it becomes the edited version of its scene.

        Raises ValueError (from scene_from_dict) if `data` is not a galaxy
        setup; nothing is changed then.
        """
        name, scene = scene_from_dict(data)
        save_edits(name, scene)
        self.scene_name = name

    def load_scenario(self, data):
        self.prepare_scenario(data)
        self.reset_requested = True

    def discard_edits(self):
        clear_edits()

    def teardown(self):
        self.renderer.release()
        if self.gpu is not None:
            self.gpu.release()
