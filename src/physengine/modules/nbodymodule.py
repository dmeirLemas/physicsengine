import time
from copy import deepcopy
from functools import partial

import numpy as np

from physengine.core.module import BaseModule
from physengine.core.spec import Choice, SliderFloat
from physengine.modules.nbody_collisions import (
    MERGE_FRACTION,
    N_FRAGMENTS,
    N_TIDAL,
    fragments,
    largest_remnant_fraction,
    tidal_pieces,
)
from physengine.modules.nbody_debris import split_acc
from physengine.modules.nbody_editor import (
    FOLLOW_COM,
    FOLLOW_NONE,
    SceneEditor,
    clear_restart_point,
    load_restart_point,
    save_restart_point,
)
from physengine.modules.nbody_editor import reset_settings as reset_editor_settings
from physengine.modules.nbody_editor import settings as editor_settings
from physengine.modules.nbody_plunge import CAPACITY as PLUNGE_CAPACITY
from physengine.modules.nbody_plunge import ClosePasses
from physengine.modules.nbody_renderer import NBodyRenderer
from physengine.modules.nbody_scenes import SCENES
from physengine.modules.nbody_sheet import SpacetimeSheet
from physengine.modules.nbody_trails import TrailRenderer
from physengine.modules.nbodystate import (
    CAPTURE_RADIUS,
    HANDOVER_RADIUS,
    LIGHT_BODY,
    Integrator,
    NBodyState,
    acc,
    energy,
    hole_timescale,
    merge_pair,
    min_timescale,
    pair_timescales,
    total_angular_momentum,
    total_momentum,
)
from physengine.modules.postnewtonian import (
    C_AU_PER_YEAR,
    eih_acc,
    eih_energy,
    quadrupole_ddot,
    rr_acc,
)
from physengine.modules.starfield import Starfield

G = 4.0 * np.pi**2

force = partial(acc, G=G, eps=0.0)
e = partial(energy, G=G, eps=0.0)
timescale = partial(min_timescale, G=G)

NEWTON = "Newton"
RELATIVITY = "Newton + 1PN (relativity)"
DEFAULT_GRAVITY = RELATIVITY


def _whole_1pn(pos, vel, mass):
    return eih_acc(pos, vel, mass, G, C_AU_PER_YEAR) + rr_acc(pos, vel, mass, G, C_AU_PER_YEAR)


def force_1pn(pos, mass, vel, debris=None):
    """1PN gravity plus the gravitational-wave reaction (2.5PN), see postnewtonian.py.
    With debris, split as in nbody_debris.py."""
    if debris is None:
        return _whole_1pn(pos, vel, mass)
    return split_acc(pos, vel, mass, debris, _whole_1pn, G, C_AU_PER_YEAR, True)


def force_newton(pos, mass, debris=None):
    if debris is None:
        return force(pos, mass)
    return split_acc(
        pos, np.zeros_like(pos), mass, debris, lambda p, v, m: force(p, m), G, C_AU_PER_YEAR, False
    )


def e_1pn(pos, vel, mass):
    return eih_energy(pos, vel, mass, G, C_AU_PER_YEAR)


# Substeps with 1PN forces. The leapfrog's own pericentre drift scales as
# ETA^2; at the ordinary ETA = 0.02 it is a quarter of the 1PN advance on the
# "Star near a black hole" orbit, at a quarter of that (0.005) it is 1/60.
# The finer substeps are only needed where 1PN is strong, so each pair's
# timescale is shortened by a factor that goes from 1 where its
# G (m_i + m_j) / (c^2 r_ij) is below STRONG_FIELD (solar system: ~1e-8, 1PN
# effects far below what the drawing shows) down to FINE_1PN at 16 times
# that. The factor is part of the timescale, a function of the state, so the
# integrator's substep selection stays time symmetric. (Until 2026-11-22 ETA
# itself was switched once per step, which broke that symmetry during deep
# encounters and cost energy errors of order one.)
STRONG_FIELD = 1e-5
FINE_1PN = 0.25


def compactness(nb):
    """Largest G (m_i + m_j) / (c^2 r_ij) over pairs of bodies."""
    if nb.N < 2:
        return 0.0
    _, r, m = pair_timescales(nb.pos, nb.vel, nb.mass, G, nb.debris)
    return float(np.max(G * m / (C_AU_PER_YEAR**2 * np.maximum(r, 1e-300))))


def timescale_1pn(pos, vel, mass, debris=None):
    """Shortest pairwise timescale, shortened where 1PN is strong (see STRONG_FIELD)."""
    if len(mass) < 2:
        return np.inf
    if debris is None:
        debris = np.zeros(len(mass), dtype=np.bool_)
    tau, r, m = pair_timescales(pos, vel, mass, G, debris)
    strength = G * m / (C_AU_PER_YEAR**2 * np.maximum(r, 1e-300))
    factor = np.clip(np.sqrt(STRONG_FIELD / np.maximum(strength, 1e-300)), FINE_1PN, 1.0)
    return float(np.min(tau * factor))


def make_integrator(gravity=NEWTON):
    """Leapfrog with adaptive substeps for close encounters; 1PN forces if asked."""
    if gravity == RELATIVITY:
        return Integrator(force_1pn, timescale_1pn, velocity_dependent=True)
    return Integrator(force_newton, timescale)


DEFAULT_SCENE = "Sun-Earth"
# When the camera starts following a body it zooms in to this many drawn
# radii away, if it is further out than that.
FOLLOW_DISTANCE_RADII = 25.0
MAX_TRAIL_YEARS = 5.0
# Predicted paths of the selected or placed body get one point per this many
# physics steps.
PREVIEW_STRIDE = 2
# Wall-clock seconds per frame spent extending a predicted path. The path is
# computed once and extended as the simulation moves on (see PathPrediction);
# recomputing all of it every frame cost 30 to 400 ms per frame.
PREDICT_BUDGET = 0.004
# ...and the larger allowance for just keeping the path's lead over the
# running simulation. If even that is not enough the path gets shorter.
PREDICT_KEEP_UP = 0.012
# Wall-clock seconds one physics step may take. An encounter that needs more
# substeps than fit (a tight black-hole pair, debris whipping round a hole)
# then runs in slow motion instead of dragging the frame rate down.
STEP_BUDGET = 0.012
# No more debris is made beyond this many bodies (bodies then simply merge).
MAX_BODIES = 700
MAX_TRAIL_STEPS = 1200  # trail samples kept per body
MAX_LOOKAHEAD_STEPS = 2400
MOLTEN = (1.0, 0.5, 0.15)  # display colour of fresh collision fragments
# A distant observer never sees a body cross a horizon: its image freezes
# there, reddens and dims, e-folding every ~4 G M / c^3 (80 s for Sgr A*,
# 20 microseconds for one solar mass). That is far too fast to see, and for
# most holes here the horizon is far below a pixel, so it is shown stylised:
# the swallowed body stays at the edge of the hole as drawn, reddening and
# fading over this many seconds of wall-clock time. Display only.
FADE_SECONDS = 1.6
GAS_SPREAD = 4.0  # thin gas is drawn up to this many times wider than dense gas


# Automatic slow motion near black holes. Close to a hole things happen in
# hours or minutes; at a year per second a body would go round it several
# times between two frames and its fall could not be seen. So a step never
# advances more than this fraction of the shortest dynamical timescale
# (nbodystate.min_timescale) of any pair that includes a black hole: an
# orbit then takes at least 2 pi / SLOW_MOTION = 63 steps. Only the pace
# changes, not the motion.
#
# For the same reason a body on screen near a hole moves at most
# PIXELS_PER_STEP pixels per step relative to it: a plunge covers a tenth of
# the remaining distance per step by the rule above, which crosses the view
# in a few frames.
#
# The slowing stops at what can be seen: separations count as at least
# `visible` (a few pixels of the current view) and at least the hole's
# handover radius, and the pace never drops below 1 / MAX_SLOW_MOTION.
# Without that, one fragment a millionth of an AU from a small hole slowed
# time a billion times and everything else stood still (the "frozen body"
# of 2026-11-22). What happens below that scale then happens within a step.
SLOW_MOTION = 0.1
MAX_SLOW_MOTION = 3.0e4
VISIBLE_PIXELS = 4.0
PIXELS_PER_STEP = 1.5
UNRESOLVED = 0.5  # see NBodyModule._accrete_unresolved
CLOSE_STEP = 3.0  # see NBodyModule._close_pace


def step_time(state, sim_dt, visible=0.0, zone=0.0):
    """Simulated time one physics step advances `state` (see SLOW_MOTION).

    `visible` is the world size of VISIBLE_PIXELS and `zone` the distance
    from a hole within which the pixel rule applies (both 0 without a view).
    """
    if not state.black_hole.any():
        return sim_dt
    travel = visible / VISIBLE_PIXELS * PIXELS_PER_STEP / SLOW_MOTION
    tau = hole_timescale(
        state.pos, state.vel, state.mass, G, state.black_hole, state.radius, visible, travel, zone
    )
    return min(sim_dt, max(SLOW_MOTION * tau, sim_dt / MAX_SLOW_MOTION))


def advance_step(integrator, state, dt, on_contact):
    """One physics step of dt with substeps, as NBodyModule.step does it.

    `on_contact(pair)` is called when two bodies touch. It returns the state
    to continue with (a merger replaces the state), or None to end the step
    there. Returns False if the step was cut short by STEP_BUDGET (the
    state is then less than dt further).
    """
    remaining, start, substeps = dt, time.perf_counter(), 0
    while remaining > 1e-9 * dt:
        if time.perf_counter() - start > STEP_BUDGET:
            integrator.last_substeps = substeps
            return False
        pair, t = integrator.advance(state, remaining, check_contact=True, max_substeps=32)
        substeps += integrator.last_substeps
        remaining -= t
        if pair is not None:
            state = on_contact(pair)
            if state is None:
                break
    integrator.last_substeps = substeps
    return True


class PathPrediction:
    """The future path of one body, from a trial run kept ahead of the simulation.

    The trial run takes the same steps as the module (advance_step) on a
    copy of the state, so it stays on the simulation's own path, and its
    points remain valid while the simulation advances: each frame only the
    far end is extended, within PREDICT_BUDGET. It is thrown away when its
    key changes (another body, an edit, a merger) or the simulation leaves
    the predicted path.
    """

    def __init__(self, module, key, idx, extra):
        self.key = key
        old = module.nb
        state = NBodyState(old.N + (extra is not None))
        state.pos[: old.N] = old.pos
        state.vel[: old.N] = old.vel
        state.mass[: old.N] = old.mass
        state.radius[: old.N] = old.radius
        state.black_hole[: old.N] = old.black_hole
        state.debris[: old.N] = old.debris
        if extra is not None:
            state.pos[-1] = extra["pos"]
            state.vel[-1] = extra["vel"]
            state.mass[-1] = extra["mass"]
            state.radius[-1] = extra["phys_radius"]
            state.black_hole[-1] = extra.get("black_hole", False)
            idx = old.N
        self.idx = idx
        self.n_old = old.N  # the followed point is computed over the existing bodies only
        self.state = state
        self.integrator = make_integrator(module.gravity)
        self.target = module.follow_target()
        self.step0 = module.step_index
        self.steps = 0  # trial steps taken
        self.base = 0  # stride index of rel[0]
        self.rel = [self._rel(module)]  # position relative to the followed point, per stride
        self.ended = False  # the body hit another: the path stops there
        self.lead = 0  # steps the trial run was ahead of the simulation after the last extend

    def _rel(self, module):
        st = self.state
        anchor = module.anchor_point(st.pos[: self.n_old], st.mass[: self.n_old], self.target)
        return st.pos[self.idx].copy() if anchor is None else st.pos[self.idx] - anchor

    def _hit(self, pair):
        # Other bodies touching each other are let pass in the trial run.
        self.ended = self.idx in pair
        return None if self.ended else self.state

    def extend(self, module, ahead_steps):
        now = module.step_index - self.step0
        goal = now + ahead_steps
        # Where the last frame's lead is restored (at least: not overtaken).
        keep = now + min(max(self.lead, PREVIEW_STRIDE), ahead_steps)
        start = time.perf_counter()
        caught_up = None  # when the lead was restored; the budget for growing counts from there
        while not self.ended and self.steps < goal:
            t = time.perf_counter()
            if self.steps < keep:
                if t - start > PREDICT_KEEP_UP:
                    break
            else:
                if caught_up is None:
                    caught_up = t
                if t - caught_up > PREDICT_BUDGET:
                    break
            advance_step(
                self.integrator,
                self.state,
                step_time(self.state, module.sim_dt, module.visible_scale, module.view_zone),
                self._hit,
            )
            self.steps += 1
            if self.steps % PREVIEW_STRIDE == 0 or self.ended:
                self.rel.append(self._rel(module))
        self.lead = self.steps - now

    def points(self, module, here, ahead_steps):
        """The path from `here` (the body's position now) on, or None if the
        simulation is no longer on the predicted path."""
        k = module.step_index - self.step0
        j = k // PREVIEW_STRIDE - self.base
        anchor = module.anchor_point(module.nb.pos, module.nb.mass, self.target)
        here = here if anchor is None else here - anchor
        if j >= len(self.rel):
            return None  # the simulation overtook the trial run
        if k % PREVIEW_STRIDE == 0:
            moved = np.linalg.norm(self.state.vel[self.idx]) * module.sim_dt * PREVIEW_STRIDE
            if np.linalg.norm(self.rel[j] - here) > 0.5 * moved + 1e-12:
                return None
        if j > 256:
            del self.rel[:j]
            self.base += j
            j = 0
        last = (k + ahead_steps) // PREVIEW_STRIDE - self.base
        pts = np.array([here, *self.rel[j + 1 : last + 1]])
        return pts if anchor is None else pts + anchor


class NBodyModule(BaseModule):
    def __init__(self) -> None:
        # Changing the scene, or Restart, makes the app rebuild this module
        # and set scene_name after construction, so the scene is loaded in
        # setup(), not here. Loading a different scene first would discard
        # the restart point.
        self.scene_name = DEFAULT_SCENE
        self.gravity = DEFAULT_GRAVITY
        self.sheet_mode = "Auto"
        self.wave_gain_log = 0.0
        self.specs = ()
        self.sheet = SpacetimeSheet()

        self.background = Starfield()
        self.camera = None  # set in render(); the editor needs it
        self.editor = SceneEditor(self)
        # Set when body colours, radii or count change; the body renderer is
        # rebuilt at the next render().
        self.renderer_dirty = False

    def load_scene(self, name):
        """Load scene `name`: from its restart point if it has one (Restart
        after edits), otherwise fresh from its definition. A restart point
        for another scene is discarded, so switching scenes starts fresh."""
        scene = load_restart_point(name)
        if scene is None:
            clear_restart_point()
            scene = SCENES[name](G)

        self.scene = scene
        self.loaded_scene_name = name
        self.nb = scene.state
        # Positions and velocities Restart returns to.
        self.start_state = deepcopy(scene.state)
        self.integrator = make_integrator(self.gravity)
        # Set when bodies merged since the start state; the start state then
        # no longer matches the current list of names and looks.
        self.merged_since_start = False
        self.renderer = NBodyRenderer(
            scene.colors, scene.radii, holes=self.nb.black_hole + 2 * self.nb.gas
        )
        self.sheet.clear()
        # One trail sample per physics step.
        # Simulated years per physics step. The app's dt is wall-clock
        # seconds per step; a scene may run slower than a year per second.
        self.sim_dt = self.dt * scene.years_per_second
        self.trails = TrailRenderer(self.nb.N, min(MAX_TRAIL_YEARS / self.dt, MAX_TRAIL_STEPS))
        # Camera distance to apply at the first render().
        self.pending_view_distance = scene.view_distance
        self.pending_view_pitch = scene.view_pitch

        # A followed body carried over from another scene may not exist here.
        if editor_settings["follow"] not in (FOLLOW_NONE, FOLLOW_COM, *scene.names):
            editor_settings["follow"] = FOLLOW_NONE
        self.followed = None  # what the camera followed on the previous frame
        self.followed_pos = None  # and where it was

        self.step_index = 0  # physics steps taken
        self.edit_version = 0  # counts changes that make predicted paths invalid
        self.prediction = None  # PathPrediction of the selected or placed body
        self.slowed = False  # the last step was cut short by STEP_BUDGET
        self.elapsed = 0.0  # simulated years so far
        self.last_step_time = 0.0  # simulated years advanced by the last step
        self.slow_motion = 1.0  # how many times slower than asked the last step ran
        self.visible_scale = 0.0  # world size of VISIBLE_PIXELS in the current view
        self.view_zone = 0.0  # distance from a hole that counts as on screen
        # Swallowed bodies shown frozen at their hole's edge, fading:
        # (hole name, direction, colour, drawn radius, kind, start time).
        self.fading = []
        self.rng = np.random.default_rng(2026)  # for debris
        # Light bodies inside a hole's capture sphere, on exact geodesics.
        self.close = ClosePasses(G, C_AU_PER_YEAR)
        # Flashes of collisions and mergers: (position, colour, size [AU], start time).
        self.flashes = []

        self._rebase()
        self.build_specs()

    def energy(self):
        """The energy the current gravity conserves: Newtonian, or 1PN (EIH)."""
        nb = self.nb
        if self.gravity == RELATIVITY:
            return e_1pn(nb.pos, nb.vel, nb.mass)
        return e(nb.pos, nb.vel, nb.mass)

    def rebase_diagnostics(self):
        """Take the state as it will be at the end of this step as the reference for
        the conservation diagnostics (done once, however many events ask for it)."""
        self.rebase_pending = True

    def _rebase(self):
        self.rebase_pending = False
        nb = self.nb
        self.start_e = self.energy()
        self.start_momentum = total_momentum(nb)
        self.start_angular_momentum = total_angular_momentum(nb)

    def build_specs(self):
        self.specs = (
            Choice(
                attr_name="scene_name",
                label="Scene",
                default=DEFAULT_SCENE,
                options=tuple(SCENES),
                need_reset=True,
            ),
            Choice(
                attr_name="gravity",
                label="Gravity",
                default=DEFAULT_GRAVITY,
                options=(RELATIVITY, NEWTON),
                need_reset=True,
            ),
            Choice(
                attr_name="sheet_mode",
                label="Spacetime sheet",
                default="Auto",
                options=("Auto", "On", "Off"),
            ),
            SliderFloat(
                attr_name="wave_gain_log",
                label="Wave height [log10 gain]",
                default=0.0,
                min=0.0,
                max=12.0,
            ),
        )

    # ------------------------------------------------ editing (scene editor)

    def save_restart(self):
        """Store the start state with the current names and looks as this
        scene's restart point."""
        if self.merged_since_start:
            # The names and looks now describe the merged system, so the
            # merged system is what Restart must return to.
            self.start_state = deepcopy(self.nb)
            self.merged_since_start = False
        scene = self.scene
        save_restart_point(
            self.scene_name,
            self.start_state,
            scene.names,
            scene.colors,
            scene.radii,
            scene.view_distance,
        )

    def state_edited(self, moved=None):
        """Call after changing masses, positions or velocities in self.nb, or
        adding or removing bodies.

        The system as it is now becomes what Restart returns to. `moved` is
        the index of a body whose position jumped; its trail is cleared so no
        line is drawn across the jump.
        """
        self.integrator.invalidate()
        self.edit_version += 1
        self.renderer_dirty = True  # a body may have become a black hole
        # The conserved totals changed; measure from here on.
        self.rebase_diagnostics()
        if moved is not None:
            self.trails.forget(moved)
        self.start_state = deepcopy(self.nb)
        self.save_restart()

    def style_edited(self):
        """Call after changing a body's colour or drawn radius. Restart keeps
        the new look but still returns to the same positions and velocities."""
        self.renderer_dirty = True
        self.save_restart()

    def rename_body(self, i, name):
        old = self.scene.names[i]
        self.scene.names[i] = name
        if editor_settings["follow"] == old:
            editor_settings["follow"] = name
            self.followed = name
        self.save_restart()

    def _resize_state(self, keep):
        """Replace the state with one holding the bodies at indices `keep`,
        plus an empty slot for each entry of keep that is None."""
        old = self.nb
        nb = NBodyState(len(keep))
        new = np.array([k for k, old_i in enumerate(keep) if old_i is not None], dtype=int)
        src = np.array([old_i for old_i in keep if old_i is not None], dtype=int)
        for name in ("pos", "vel", "mass", "radius", "black_hole", "debris", "gas"):
            getattr(nb, name)[new] = getattr(old, name)[src]
        self.nb = self.scene.state = nb

    def add_body(self, body):
        """Add a body ({"mass", "pos", "vel", "color", "radius", "phys_radius",
        "black_hole"}) to the running system."""
        scene = self.scene
        old_n = self.nb.N
        self._resize_state([*range(old_n), None])
        self.nb.pos[old_n] = body["pos"]
        self.nb.vel[old_n] = body["vel"]
        self.nb.mass[old_n] = body["mass"]
        self.nb.radius[old_n] = body["phys_radius"]
        self.nb.black_hole[old_n] = body.get("black_hole", False)

        # A name not already taken, so following by name stays unambiguous.
        k = old_n + 1
        while f"Body {k}" in scene.names:
            k += 1
        scene.names.append(f"Body {k}")
        scene.colors.append(tuple(body["color"]))
        scene.radii.append(body["radius"])

        self.trails.add_bodies(1)
        self.renderer_dirty = True
        self.state_edited()

    def _drop_body(self, i, replacement=None):
        """Delete body i everywhere. `replacement` is the body that takes
        over its follow and selection (after a merge), or None."""
        scene = self.scene
        if editor_settings["follow"] == scene.names[i]:
            editor_settings["follow"] = (
                FOLLOW_NONE if replacement is None else scene.names[replacement]
            )

        self._resize_state([k for k in range(self.nb.N) if k != i])
        del scene.names[i]
        del scene.colors[i]
        del scene.radii[i]

        self.trails.remove_body(i)
        self.editor.on_body_removed(i, replacement)
        self.integrator.invalidate()
        self.edit_version += 1
        self.renderer_dirty = True

    def remove_body(self, i):
        """Remove body i from the running system."""
        self._drop_body(i)
        self.state_edited()

    def merge(self, i, j):
        """Merge two touching bodies (see merge_pair). The heavier one
        remains, except that a black hole always remains: it absorbs the
        other body whatever their masses."""
        nb = self.nb
        if nb.black_hole[i] != nb.black_hole[j]:
            keep, gone = (i, j) if nb.black_hole[i] else (j, i)
            absorbed = True
        else:
            keep, gone = (i, j) if nb.mass[i] >= nb.mass[j] else (j, i)
            absorbed = False
        merge_pair(nb, keep, gone)
        if not absorbed:
            # Drawn size: same volume rule as the physical radius. A black
            # hole that absorbs a body keeps its drawn size.
            radii = self.scene.radii
            radii[keep] = float(np.cbrt(radii[keep] ** 3 + radii[gone] ** 3))

        self._drop_body(gone, replacement=keep)
        self.merged_since_start = True
        # The merger itself loses kinetic energy; measure the integration
        # error from here on.
        self.rebase_diagnostics()

    def _spawn(self, pieces, names, color, drawn_radius, gas=False):
        """Add debris (rows of pos, vel, mass, radius) made by the simulation itself."""
        old_n = self.nb.N
        k = len(names)
        self._resize_state([*range(old_n), *([None] * k)])
        nb = self.nb
        nb.pos[old_n:], nb.vel[old_n:], nb.mass[old_n:], nb.radius[old_n:] = pieces
        nb.debris[old_n:] = True
        nb.gas[old_n:] = gas
        self.scene.names.extend(names)
        self.scene.colors.extend([tuple(color)] * k)
        self.scene.radii.extend([drawn_radius] * k)
        self.trails.add_bodies(k)
        self.integrator.invalidate()
        self.edit_version += 1
        self.renderer_dirty = True
        self.merged_since_start = True
        self.rebase_diagnostics()

    def _debris_names(self, parent, k):
        used = set(self.scene.names)
        names, n = [], 1
        while len(names) < k:
            name = f"{parent} debris {n}"
            if name not in used:
                names.append(name)
            n += 1
        return names

    def collide(self, i, j):
        """Two ordinary bodies touch: they merge, or shatter if they hit hard
        enough (nbody_collisions.py)."""
        nb = self.nb
        m1, m2 = nb.mass[i], nb.mass[j]
        speed = float(np.linalg.norm(nb.vel[i] - nb.vel[j]))
        both_whole = not (nb.debris[i] or nb.debris[j])
        kept = largest_remnant_fraction(m1, m2, speed, G) if both_whole else 1.0
        keep = i if m1 >= m2 else j
        keep_name = self.scene.names[keep]
        self.merge(i, j)
        if kept >= MERGE_FRACTION or self.nb.N + N_FRAGMENTS > MAX_BODIES:
            return
        nb = self.nb
        keep = self.scene.names.index(keep_name)
        total = nb.mass[keep]
        impact_energy = 0.5 * m1 * m2 / total * speed**2
        nb.mass[keep] = kept * total
        nb.radius[keep] *= kept ** (1.0 / 3.0)
        self.scene.radii[keep] *= kept ** (1.0 / 3.0)
        offsets, vel, m, radius = fragments(
            nb.mass[keep], nb.radius[keep], (1.0 - kept) * total, impact_energy, G, self.rng
        )
        drawn = self.scene.radii[keep] * (m / nb.mass[keep]) ** (1.0 / 3.0)
        parent = self.scene.radii[keep] / kept ** (1.0 / 3.0)  # drawn size before the impact
        # Display: molten fragments, and a flash where it happened.
        molten = tuple(
            0.3 * c + 0.7 * h for c, h in zip(self.scene.colors[keep], MOLTEN, strict=True)
        )
        self.flashes.append(
            (nb.pos[keep].copy(), (1.0, 0.8, 0.5), 6.0 * parent, time.perf_counter())
        )
        self._spawn(
            (nb.pos[keep] + offsets, nb.vel[keep] + vel, m, radius),
            self._debris_names(keep_name, N_FRAGMENTS),
            molten,
            max(drawn, 0.3 * parent),
        )

    def disrupt(self, body):
        """A black hole's tide pulls `body` apart: it becomes N_TIDAL pieces of debris."""
        nb = self.nb
        name, color, drawn = (
            self.scene.names[body],
            self.scene.colors[body],
            self.scene.radii[body],
        )
        pos, vel, mass, radius = (
            nb.pos[body].copy(),
            nb.vel[body].copy(),
            nb.mass[body],
            nb.radius[body],
        )
        offsets = tidal_pieces(radius, self.rng)
        self._drop_body(body)
        self._spawn(
            # The pieces' own radius only matters for being captured by a hole.
            (pos + offsets, vel, mass / N_TIDAL, 1e-3 * radius),
            self._debris_names(name, N_TIDAL),
            color,
            0.3 * drawn,
            gas=True,
        )

    def contact(self, pair):
        """Two bodies touch (nbodystate.first_contact): decide what happens."""
        i, j = pair
        nb = self.nb
        if nb.black_hole[i] == nb.black_hole[j]:
            if nb.black_hole[i]:
                size = 4.0 * CAPTURE_RADIUS * (nb.radius[i] + nb.radius[j])
                where = 0.5 * (nb.pos[i] + nb.pos[j])
                self.flashes.append((where, (0.75, 0.85, 1.0), size, time.perf_counter()))
                self.merge(i, j)
            else:
                self.collide(i, j)
            return
        hole, body = (i, j) if nb.black_hole[i] else (j, i)
        light = nb.mass[body] <= LIGHT_BODY * nb.mass[hole]
        touching = (
            nb.radius[body] + (HANDOVER_RADIUS if light else CAPTURE_RADIUS) * nb.radius[hole]
        )
        distance = np.linalg.norm(nb.pos[body] - nb.pos[hole])
        tidal = nb.radius[body] * (nb.mass[hole] / max(nb.mass[body], 1e-300)) ** (1.0 / 3.0)
        if (
            not nb.debris[body]
            and tidal > 1.05 * touching
            and distance > 1.05 * touching
            and nb.N + N_TIDAL <= MAX_BODIES
        ):
            self.disrupt(body)
            return
        meta = {
            "name": self.scene.names[body],
            "color": self.scene.colors[body],
            "drawn": self.scene.radii[body],
            "mass": float(nb.mass[body]),
            "radius": float(nb.radius[body]),
            "debris": bool(nb.debris[body]),
            "gas": bool(nb.gas[body]),
        }
        direction = nb.pos[body] - nb.pos[hole]
        if not (light and self._can_follow(hole, body)):
            self._fade(
                self.scene.names[hole],
                direction,
                self.scene.colors[body],
                self.scene.radii[body],
                bool(nb.gas[body]),
            )
        if light and self.close.enter(
            self.scene.names[hole],
            nb.mass[hole],
            nb.pos[body] - nb.pos[hole],
            nb.vel[body] - nb.vel[hole],
            meta,
        ):
            # Followed on its exact orbit round the hole (nbody_plunge.py)
            # until it is swallowed or leaves again.
            self._drop_body(body)
            self.merged_since_start = True
            self.rebase_diagnostics()
            return
        self.merge(i, j)

    def _style_gas(self):
        """Display: draw each piece of gas about as wide as the gap to its
        neighbours, and fainter the wider it is, so that thin gas reads as a
        faint cloud and not as separate dots."""
        nb, scene = self.nb, self.scene
        radii = np.array(scene.radii, dtype=float)
        colors = np.array(scene.colors, dtype=float)
        gas = np.flatnonzero(nb.gas)
        if len(gas) > 3:
            p = nb.pos[gas]
            d = np.linalg.norm(p[:, None] - p[None], axis=2)
            gap = np.partition(d, 3, axis=1)[:, 3]  # distance to the third nearest piece
            wide = np.clip(0.8 * gap, radii[gas], GAS_SPREAD * radii[gas])
            colors[gas] *= np.clip(radii[gas] / wide, 0.3, 1.0)[:, None] ** 0.7
            radii[gas] = wide
        order = self.renderer.order
        kinds = (nb.black_hole + 2 * nb.gas).astype(float)
        self.renderer.set_style(colors[order], radii[order], kinds[order])

    def _can_follow(self, hole, body):
        return self.close.count < PLUNGE_CAPACITY

    def _fade(self, hole_name, direction, color, drawn, gas):
        """Show a swallowed body frozen at the hole's edge, fading (see FADE_SECONDS)."""
        norm = np.linalg.norm(direction)
        unit = direction / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0])
        self.fading.append((hole_name, unit, tuple(color), drawn, gas, time.perf_counter()))
        del self.fading[:-PLUNGE_CAPACITY]

    def _fading_sprites(self):
        now = time.perf_counter()
        names = self.scene.names
        self.fading = [f for f in self.fading if now - f[5] < FADE_SECONDS and f[0] in names]
        if not self.fading:
            return None
        pos, colors, radii, kinds = [], [], [], []
        for hole_name, unit, color, drawn, gas, start in self.fading:
            h = names.index(hole_name)
            g = 1.0 - (now - start) / FADE_SECONDS
            # At the edge of the hole as it is drawn, flattened against it.
            pos.append(self.nb.pos[h] + unit * self.scene.radii[h])
            # Redshift: blue and green go first, then the whole dims.
            colors.append(np.array(color) * np.array([1.0, g, g * g]) * g**1.5)
            radii.append(min(drawn, 0.6 * self.scene.radii[h]) * (0.6 + 0.4 * g))
            kinds.append(2.0)  # drawn soft, as a glow on the horizon
        return np.array(pos), np.array(colors), np.array(radii), np.array(kinds)

    def _accrete_unresolved(self, advanced):
        """Count debris bound to a hole far below what can be followed as accreted.

        A piece whose dynamical time about a hole, sqrt(r^3 / (G M)), is
        shorter than UNRESOLVED of the step just taken goes round the hole
        many times per step; it cannot be integrated at this pace and is
        far inside a pixel. If it is also bound to the hole (v^2 / 2 <
        G M / r), it belongs to the small disc that forms round the hole
        and drains into it: the hole is given its mass and momentum now.
        (Whole bodies are never treated this way, only debris.)
        """
        nb = self.nb
        pieces = np.flatnonzero(nb.debris)
        if len(pieces) == 0:
            return
        gone = {}
        for h in np.flatnonzero(nb.black_hole):
            rel = nb.pos[pieces] - nb.pos[h]
            r = np.linalg.norm(rel, axis=1)
            v2 = np.sum((nb.vel[pieces] - nb.vel[h]) ** 2, axis=1)
            gm = G * nb.mass[h]
            lost = (np.sqrt(r**3 / gm) < UNRESOLVED * advanced) & (
                0.5 * v2 < gm / np.maximum(r, 1e-300)
            )
            for k in pieces[lost]:
                gone.setdefault(int(k), int(h))
        if not gone:
            return
        for k, h in gone.items():
            total = nb.mass[h] + nb.mass[k]
            nb.vel[h] = (nb.mass[h] * nb.vel[h] + nb.mass[k] * nb.vel[k]) / total
            nb.radius[h] *= total / nb.mass[h]
            nb.mass[h] = total
            self._fade(
                self.scene.names[h],
                nb.pos[k] - nb.pos[h],
                self.scene.colors[k],
                self.scene.radii[k],
                bool(nb.gas[k]),
            )
        for k in sorted(gone, reverse=True):
            self._drop_body(k)
        self.merged_since_start = True
        self.rebase_diagnostics()

    def _close_pace(self, sim_dt):
        """Longest step while bodies are on their exact orbits round a hole whose
        handover sphere is large enough to see: CLOSE_STEP G M / c^3, so that
        their winding in is watched at its own pace (see SLOW_MOTION)."""
        pace = sim_dt
        masses = self._hole_masses()
        for name in self.close.holes:
            if name not in masses:
                continue
            r_g = G * masses[name] / C_AU_PER_YEAR**2
            if 2.0 * HANDOVER_RADIUS * r_g >= self.visible_scale > 0.0:
                pace = min(pace, CLOSE_STEP * r_g / C_AU_PER_YEAR)
        return max(pace, sim_dt / MAX_SLOW_MOTION)

    def _hole_masses(self):
        nb = self.nb
        return {self.scene.names[k]: float(nb.mass[k]) for k in np.flatnonzero(nb.black_hole)}

    def _close_passes(self):
        """Advance the bodies inside capture spheres; settle those that were swallowed or left."""
        if self.close.count == 0:
            return
        swallowed, left = self.close.advance(self.last_step_time, self._hole_masses())
        nb = self.nb
        for hole_name, meta, direction in swallowed:
            if hole_name not in self.scene.names:
                continue
            h = self.scene.names.index(hole_name)
            self._fade(hole_name, direction, meta["color"], meta["drawn"], meta["gas"])
            # The hole takes the mass; its horizon grows in proportion. (The
            # body's momentum relative to the hole is neglected: it is at
            # most LIGHT_BODY of the hole's mass.)
            nb.radius[h] *= (nb.mass[h] + meta["mass"]) / nb.mass[h]
            nb.mass[h] += meta["mass"]
            self.integrator.invalidate()
        for hole_name, meta, rel, vel in left:
            if hole_name not in self.scene.names or self.nb.N >= MAX_BODIES:
                continue
            h = self.scene.names.index(hole_name)
            pos, v = self.nb.pos[h] + rel, self.nb.vel[h] + vel
            name = (
                meta["name"]
                if meta["name"] not in self.scene.names
                else self._debris_names(meta["name"], 1)[0]
            )
            self._spawn(
                (pos[None], v[None], meta["mass"], meta["radius"]),
                [name],
                meta["color"],
                meta["drawn"],
                gas=meta["gas"],
            )
            self.nb.debris[-1] = meta["debris"]
        if swallowed or left:
            self.rebase_diagnostics()

    def discard_edits(self):
        """Forget all edits and editor settings: the next rebuild starts as the module does."""
        clear_restart_point()
        reset_editor_settings()

    # ---------------------------------------------------------------- follow

    def follow_target(self):
        """What the camera follows: None, "com" or a body index."""
        f = editor_settings["follow"]
        if f == FOLLOW_COM:
            return "com" if self.nb.N > 0 else None
        if f in self.scene.names:
            return self.scene.names.index(f)
        return None

    def anchor_point(self, pos, mass, target):
        """Position of the followed point for positions `pos` of shape (..., N, 3).

        Returns None if nothing is followed or the centre of mass is undefined.
        """
        if target is None:
            return None
        if target == "com":
            total = mass.sum()
            if total <= 0.0:
                return None
            return np.einsum("n,...nd->...d", mass, pos) / total
        return pos[..., target, :]

    def anchor_state(self):
        """Position and velocity that placed bodies are relative to: the
        followed point, or the origin at rest."""
        nb = self.nb
        target = self.follow_target()
        pos = self.anchor_point(nb.pos, nb.mass, target)
        if pos is None:
            return np.zeros(3), np.zeros(3)
        return pos.copy(), self.anchor_point(nb.vel, nb.mass, target).copy()

    def update_follow(self, camera):
        """Move the camera target along with the followed body or centre of mass.

        The target is shifted by the followed point's displacement since the
        last frame rather than pinned to it, so panning still works while
        following.
        """
        target = self.follow_target()
        pos = self.anchor_point(self.nb.pos, self.nb.mass, target)
        if pos is None:
            self.followed = None
            return

        key = editor_settings["follow"]
        if key != self.followed:
            # Just started following: centre on it, and zoom in on a body.
            camera.target[:] = pos
            if target != "com":
                camera.distance = min(
                    camera.distance, FOLLOW_DISTANCE_RADII * self.scene.radii[target]
                )
        else:
            camera.target += pos - self.followed_pos

        self.followed = key
        self.followed_pos = pos.copy()

    # ------------------------------------------------------------ prediction

    def predict_path(self, idx=None, extra=None):
        """Points along the future path of one body, from a trial run.

        The trial run uses the module's own integrator and force on a copy
        of the current state, so the prediction includes every body's pull.
        Predicts body `idx`, or, with `extra` (a body dict), the path that
        body would take if it were added now. With something followed, the
        path is given as seen from it, like the trails. The path is kept
        between frames (see PathPrediction) and may be shorter than the
        look-ahead while it is still being computed.
        """
        ahead_steps = min(int(editor_settings["lookahead_years"] / self.dt), MAX_LOOKAHEAD_STEPS)
        if ahead_steps < PREVIEW_STRIDE:
            return None

        if extra is None:
            key = (idx, self.follow_target(), self.edit_version)
            here = self.nb.pos[idx]
        else:
            # A body being placed is not in the simulation: its path starts
            # from the present state, so it is redone when anything moves.
            here = np.asarray(extra["pos"], dtype=float)
            key = (
                "extra",
                self.follow_target(),
                self.edit_version,
                self.step_index,
                tuple(here),
                tuple(extra["vel"]),
                extra["mass"],
                extra["phys_radius"],
            )

        for _ in range(2):
            if self.prediction is None or self.prediction.key != key:
                self.prediction = PathPrediction(self, key, idx, extra)
            self.prediction.extend(self, ahead_steps)
            points = self.prediction.points(self, here, ahead_steps)
            if points is not None:
                return points
            self.prediction = None  # left the predicted path: start again
        return None

    # ------------------------------------------------------------ lifecycle

    def step(self, dt):
        # Advance, merging bodies that touch. advance() stops early at a
        # contact; the rest of the step continues after the merge.
        sim_dt = dt * self.scene.years_per_second
        advanced = min(
            step_time(self.nb, sim_dt, self.visible_scale, self.view_zone),
            self._close_pace(sim_dt),
        )
        self.slow_motion = sim_dt / advanced
        self.slowed = not advance_step(self.integrator, self.nb, advanced, self._merge_contact)
        self.elapsed += advanced
        self.last_step_time = advanced
        self.step_index += 1
        self.trails.record(self.nb.pos)
        self.trails.valid[self.nb.gas] = 0  # a cloud of gas is drawn without trails
        self._close_passes()
        self._accrete_unresolved(advanced)
        if self.rebase_pending:
            self._rebase()
        nb = self.nb
        acc = self.integrator.acc_cache
        if acc is not None and acc.shape == nb.pos.shape:
            self.sheet.record(quadrupole_ddot(nb.pos, nb.vel, acc, nb.mass), G, C_AU_PER_YEAR)

    def _merge_contact(self, pair):
        self.contact(pair)
        return self.nb

    def diagnostics(self):
        if self.rebase_pending:
            self._rebase()
        nb = self.nb
        e_change = np.abs(self.energy() - self.start_e)
        if self.start_e == 0.0:
            # No energy scale to divide by: report the absolute change.
            energy_line = f"Absolute Energy Error: {e_change}"
        else:
            energy_line = f"Relative Energy Error: {e_change / np.abs(self.start_e)}"
        diff_momentum = total_momentum(nb) - self.start_momentum
        diff_angular_momentum = total_angular_momentum(nb) - self.start_angular_momentum

        return (
            f" {energy_line}\n Momentum Change: {diff_momentum}\n Angular Momentum Change: {diff_angular_momentum}"
            f"\n Substeps: {self.integrator.last_substeps}"
            + (f"\n {self.scene.info}" if self.scene.info else "")
            + f"\n Time: {self._elapsed()}"
            + (
                f"\n Slow motion x{self.slow_motion:.0f}: close to a black hole"
                if self.slow_motion > 1.5
                else ""
            )
            + (
                "\n Slow motion: a close encounter needs more substeps than allowed per step"
                if self.slowed
                else ""
            )
        )

    def _elapsed(self):
        years = self.elapsed
        if years >= 0.1:
            return f"{years:.2f} yr"
        hours = years * 8766.0
        return f"{hours:.2f} h" if hours >= 1.0 else f"{hours * 60.0:.1f} min"

    def draw_ui(self):
        self.editor.update(self.camera, G)

    def setup(self, ctx):
        self.load_scene(self.scene_name)

        self.renderer.setup(ctx, self.nb.N)
        self.sheet.setup(ctx)
        self.trails.setup(ctx)
        self.background.setup(ctx)
        self.plunge_renderer = NBodyRenderer(
            np.zeros((PLUNGE_CAPACITY, 3)), np.zeros(PLUNGE_CAPACITY), min_radius_px=1.0
        )
        self.plunge_renderer.setup(ctx, PLUNGE_CAPACITY)

    def render(self, ctx, camera):
        self.camera = camera
        if self.pending_view_distance is not None:
            camera.distance = self.pending_view_distance
            camera.pitch = self.pending_view_pitch
            self.pending_view_distance = None

        # GL objects are rebuilt here, not in draw_ui(): replacing them in the
        # middle of the ImGui frame breaks the ImGui backend's state restore.
        if self.renderer_dirty:
            self.renderer.release()
            self.renderer = NBodyRenderer(
                self.scene.colors, self.scene.radii, holes=self.nb.black_hole + 2 * self.nb.gas
            )
            self.renderer.setup(ctx, self.nb.N)
            self.renderer_dirty = False

        self.update_follow(camera)
        self.background.render(ctx, camera)
        if self.sheet_mode == "On" or (self.sheet_mode == "Auto" and self.scene.sheet):
            nb = self.nb
            self.sheet.render(
                ctx,
                camera,
                nb.pos,
                nb.mass,
                C_AU_PER_YEAR * self.sim_dt,
                G * nb.mass.sum() / C_AU_PER_YEAR**2,
                10.0**self.wave_gain_log,
            )
        if editor_settings["trail_years"] > 0.0:
            self.trails.render(
                ctx,
                camera,
                self.nb.pos,
                self.nb.mass,
                self.scene.colors,
                editor_settings["trail_years"] / self.dt,
                self.follow_target(),
            )
        # Bodies inside a hole's capture sphere, under the holes' black discs.
        falling = self.close.sprites(self.scene.names, self.nb.pos, self._hole_masses())
        if falling is not None:
            pos, colors, radii, kinds = falling
            self.plunge_renderer.set_style(colors, radii, kinds)
            self.plunge_renderer.render(ctx, camera, pos, count=len(pos))
        if self.nb.gas.any() and not self.renderer_dirty:
            self._style_gas()
        self.renderer.render(ctx, camera, self.nb.pos)
        # Swallowed bodies, frozen on the holes' edges and fading.
        frozen = self._fading_sprites()
        if frozen is not None:
            pos, colors, radii, kinds = frozen
            self.plunge_renderer.set_style(colors, radii, kinds)
            self.plunge_renderer.render(ctx, camera, pos, count=len(pos))
        self.visible_scale = (
            VISIBLE_PIXELS
            * 2.0
            * camera.distance
            * np.tan(camera.fov / 2.0)
            / max(ctx.viewport[3], 1)
        )
        self.view_zone = camera.distance

    def teardown(self):
        self.renderer.release()
        self.sheet.release()
        self.trails.release()
        self.background.release()
        self.plunge_renderer.release()
