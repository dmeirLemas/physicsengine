import time
from copy import deepcopy
from functools import partial

import numpy as np
from imgui_bundle import imgui

from physengine.core.module import BaseModule
from physengine.core.scalebar import AU_M, draw_scale_bar
from physengine.core.spec import Choice, SliderFloat
from physengine.modules import scenario_files
from physengine.modules.nbody_binary import (
    NO_PAIRS,
    UNRESOLVED_PERIOD,
    Held,
    advance_pairs,
    find_pairs,
    held_force,
    held_timescale,
    internal_acc,
    merged_state,
)
from physengine.modules.nbody_collisions import (
    MERGE_FRACTION,
    N_FRAGMENTS,
    N_TIDAL,
    fragments,
    is_torn_before_horizon,
    largest_remnant_fraction,
    tidal_pieces,
    tidal_radius,
)
from physengine.modules.nbody_debris import split_acc
from physengine.modules.nbody_editor import (
    FOLLOW_COM,
    FOLLOW_NONE,
    HOLE_MIN_PX,
    SceneEditor,
    clear_restart_point,
    load_restart_point,
    save_restart_point,
)
from physengine.modules.nbody_editor import reset_settings as reset_editor_settings
from physengine.modules.nbody_editor import settings as editor_settings
from physengine.modules.nbody_plunge import CAPACITY as PLUNGE_CAPACITY
from physengine.modules.nbody_plunge import (
    FREEZE_LOOK,
    FROZEN_OUTSIDE,
    ClosePasses,
    redshift_look,
)
from physengine.modules.nbody_renderer import NBodyRenderer, gas_strands
from physengine.modules.nbody_scenes import CUSTOM as CUSTOM_SCENE
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
    hole_pace,
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
    if len(r) == 0:  # every body is debris: no pairs
        return 0.0
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
    # No pairs left when every body is debris.
    return float(np.min(tau * factor)) if len(tau) else np.inf


def make_integrator(gravity=NEWTON):
    """Leapfrog with adaptive substeps for close encounters; 1PN forces if asked.

    Its force, timescale and contact check treat the pairs in `integrator.held`
    (none, until hold_pairs finds some) as single bodies, see nbody_binary.py."""
    if gravity == RELATIVITY:
        held = Held(G, 1.0 / C_AU_PER_YEAR**2)
        integrator = Integrator(
            held_force(force_1pn, held, True),
            held_timescale(timescale_1pn, held),
            velocity_dependent=True,
        )
    else:
        held = Held(G, 0.0)
        integrator = Integrator(
            held_force(force_newton, held, False), held_timescale(timescale, held)
        )
    integrator.held = held
    return integrator


def hold_pairs(integrator, state, sim_dt, visible):
    """Find the bound pairs that cannot be seen and make `integrator` treat each as
    one body (nbody_binary.find_pairs): orbits quicker than UNRESOLVED_PERIOD
    steps of `sim_dt` and smaller than one pixel, or, round a black hole,
    than `visible`, the world size of VISIBLE_PIXELS, which is the smallest
    radius a hole is drawn with (HOLE_MIN_PX): the companion is under the
    hole's dot. Without a view (visible = 0) there are none. Returns the
    pairs."""
    held = integrator.held
    if held is None:
        return NO_PAIRS
    pairs = NO_PAIRS
    if visible > 0.0 and state.N >= 2:
        pairs = find_pairs(
            state.pos,
            state.vel,
            state.mass,
            state.radius,
            state.black_hole,
            state.debris,
            held.G,
            held.ic2,
            UNRESOLVED_PERIOD * sim_dt,
            visible / VISIBLE_PIXELS,
            visible,
            sim_dt,
            CAPTURE_RADIUS,
        )
    if not np.array_equal(pairs, held.pairs):
        held.pairs = pairs
        integrator.invalidate()  # the cached accelerations were for other pairs
    return pairs


DEFAULT_SCENE = "Sun-Earth"
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
# Sprites drawn on top of the bodies per frame: bodies on exact orbits round
# a hole, fading ones, torn stars.
OVERLAY_CAPACITY = 2 * PLUNGE_CAPACITY + 16
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
# ...during the first part of which a body that was swallowed away from the
# hole's edge as drawn (see _accrete_unresolved) is moved onto it.
FADE_APPROACH = 0.35
# (Black holes are never drawn smaller than nbody_editor.HOLE_MIN_PX pixels, a
# display minimum stated in the editor: a small hole's horizon is far below
# a pixel.)
# Drawn radius of a piece of gas / drawn radius of the star it came from.
# Small enough that the N_TIDAL blobs together cover what the star covered.
GAS_PIECE = 0.15
TRUE_SIZE_FROM, TRUE_SIZE_AT = 6.0, 3.0  # tidal radii, see NBodyModule._display_radii
NOTE_SECONDS = 4.0  # how long an on-screen note about an event stays


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
#
# Slow motion is for events, and events end (2026-11-23; before, a tear near
# a small hole left the scene x1500 slower for good). A pair of a hole and
# a body only counts (nbodystate.hole_pace)
# - if the body is further from the hole than can be told apart from it on
#   screen (`unseen_radii`): what goes on under the hole's dot is not slowed;
# - and it is unbound (a passage), or on its way to being captured, handed to
#   the exact orbit or torn, or on a bound orbit of at least FAST_ORBIT
#   steps. A faster bound orbit is left at the scene's pace like any moon's.
# The pixel rule applies to passages and to falls only. Debris that cannot
# be followed at the pace this leaves is counted as accreted
# (NBodyModule._accrete_unresolved).
SLOW_MOTION = 0.1
MAX_SLOW_MOTION = 3.0e4
VISIBLE_PIXELS = 4.0
PIXELS_PER_STEP = 1.5
FAST_ORBIT = 24.0  # physics steps: a bound orbit quicker than this is not slowed down for
UNRESOLVED_ORBIT = 2.0  # steps as taken, see NBodyModule._accrete_unresolved
CLOSE_STEP = 3.0  # see NBodyModule._close_pace


def unseen_radii(state, visible, drawn=None):
    """Per body: for a black hole, the distance from it within which nothing can be
    told apart from it on screen, the larger of `visible` (the world size of
    VISIBLE_PIXELS, which is also the smallest a hole is drawn, HOLE_MIN_PX)
    and its drawn radius. 0 for other bodies, and for all without a view."""
    out = np.zeros(state.N)
    if visible > 0.0:
        size = state.radius if drawn is None else np.asarray(drawn, dtype=float)
        out[state.black_hole] = np.maximum(size[state.black_hole], visible)
    return out


def step_time(state, sim_dt, visible=0.0, zone=0.0, drawn=None, pairs=NO_PAIRS):
    """Simulated time one physics step advances `state` (see SLOW_MOTION).

    `visible` is the world size of VISIBLE_PIXELS and `zone` the distance
    from a hole within which the pixel rule applies (both 0 without a view);
    `drawn` the bodies' drawn radii (default: their physical radii). Each
    of `pairs` (hold_pairs) counts as one body: what goes on inside a pair
    that cannot be seen is not slowed down for.
    """
    if not state.black_hole.any():
        return sim_dt
    state, drawn = merged_state(state, pairs, drawn)
    travel = visible / VISIBLE_PIXELS * PIXELS_PER_STEP / SLOW_MOTION
    tau = hole_pace(
        state.pos,
        state.vel,
        state.mass,
        G,
        state.black_hole,
        state.radius,
        state.debris,
        visible,
        unseen_radii(state, visible, drawn),
        travel,
        zone,
        FAST_ORBIT * sim_dt,
        CAPTURE_RADIUS,
        HANDOVER_RADIUS,
        LIGHT_BODY,
    )
    return min(sim_dt, max(SLOW_MOTION * tau, sim_dt / MAX_SLOW_MOTION))


def _number(value, what, positive=False):
    """`value` as a finite float, or ValueError naming `what`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} is not a number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{what} is not finite")
    if positive and value <= 0.0:
        raise ValueError(f"{what} must be positive")
    return value


def _vector(value, what):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{what} is not a list of three numbers")
    return [_number(v, what) for v in value]


def validate_scenario(data):
    """A saved N-body scenario, checked and with defaults filled in.

    Raises ValueError with a message for the user if `data` is not a usable
    scenario (a file of another kind, missing or malformed values). Each
    body needs "pos", "vel" and "mass"; everything else has a default.
    Names are made unique (bodies are followed by name).
    """
    if not isinstance(data, dict):
        raise ValueError("not an N-body scenario (the file does not hold a dictionary)")
    bodies = data.get("bodies")
    if not isinstance(bodies, list):
        raise ValueError("not an N-body scenario (it has no list of bodies)")
    if len(bodies) > MAX_BODIES:
        raise ValueError(f"too many bodies ({len(bodies)}; at most {MAX_BODIES})")
    clean, names = [], set()
    for i, body in enumerate(bodies):
        what = f"body {i + 1}"
        if not isinstance(body, dict):
            raise ValueError(f"{what} is not a dictionary")
        for key in ("pos", "vel", "mass"):
            if key not in body:
                raise ValueError(f"{what} has no '{key}'")
        name = body.get("name")
        name = name.strip() if isinstance(name, str) else ""
        name = unique_name(name or f"Body {i + 1}", names)
        names.add(name)
        color = body.get("color")
        try:
            color = [min(max(c, 0.0), 1.0) for c in _vector(color, "colour")]
        except ValueError:
            color = [0.8, 0.8, 0.8]  # a look, not worth refusing the file for
        mass = _number(body["mass"], f"the mass of {what}")
        radius = _number(body.get("radius", 0.0), f"the radius of {what}")
        drawn = body.get("drawn_radius", 0.02)
        drawn = _number(0.02 if drawn is None else drawn, f"the drawn radius of {what}")
        if mass < 0.0 or radius < 0.0 or drawn < 0.0:
            raise ValueError(f"{what} has a negative mass or radius")
        flags = {}
        for key in ("black_hole", "debris", "gas"):
            flag = body.get(key, False)
            if flag is None:
                flag = False
            if not isinstance(flag, bool):
                raise ValueError(f"'{key}' of {what} is not true or false")
            flags[key] = flag
        clean.append(
            {
                "name": name,
                "color": color,
                "drawn_radius": drawn,
                "pos": _vector(body["pos"], f"the position of {what}"),
                "vel": _vector(body["vel"], f"the velocity of {what}"),
                "mass": mass,
                "radius": radius,
                **flags,
            }
        )
    out = {"bodies": clean}
    out["view_distance"] = _number(data.get("view_distance", 5.0), "view_distance", positive=True)
    out["view_pitch"] = _number(data.get("view_pitch", 0.0), "view_pitch")
    out["years_per_second"] = _number(
        data.get("years_per_second", 1.0), "years_per_second", positive=True
    )
    out["sheet"] = bool(data.get("sheet", False))
    info = data.get("info", "")
    out["info"] = info if isinstance(info, str) else ""
    gravity = data.get("gravity")
    out["gravity"] = gravity if gravity in (RELATIVITY, NEWTON) else None
    return out


def unique_name(name, taken):
    """`name`, or `name (2)`, `name (3)`, ... if it is among `taken`."""
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


def advance_step(integrator, state, dt, on_contact, rehold=None, on_meet=None):
    """One physics step of dt with substeps, as NBodyModule.step does it.

    `on_contact(pair)` is called when two bodies touch. It returns the state
    to continue with (a merger replaces the state), or None to end the step
    there. Returns False if the step was cut short by STEP_BUDGET (the
    state is then less than dt further, by integrator.last_time).

    The pairs held by the integrator (hold_pairs, called before this) move
    with their centres of mass during the substeps: both members are given
    its velocity for the leapfrog, and their own velocities relative to it
    are put back afterwards. Their own orbits are then advanced by the time
    integrated (nbody_binary.advance_pairs), at the end of the step and
    before every contact. A contact changes the list of bodies, so the
    pairs are let go for it and found again by `rehold(state)` afterwards.
    `on_meet(pair)` is called for a held pair whose orbit has decayed until
    its members meet; it returns like on_contact.
    """
    remaining, start, substeps = dt, time.perf_counter(), 0
    held = integrator.held
    pending = 0.0  # time integrated since the held pairs' own orbits were advanced
    own = None  # the held members' velocities relative to their pairs' centres of mass

    def carry(state):
        """Give the members of each held pair their centre of mass's velocity."""
        nonlocal own
        own = None
        if held is None or held.active(state.N) is None:
            return
        i, j = held.pairs[:, 0], held.pairs[:, 1]
        total = state.mass[i] + state.mass[j]
        centre = (state.mass[i, None] * state.vel[i] + state.mass[j, None] * state.vel[j]) / total[
            :, None
        ]
        own = (state.vel[i] - centre, state.vel[j] - centre)
        state.vel[i] = centre
        state.vel[j] = centre

    def settle(state):
        """Bring the held pairs up to date; returns the state, or None to end the step."""
        nonlocal pending, own
        if own is None:
            pending = 0.0
            return state
        state.vel[held.pairs[:, 0]] += own[0]
        state.vel[held.pairs[:, 1]] += own[1]
        own = None
        met = advance_pairs(state, held.pairs, held.G, held.ic2, pending, CAPTURE_RADIUS)
        pending = 0.0
        if len(met) and on_meet is not None:
            # One at a time: the meeting changes the list of bodies. Any other
            # pair that met is found touching by the next substep.
            held.pairs = NO_PAIRS
            integrator.invalidate()
            state = on_meet((int(met[0, 0]), int(met[0, 1])))
            if state is not None and rehold is not None:
                rehold(state)
        return state

    complete = True
    carry(state)
    while remaining > 1e-9 * dt:
        if time.perf_counter() - start > STEP_BUDGET:
            complete = False
            break
        if own is None and held is not None and len(held.pairs):
            carry(state)
        pair, t = integrator.advance(state, remaining, check_contact=True, max_substeps=32)
        substeps += integrator.last_substeps
        remaining -= t
        pending += t
        if pair is not None:
            bodies = state.N
            state = settle(state)
            if state is None:
                break
            if state.N != bodies:
                # A held pair met and the list of bodies changed under this
                # contact: the next substep finds it again.
                continue
            if held is not None and len(held.pairs):
                held.pairs = NO_PAIRS
                integrator.invalidate()
            state = on_contact(pair)
            if state is None:
                break
            if rehold is not None:
                rehold(state)
    if state is not None:
        settle(state)
    integrator.last_substeps = substeps
    integrator.last_time = dt - remaining  # time integrated, less than dt if cut short
    return complete


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
        # Drawn radii, for what counts as on screen (see unseen_radii).
        self.drawn = np.array(
            [*module.scene.radii, *([] if extra is None else [extra["radius"]])], dtype=float
        )
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
        sim_dt, visible = module.sim_dt, module.visible_scale

        def hold(state):
            return hold_pairs(self.integrator, state, sim_dt, visible)

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
                step_time(
                    self.state, sim_dt, visible, module.view_zone, self.drawn, hold(self.state)
                ),
                self._hit,
                hold,
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
        # Set by setup() when a saved scenario could not be opened; shown in
        # the editor window and the diagnostics.
        self.load_message = ""

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
        # One number per body that never changes and is never reused: what
        # the module's own bookkeeping refers to a body by (the hole a body
        # is falling into, the remnant of a collision). Names are for the
        # user, who can change them.
        self.ids = list(range(scene.state.N))
        self.next_id = scene.state.N
        # Positions and velocities Restart returns to.
        self.start_state = deepcopy(scene.state)
        self.integrator = make_integrator(self.gravity)
        # Set when bodies merged since the start state; the start state then
        # no longer matches the current list of names and looks.
        self.merged_since_start = False
        self.renderer = self._body_renderer()
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
        # Swallowed bodies shown frozen at their hole's edge, fading (see _fade).
        self.fading = []
        # Display: stars just torn, drawn over their gas until it has
        # stretched (see _ghost_sprites), and notes about events that were
        # too small to see: (hole id, text, start time).
        self.ghosts = []
        self.notes = []
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
            view_pitch=scene.view_pitch,
            sheet=scene.sheet,
            info=scene.info,
            years_per_second=scene.years_per_second,
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
        """Give body i another name. Names stay unique, because the camera
        follows a body by its name: a name already taken (also by a body
        that is on a close pass round a hole) gets a number, and an empty
        one is refused. Returns the name given."""
        old = self.scene.names[i]
        name = name.strip()
        if not name or name == old:
            return old
        taken = {n for k, n in enumerate(self.scene.names) if k != i} | set(self.away_names())
        name = unique_name(name, taken | {FOLLOW_NONE, FOLLOW_COM})
        self.scene.names[i] = name
        if editor_settings["follow"] == old:
            editor_settings["follow"] = name
            self.followed = name
        self.save_restart()
        return name

    def index_of(self, body_id):
        """Index of the body with this id, or None if it no longer exists."""
        try:
            return self.ids.index(body_id)
        except ValueError:
            return None

    def _new_ids(self, k):
        ids = list(range(self.next_id, self.next_id + k))
        self.next_id += k
        return ids

    def _hole_renderer(self):
        """The black holes alone, for drawing them over the sprites (_draw_bodies)."""
        holes = self.nb.black_hole
        return NBodyRenderer(
            np.array(self.scene.colors, dtype=float).reshape(-1, 3)[holes],
            np.array(self.scene.radii, dtype=float)[holes],
            holes=np.ones(int(holes.sum())),
            hole_min_radius_px=HOLE_MIN_PX,
        )

    def _body_renderer(self):
        nb = self.nb
        return NBodyRenderer(
            self.scene.colors,
            self.scene.radii,
            holes=nb.black_hole + 2 * nb.gas,
            hole_min_radius_px=HOLE_MIN_PX,
        )

    def min_radius_px(self, i):
        """Smallest radius body i is drawn with, in pixels."""
        return HOLE_MIN_PX if self.nb.black_hole[i] else self.renderer.min_radius_px

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
        taken = {*scene.names, *self.away_names()}
        while f"Body {k}" in taken:
            k += 1
        scene.names.append(f"Body {k}")
        self.ids.extend(self._new_ids(1))
        scene.colors.append(tuple(body["color"]))
        scene.radii.append(body["radius"])

        self.trails.add_bodies(1)
        self.renderer_dirty = True
        self.state_edited()

    def _drop_body(self, i, replacement=None, keep_follow=False, keep_trail=True):
        """Delete body i everywhere. `replacement` is the body that takes
        over its follow and selection (after a merge), or None. With
        keep_follow the camera goes on following its name (the body is on
        a close pass and may come back, see _close_passes). With keep_trail
        its trail stays behind and fades out (nbody_trails.py)."""
        scene = self.scene
        color, body_id = scene.colors[i], self.ids[i]
        if editor_settings["follow"] == scene.names[i] and not keep_follow:
            editor_settings["follow"] = (
                FOLLOW_NONE if replacement is None else scene.names[replacement]
            )

        self._resize_state([k for k in range(self.nb.N) if k != i])
        del scene.names[i]
        del scene.colors[i]
        del scene.radii[i]
        del self.ids[i]

        self.trails.remove_body(i, color if keep_trail else None, body_id)
        self.editor.on_body_removed(i, replacement)
        self.integrator.invalidate()
        self.edit_version += 1
        self.renderer_dirty = True

    def remove_body(self, i):
        """Remove body i from the running system."""
        self._drop_body(i, keep_trail=False)
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
        true_size = nb.black_hole[keep] and self._drawn_true_size(keep)
        merge_pair(nb, keep, gone)
        if true_size:
            self.scene.radii[keep] = float(nb.radius[keep])  # the horizon grew
        elif not absorbed:
            # Drawn size: same volume rule as the physical radius. A black
            # hole that absorbs a body keeps its drawn size.
            radii = self.scene.radii
            radii[keep] = float(np.cbrt(radii[keep] ** 3 + radii[gone] ** 3))

        self._drop_body(gone, replacement=keep)
        self.merged_since_start = True
        # The merger itself loses kinetic energy; measure the integration
        # error from here on.
        self.rebase_diagnostics()

    def _drawn_true_size(self, h):
        """Whether hole h is drawn with its horizon's own radius."""
        return bool(np.isclose(self.scene.radii[h], self.nb.radius[h], rtol=1e-6, atol=0.0))

    def _feed_hole(self, h, mass):
        """Hole h takes `mass`; its horizon, 2 G M / c^2, grows in proportion
        (and its drawn size, if it is drawn at its true size)."""
        nb = self.nb
        true_size = self._drawn_true_size(h)
        if nb.mass[h] > 0.0:
            nb.radius[h] *= (nb.mass[h] + mass) / nb.mass[h]
        nb.mass[h] += mass
        if true_size:
            self.scene.radii[h] = float(nb.radius[h])
            self.renderer_dirty = True

    def _spawn(self, pieces, names, color, drawn_radius, gas=False, ids=None):
        """Add debris (rows of pos, vel, mass, radius) made by the simulation itself."""
        old_n = self.nb.N
        k = len(names)
        self._resize_state([*range(old_n), *([None] * k)])
        nb = self.nb
        nb.pos[old_n:], nb.vel[old_n:], nb.mass[old_n:], nb.radius[old_n:] = pieces
        nb.debris[old_n:] = True
        nb.gas[old_n:] = gas
        self.scene.names.extend(names)
        self.ids.extend(self._new_ids(k) if ids is None else ids)
        self.scene.colors.extend([tuple(color)] * k)
        self.scene.radii.extend([drawn_radius] * k)
        self.trails.add_bodies(k)
        self.integrator.invalidate()
        self.edit_version += 1
        self.renderer_dirty = True
        self.merged_since_start = True
        self.rebase_diagnostics()

    def _debris_names(self, parent, k):
        used = {*self.scene.names, *self.away_names()}
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
        keep_id = self.ids[keep]
        self.merge(i, j)
        if kept >= MERGE_FRACTION or self.nb.N + N_FRAGMENTS > MAX_BODIES:
            return
        nb = self.nb
        keep = self.index_of(keep_id)
        keep_name = self.scene.names[keep]
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

    def disrupt(self, body, hole=None):
        """A black hole's tide pulls `body` apart: it becomes N_TIDAL pieces of debris.

        The pieces fill the body's volume and all have its velocity. Those
        that are already inside the handover sphere of `hole` (the body was
        torn there) are handed to their exact orbits at once.
        """
        nb = self.nb
        name, color, drawn = (
            self.scene.names[body],
            self.scene.colors[body],
            float(self._display_radii()[body]),  # the size it is drawn with now
        )
        pos, vel, mass, radius = (
            nb.pos[body].copy(),
            nb.vel[body].copy(),
            nb.mass[body],
            nb.radius[body],
        )
        hole_id = None if hole is None else self.ids[hole]
        offsets = tidal_pieces(radius, self.rng)
        self._drop_body(body)
        ids = self._new_ids(N_TIDAL)
        self._spawn(
            # The pieces' own radius only matters for being captured by a hole.
            (pos + offsets, vel, mass / N_TIDAL, 1e-3 * radius),
            self._debris_names(name, N_TIDAL),
            color,
            GAS_PIECE * drawn,
            gas=True,
            ids=ids,
        )
        # Display: the star stays drawn over its gas until that has stretched.
        self.ghosts.append(
            {"ids": set(ids), "color": tuple(color), "drawn": drawn, "radius": radius}
        )
        del self.ghosts[:-8]
        h = None if hole_id is None else self.index_of(hole_id)
        if h is None:
            return
        nb = self.nb
        distance = float(np.linalg.norm(pos - nb.pos[h]))
        if distance < unseen_radii(nb, self.visible_scale, self.scene.radii)[h]:
            self.notes.append(
                (
                    hole_id,
                    f"{name} was torn apart {distance:.2g} AU from the hole:"
                    " too close to see at this zoom",
                    time.perf_counter(),
                )
            )
            del self.notes[:-4]
        first = nb.N - N_TIDAL
        d = np.linalg.norm(nb.pos[first:] - nb.pos[h], axis=1)
        light = nb.mass[first:] <= LIGHT_BODY * nb.mass[h]
        reach = nb.radius[first:] + np.where(light, HANDOVER_RADIUS, CAPTURE_RADIUS) * nb.radius[h]
        self._capture_debris(h, first + np.flatnonzero(d < reach))

    def contact(self, pair, met=False):
        """Two bodies touch (nbodystate.first_contact): decide what happens.

        A black hole and a body: whole or torn is one rule wherever the body
        is. Inside its tidal radius r_t it is torn, unless r_t is inside the
        horizon (the hole is above the body's Hills mass), in which case it
        goes in whole (nbody_collisions.is_torn_before_horizon). A light
        body that reaches the handover sphere while still outside r_t is
        followed there whole and torn when it gets to r_t (nbody_plunge.py).
        A body too heavy for an exact orbit merges at the capture radius; if
        its r_t is outside the horizon it is torn there, since it would
        cross r_t on the way in.

        With `met` the two are a pair whose orbit was followed analytically
        (nbody_binary.py) and has decayed until they meet, at the capture
        radius or the tidal radius: a light body is then treated like a
        heavy one (torn, or taken by the hole at once). Its last orbits are
        below what can be seen, and an exact orbit started from there
        could bring it back out.
        """
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
        light = not met and nb.mass[body] <= LIGHT_BODY * nb.mass[hole]
        touching = (
            nb.radius[body] + (HANDOVER_RADIUS if light else CAPTURE_RADIUS) * nb.radius[hole]
        )
        distance = np.linalg.norm(nb.pos[body] - nb.pos[hole])
        tidal = tidal_radius(nb.radius[body], nb.mass[body], nb.mass[hole])
        tearable = (
            not nb.debris[body]
            and is_torn_before_horizon(
                nb.radius[body], nb.mass[body], nb.mass[hole], nb.radius[hole]
            )
            and nb.N + N_TIDAL <= MAX_BODIES
        )
        if tearable and (distance <= 1.05 * tidal or distance > 1.05 * touching or not light):
            self.disrupt(body, hole)
            return
        if nb.debris[body]:
            # With it, every other piece that has reached this hole by now:
            # a stream arrives as hundreds of pieces within a few steps.
            self._capture_debris(hole, self._debris_within_reach(hole, body))
        else:
            self._capture(hole, body, tidal if tearable else 0.0, follow=not met)

    def _debris_within_reach(self, hole, body):
        """`body` and all other debris within its reach of `hole` (pair_reach)."""
        nb = self.nb
        pieces = np.flatnonzero(nb.debris & ~nb.black_hole)
        d = np.linalg.norm(nb.pos[pieces] - nb.pos[hole], axis=1)
        light = nb.mass[pieces] <= LIGHT_BODY * nb.mass[hole]
        reach = (
            nb.radius[pieces] + np.where(light, HANDOVER_RADIUS, CAPTURE_RADIUS) * nb.radius[hole]
        )
        return np.union1d(pieces[d < reach], [body])

    def _capture_debris(self, hole, rows):
        """_capture for several pieces of debris at once (rows all after `hole`
        or before it; the hole's own index is found again by its id)."""
        hole_id = self.ids[hole]
        nb = self.nb
        followed, merged = [], []
        for k in rows:
            k = int(k)
            light = nb.mass[k] <= LIGHT_BODY * nb.mass[hole]
            rel = nb.pos[k] - nb.pos[hole]
            if light and self.close.enter(
                hole_id, nb.mass[hole], rel, nb.vel[k] - nb.vel[hole], self._meta(k)
            ):
                followed.append(k)
            else:
                merged.append(self.ids[k])
        if followed:
            self._drop_bodies(followed)
            self.merged_since_start = True
            self.rebase_diagnostics()
        for body_id in merged:
            self._capture(self.index_of(hole_id), self.index_of(body_id))

    def _meta(self, body, tidal=0.0):
        """What the exact-orbit code keeps of a body it takes over (nbody_plunge.py)."""
        nb = self.nb
        return {
            "id": self.ids[body],
            "name": self.scene.names[body],
            "color": self.scene.colors[body],
            "drawn": float(
                self.scene.radii[body] if nb.debris[body] else self._display_radii()[body]
            ),
            "mass": float(nb.mass[body]),
            "radius": float(nb.radius[body]),
            "debris": bool(nb.debris[body]),
            "gas": bool(nb.gas[body]),
            "tidal": float(tidal) if np.isfinite(tidal) else 0.0,
            # For giving a whole body back as the same body (see _return_body).
            "index": body,
            "selected": self.editor.selected == body,
            "orbit_source": self.editor.orbit_source == body,
        }

    def _capture(self, hole, body, tidal=0.0, follow=True):
        """Body `body` has reached hole `hole`: follow it on its exact orbit
        (nbody_plunge.py) if it is light enough, otherwise the hole takes it
        now. Either way its image ends frozen on the hole's edge, fading
        (see _fade). `tidal`: its tidal radius if it is to be torn there.
        Without `follow` the hole takes it now whatever its mass."""
        nb = self.nb
        hole_id = self.ids[hole]
        light = follow and nb.mass[body] <= LIGHT_BODY * nb.mass[hole]
        whole = not nb.debris[body]
        rel = nb.pos[body] - nb.pos[hole]
        meta = self._meta(body, tidal)
        if light and self.close.enter(
            hole_id, nb.mass[hole], rel, nb.vel[body] - nb.vel[hole], meta
        ):
            # Followed on its exact orbit round the hole until it is
            # swallowed, torn or leaves again.
            if whole:
                trails = self.trails
                meta["trail"] = (
                    trails.history[:, body].copy(),
                    int(trails.valid[body]),
                    self.step_index,
                )
            self._drop_body(body, keep_follow=whole)
            self.merged_since_start = True
            self.rebase_diagnostics()
            return
        # Not followed: too heavy for a geodesic, already at the horizon, or
        # with a velocity that has no meaning there.
        self._fade(hole_id, rel, self.scene.colors[body], meta["drawn"], bool(nb.gas[body]))
        self.merge(hole, body)

    # ------------------------------------------------- display of the events

    def _display_radii(self):
        """Drawn radius of every body as it is drawn now. Bodies are usually
        drawn much larger than they are (scene.radii), which near a black
        hole would hide what happens: a star drawn four times too large
        covers its own tidal radius round a small hole. So a body that is
        drawn larger than it is shrinks to its true size as it comes from
        TRUE_SIZE_FROM to TRUE_SIZE_AT tidal radii of a hole (where its
        tide matters), and is torn at the size it really has. Display only."""
        nb = self.nb
        radii = np.array(self.scene.radii, dtype=float)
        holes = np.flatnonzero(nb.black_hole)
        bodies = np.flatnonzero(
            ~nb.black_hole & ~nb.debris & (nb.radius > 0.0) & (radii > nb.radius)
        )
        if len(holes) == 0 or len(bodies) == 0:
            return radii
        keep = np.ones(len(bodies))  # 1: as large as the scene draws it, 0: true size
        for h in holes:
            d = np.linalg.norm(nb.pos[bodies] - nb.pos[h], axis=1)
            tidal = nb.radius[bodies] * np.cbrt(nb.mass[h] / np.maximum(nb.mass[bodies], 1e-300))
            near = (d / np.maximum(tidal, 1e-300) - TRUE_SIZE_AT) / (TRUE_SIZE_FROM - TRUE_SIZE_AT)
            keep = np.minimum(keep, np.clip(near, 0.0, 1.0))
        radii[bodies] = nb.radius[bodies] + (radii[bodies] - nb.radius[bodies]) * keep
        return radii

    def _hole_edge(self, h):
        """World radius hole h is drawn with (never below HOLE_MIN_PX pixels)."""
        return max(self.scene.radii[h], self.visible_scale * HOLE_MIN_PX / VISIBLE_PIXELS)

    def _drawn_size(self, drawn, kind=0.0):
        """World radius a body of drawn radius `drawn` is drawn with (the
        renderer's minimum size in pixels applies; gas has a smaller one)."""
        min_px = 1.0 if kind == 2.0 else self.renderer.min_radius_px
        return np.maximum(drawn, self.visible_scale * min_px / VISIBLE_PIXELS)

    def _fade(self, hole_id, offset, color, drawn, gas, look=1.0, distance=None):
        """Show a swallowed body frozen at the hole's edge, reddening and fading
        (see FADE_SECONDS). `offset`: where it was relative to the hole (with
        `distance`, only its direction is used; negative: resting on the
        edge); `look`: the redshift factor it was last drawn with (1 if it
        was never on an exact orbit), from which the fade goes on, so that
        its brightness does not jump."""
        offset = np.asarray(offset, dtype=float)
        norm = float(np.linalg.norm(offset))
        unit = offset / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0])
        now = time.perf_counter()
        self.fading.append(
            {
                "hole": hole_id,
                "unit": unit,
                "distance": norm if distance is None else distance,
                "color": tuple(color),
                "drawn": drawn,
                "kind": 2.0 if gas else 0.0,
                "look": look,
                "start": now,
            }
        )
        del self.fading[:-PLUNGE_CAPACITY]

    def _prune_fading(self, now):
        """Forget the fades that are over or whose hole is gone (every step and frame)."""
        if self.fading:
            ids = set(self.ids)
            self.fading = [
                f for f in self.fading if now - f["start"] < FADE_SECONDS and f["hole"] in ids
            ]

    def _fading_sprites(self):
        """(positions, colours, radii, kinds, look, alpha) of the fading bodies, or None."""
        now = time.perf_counter()
        self._prune_fading(now)
        if not self.fading:
            return None
        pos, colors, radii, kinds, looks, alphas = [], [], [], [], [], []
        for f in self.fading:
            h = self.index_of(f["hole"])
            t = (now - f["start"]) / FADE_SECONDS
            g = f["look"] * (1.0 - t)
            # Resting on the edge of the hole as it is drawn (as
            # nbody_plunge.display_radius leaves it); one swallowed
            # elsewhere is first brought there.
            edge = self._hole_edge(h)
            size = self._drawn_size(f["drawn"], f["kind"])
            rest = edge + FROZEN_OUTSIDE * size
            approach = min(t / FADE_APPROACH, 1.0)
            start = rest if f["distance"] < 0.0 else max(f["distance"], edge)
            distance = rest + (start - rest) * (1.0 - approach) ** 2
            pos.append(self.nb.pos[h] + f["unit"] * distance)
            colors.append(redshift_look(f["color"], g))
            radii.append(f["drawn"] * (0.6 + 0.4 * (1.0 - t)))
            kinds.append(f["kind"])
            looks.append(g)
            alphas.append(min(1.0, 4.0 * g))
        return (
            np.array(pos),
            np.array(colors),
            np.array(radii),
            np.array(kinds),
            np.array(looks),
            np.array(alphas),
        )

    def _ghost_sprites(self):
        """Display: a star that was just torn, drawn over its gas at the gas's
        centre, fading out as the gas stretches from the star's own size to
        twice that, while its gas fades in (`self.ghost_gas`: rows of the
        state and the opacity factor of the gas there). At the instant of
        the tear the picture is therefore the star as it was; from there it
        changes gradually into the gas, which by then is a strand."""
        self.ghost_gas = []
        if not self.ghosts:
            return None
        nb = self.nb
        index = {body_id: k for k, body_id in enumerate(self.ids)}
        pos, colors, radii, alphas, kept = [], [], [], [], []
        for ghost in self.ghosts:
            rows = [index[i] for i in ghost["ids"] if i in index]
            if len(rows) < N_TIDAL // 2 or ghost["radius"] <= 0.0:
                continue
            p = nb.pos[rows]
            centre = p.mean(axis=0)
            # Half-length of the cloud: a uniform ball of radius R has
            # variance R^2 / 5 along every axis.
            spread = np.sqrt(5.0 * np.linalg.eigvalsh(np.cov((p - centre).T))[-1])
            alpha = 2.0 - spread / ghost["radius"]
            if alpha <= 0.0:
                continue
            kept.append(ghost)
            self.ghost_gas.append((rows, 1.0 - min(alpha, 1.0)))
            pos.append(centre)
            colors.append(ghost["color"])
            radii.append(ghost["drawn"])
            alphas.append(min(alpha, 1.0))
        self.ghosts = kept
        if not pos:
            return None
        n = len(pos)
        return np.array(pos), np.array(colors), np.array(radii), np.zeros(n), np.ones(n), alphas

    def _draw_bodies(self, ctx, camera):
        """Draw the bodies, then the bodies on exact orbits round a hole,
        stars just torn and swallowed bodies fading, and last the black
        holes again, on top: whatever overlaps a hole's disc is hidden by
        it, in every one of these states alike."""
        nb, scene = self.nb, self.scene
        close = self._close_sprites()
        metas, close_pos, close_look = (
            ([], np.zeros((0, 3)), np.zeros(0)) if close is None else close
        )
        n_close = len(metas)
        close_radii = np.array([m["drawn"] for m in metas], dtype=float)
        close_gas = np.array([m["gas"] for m in metas], dtype=bool)
        close_axes = np.zeros((n_close, 3))
        close_alpha = np.ones(n_close)
        ghosts = self._ghost_sprites()

        # Gas, inside and outside the handover spheres together, so that a
        # stream is drawn as one across them.
        gas = np.flatnonzero(nb.gas)
        radii = self._display_radii()
        axes = np.zeros((nb.N, 3))
        alpha = np.ones(nb.N)
        if len(gas) + close_gas.sum() > 0:
            width, axis, opacity = gas_strands(
                np.concatenate([nb.pos[gas], close_pos[close_gas]]),
                np.concatenate([radii[gas], close_radii[close_gas]]),
            )
            k = len(gas)
            radii[gas], axes[gas], alpha[gas] = width[:k], axis[:k], opacity[:k]
            for rows, factor in self.ghost_gas:
                alpha[rows] *= factor
            close_radii[close_gas], close_axes[close_gas] = width[k:], axis[k:]
            close_alpha[close_gas] = opacity[k:]
        if nb.N and (len(gas) or nb.black_hole.any()):
            order = self.renderer.order
            kinds = (nb.black_hole + 2 * nb.gas).astype(float)
            self.renderer.set_style(
                np.array(scene.colors, dtype=float).reshape(-1, 3)[order],
                radii[order],
                kinds[order],
                axes[order],
                None,
                alpha[order],
            )
        self.renderer.render(ctx, camera, nb.pos, without_holes=True)

        parts = []
        if n_close:
            parts.append(
                (
                    close_pos,
                    redshift_look(np.array([m["color"] for m in metas], dtype=float), close_look),
                    close_radii,
                    np.where(close_gas, 2.0, 0.0),
                    close_axes,
                    close_look,
                    close_alpha,
                )
            )
        for sprites in (ghosts, self._fading_sprites()):
            if sprites is not None:
                pos, colors, radii, kinds, look, alpha = sprites
                parts.append((pos, colors, radii, kinds, np.zeros((len(pos), 3)), look, alpha))
        if parts:
            pos, colors, radii, kinds, axes, look, alpha = (
                np.concatenate([part[k] for part in parts])[:OVERLAY_CAPACITY] for k in range(7)
            )
            self.plunge_renderer.set_style(colors, radii, kinds, axes, look, alpha)
            self.plunge_renderer.render(ctx, camera, pos, count=len(pos))
        if not self.renderer_dirty:
            self.hole_renderer.render(ctx, camera, nb.pos[nb.black_hole])

    def _close_sprites(self):
        nb = self.nb
        holes = np.flatnonzero(nb.black_hole)
        return self.close.sprites(
            {self.ids[k]: nb.pos[k] for k in holes},
            self._hole_masses(),
            lambda metas: np.array(
                [self._drawn_size(m["drawn"], 2.0 if m["gas"] else 0.0) for m in metas]
            ),
        )

    # --------------------------------------------- debris that cannot be followed

    def _accrete_unresolved(self, sim_dt, advanced):
        """Count bound debris that cannot be followed as accreted by its hole.

        Bound debris comes back to its pericentre, where the stream runs
        into itself and is accreted (Rees 1988, see nbody_collisions.py).
        The pieces here do not interact, so without this they would go round
        the hole for ever on orbits of minutes to hours, far below a pixel,
        and following them held the whole scene almost still. From each
        piece's two-body orbit about each hole (energy, period P and
        pericentre q as in nbodystate.hole_pace), a piece bound to a hole
        is given to it, mass and momentum, when

        (a) it is inside the distance from the hole that cannot be seen on
            screen (`unseen_radii`), if that is outside the handover sphere
            (inside which pieces are on exact orbits anyway): it has come
            back to a pericentre that cannot be shown;
        (b) or its orbit takes less than UNRESOLVED_ORBIT of the step just
            taken (`advanced`) and does not bring it to the hole (q outside
            the handover sphere and outside (a)'s distance): it goes round
            several times per frame, which can be neither drawn nor
            afforded, and stands for a disc draining into the hole. Such a
            piece does not itself slow the scene down (hole_pace), so
            while an event is being watched in slow motion it is followed,
            and it goes when the scene is back at its own pace;
        (c) or, when the last step was at the scene's own pace (no event
            being watched) and still did not fit in STEP_BUDGET, it is among
            the pieces with the shortest dynamical time about the hole of
            those that are not being watched (bound, period below
            FAST_ORBIT steps, q not at the hole): the ones the step was
            waiting for.

        Each is shown fading on the hole's edge. Whole bodies are never
        treated this way, only debris; unbound debris is left alone.
        """
        nb = self.nb
        pieces = np.flatnonzero(nb.debris & ~nb.black_hole)
        holes = np.flatnonzero(nb.black_hole)
        if len(pieces) == 0 or len(holes) == 0:
            return
        unseen = unseen_radii(nb, self.visible_scale, self.scene.radii)
        gone, held = {}, []
        for h in holes:
            rel = nb.pos[pieces] - nb.pos[h]
            vel = nb.vel[pieces] - nb.vel[h]
            r = np.maximum(np.linalg.norm(rel, axis=1), 1e-300)
            v2 = np.sum(vel**2, axis=1)
            mu = G * (nb.mass[h] + nb.mass[pieces])
            orbit_energy = 0.5 * v2 - mu / r
            bound = orbit_energy < 0.0
            a = np.where(bound, -mu / np.where(bound, 2.0 * orbit_energy, -1.0), np.inf)
            period = 2.0 * np.pi * np.sqrt(a**3 / mu)
            h2 = np.sum(np.cross(rel, vel) ** 2, axis=1)
            ecc = np.sqrt(np.maximum(1.0 + 2.0 * orbit_energy * h2 / mu**2, 0.0))
            q = h2 / mu / (1.0 + ecc)
            light = nb.mass[pieces] <= LIGHT_BODY * nb.mass[h]
            reach = (
                nb.radius[pieces] + np.where(light, HANDOVER_RADIUS, CAPTURE_RADIUS) * nb.radius[h]
            )
            zone = unseen[h] if unseen[h] > HANDOVER_RADIUS * nb.radius[h] else 0.0
            inside = bound & (r < zone)
            arrives = q < np.maximum(reach, zone)
            fast = bound & ~arrives & (period < UNRESOLVED_ORBIT * advanced)
            for k in pieces[inside | fast]:
                gone.setdefault(int(k), int(h))
            if self.slowed and advanced > 0.5 * sim_dt:
                waiting = bound & ~arrives & ~inside & ~fast & (period < FAST_ORBIT * sim_dt)
                tau = np.minimum(np.sqrt(r**3 / mu), r / np.sqrt(np.maximum(v2, 1e-300)))
                held.extend(
                    (float(tau[k]), int(pieces[k]), int(h)) for k in np.flatnonzero(waiting)
                )
        if held:
            shortest = min(held)[0]
            for tau, k, h in held:
                if tau <= 2.0 * shortest:
                    gone.setdefault(k, h)
        if not gone:
            return
        for k, h in gone.items():
            total = nb.mass[h] + nb.mass[k]
            nb.vel[h] = (nb.mass[h] * nb.vel[h] + nb.mass[k] * nb.vel[k]) / total
            self._fade(
                self.ids[h],
                nb.pos[k] - nb.pos[h],
                self.scene.colors[k],
                self.scene.radii[k],
                bool(nb.gas[k]),
            )
            self._feed_hole(h, nb.mass[k])
        self._drop_bodies(sorted(gone))
        self.merged_since_start = True
        self.rebase_diagnostics()

    def _drop_bodies(self, rows):
        """Delete several pieces of debris at once (nothing takes over their
        selection). As _drop_body for each, but the arrays are rebuilt once:
        one at a time, 300 pieces took 0.2 s."""
        rows = sorted({int(i) for i in rows})
        if not rows:
            return
        scene = self.scene
        if editor_settings["follow"] in {scene.names[i] for i in rows}:
            editor_settings["follow"] = FOLLOW_NONE
        gone = set(rows)
        self.trails.remove_bodies(
            rows, [scene.colors[i] for i in rows], [self.ids[i] for i in rows]
        )
        self._resize_state([k for k in range(self.nb.N) if k not in gone])
        for i in reversed(rows):
            del scene.names[i]
            del scene.colors[i]
            del scene.radii[i]
            del self.ids[i]
            self.editor.on_body_removed(i)
        self.integrator.invalidate()
        self.edit_version += 1
        self.renderer_dirty = True

    def _close_pace(self, sim_dt):
        """Longest step while bodies are on their exact orbits round a hole whose
        handover sphere is large enough to see: CLOSE_STEP G M / c^3, so that
        their winding in is watched at its own pace (see SLOW_MOTION). Only
        while one of them was handed over recently (nbody_plunge.
        WATCH_DURATION): a body that stays on an orbit inside the sphere
        would otherwise keep the scene in slow motion for good."""
        pace = sim_dt
        masses = self._hole_masses()
        for hole_id in self.close.holes:
            if hole_id not in masses or not self.close.watched(hole_id):
                continue
            r_g = G * masses[hole_id] / C_AU_PER_YEAR**2
            if 2.0 * HANDOVER_RADIUS * r_g >= self.visible_scale > 0.0:
                pace = min(pace, CLOSE_STEP * r_g / C_AU_PER_YEAR)
        return max(pace, sim_dt / MAX_SLOW_MOTION)

    def _hole_masses(self):
        """Mass of each black hole, by body id. A hole edited to mass 0 counts as gone."""
        nb = self.nb
        return {self.ids[k]: float(nb.mass[k]) for k in np.flatnonzero(nb.black_hole & (nb.mass > 0.0))}

    def away(self):
        """The whole bodies (not debris) that are on exact orbits round a hole
        and so not in the state just now: (hole id, meta, relative position,
        relative velocity)."""
        return [b for b in self.close.bodies(self._hole_masses()) if not b[1]["debris"]]

    def away_names(self):
        if self.close.count == 0:
            return []
        return [meta["name"] for _, meta, _, _ in self.away()]

    def _close_passes(self):
        """Advance the bodies inside capture spheres; settle those that were
        swallowed, left, or reached their tidal radius."""
        if self.close.count == 0:
            return
        swallowed, left, torn = self.close.advance(self.last_step_time, self._hole_masses())
        for hole_id, meta, direction in swallowed:
            h = self.index_of(hole_id)
            if h is None:
                continue
            # From the look it had when it froze on the horizon.
            self._fade(
                hole_id,
                direction,
                meta["color"],
                meta["drawn"],
                meta["gas"],
                look=FREEZE_LOOK,
                distance=-1.0,  # already resting on the horizon
            )
            # The hole takes the mass; its horizon grows in proportion. (The
            # body's momentum relative to the hole is neglected: it is at
            # most LIGHT_BODY of the hole's mass.)
            self._feed_hole(h, meta["mass"])
            self.integrator.invalidate()
            if not meta["debris"] and editor_settings["follow"] == meta["name"]:
                editor_settings["follow"] = FOLLOW_NONE
        for back, tear in ((left, False), (torn, True)):
            for hole_id, meta, rel, vel in back:
                h = self.index_of(hole_id)
                if h is None or self.nb.N >= MAX_BODIES:
                    continue
                k = self._return_body(h, meta, rel, vel)
                if tear:
                    self.disrupt(k, self.index_of(hole_id))
        if swallowed or left or torn:
            self.rebase_diagnostics()

    def _return_body(self, h, meta, rel, vel):
        """Give a body that was on an exact orbit round hole h back to the
        system, with the state it has now. A whole body comes back as the
        body it was: same id, name, place in the list, selection and (if it
        was away for less than a step) trail. Returns its index."""
        pos, v = self.nb.pos[h] + rel, self.nb.vel[h] + vel
        name = unique_name(meta["name"], {*self.scene.names, *self.away_names()})
        self._spawn(
            (pos[None], v[None], meta["mass"], meta["radius"]),
            [name],
            meta["color"],
            meta["drawn"],
            gas=meta["gas"],
            ids=[meta["id"]],
        )
        self.nb.debris[-1] = meta["debris"]
        k = self.nb.N - 1
        if meta["debris"]:
            return k
        k = self._move_last_to(min(meta["index"], k))
        trails = self.trails
        history, valid, step = meta.get("trail", (None, 0, -1))
        if history is not None and step == self.step_index - 1 and valid < trails.capacity:
            # Away within one step: its trail goes on, with the sample of
            # this step (taken while it was away) filled in.
            trails.history[:, k] = history
            trails.history[(trails.head - 1) % trails.capacity, k] = pos
            trails.valid[k] = valid + 1
            trails.drop_left(meta["id"])  # not also the copy it left behind
        if meta.get("selected"):
            self.editor.selected = k
        if meta.get("orbit_source"):
            self.editor.orbit_source = k
        return k

    def _move_last_to(self, index):
        """Move the last body to position `index` in the list of bodies."""
        last = self.nb.N - 1
        if index >= last:
            return last
        order = [*range(index), last, *range(index, last)]
        self._resize_state(order)
        scene = self.scene
        for values in (scene.names, scene.colors, scene.radii, self.ids):
            values.insert(index, values.pop())
        trails = self.trails
        trails.history = np.ascontiguousarray(trails.history[:, order])
        trails.valid = trails.valid[order]
        self.editor.on_body_inserted(index)
        return index

    # ------------------------------------------------------- save and load

    def scenario(self):
        """What is on screen now, as a dictionary for scenario_files."""
        nb, scene = self.nb, self.scene
        bodies = [
            {
                "name": scene.names[i],
                "color": [float(c) for c in scene.colors[i]],
                "drawn_radius": float(scene.radii[i]),
                "pos": nb.pos[i].tolist(),
                "vel": nb.vel[i].tolist(),
                "mass": float(nb.mass[i]),
                "radius": float(nb.radius[i]),
                "black_hole": bool(nb.black_hole[i]),
                "debris": bool(nb.debris[i]),
                "gas": bool(nb.gas[i]),
            }
            for i in range(nb.N)
        ]
        # Bodies on exact orbits round a hole are not in the state: saved
        # where they are, in the module's coordinates. When the file is
        # loaded they are inside the handover sphere and handed over again.
        for hole_id, meta, rel, vel in self.close.bodies(self._hole_masses()):
            h = self.index_of(hole_id)
            bodies.append(
                {
                    "name": meta["name"],
                    "color": [float(c) for c in meta["color"]],
                    "drawn_radius": float(meta["drawn"]),
                    "pos": (nb.pos[h] + rel).tolist(),
                    "vel": (nb.vel[h] + vel).tolist(),
                    "mass": meta["mass"],
                    "radius": meta["radius"],
                    "black_hole": False,
                    "debris": meta["debris"],
                    "gas": meta["gas"],
                }
            )
        return {
            "gravity": self.gravity,
            "view_distance": float(self.camera.distance)
            if self.camera is not None
            else scene.view_distance,
            "view_pitch": float(self.camera.pitch) if self.camera is not None else scene.view_pitch,
            "sheet": scene.sheet,
            "years_per_second": scene.years_per_second,
            "info": scene.info,
            "bodies": bodies,
        }

    def prepare_scenario(self, data):
        """Make a saved scenario what the Custom scene loads next (see load_scene).

        Raises ValueError, with nothing changed, if `data` is not a usable
        scenario (see validate_scenario)."""
        data = validate_scenario(data)
        bodies = data["bodies"]
        state = NBodyState(len(bodies))
        for i, body in enumerate(bodies):
            state.pos[i], state.vel[i] = body["pos"], body["vel"]
            state.mass[i], state.radius[i] = body["mass"], body["radius"]
            state.black_hole[i] = body["black_hole"]
            state.debris[i] = body["debris"]
            state.gas[i] = body["gas"]
        save_restart_point(
            CUSTOM_SCENE,
            state,
            [body["name"] for body in bodies],
            [tuple(body["color"]) for body in bodies],
            [body["drawn_radius"] for body in bodies],
            data["view_distance"],
            view_pitch=data["view_pitch"],
            sheet=data["sheet"],
            info=data["info"],
            years_per_second=data["years_per_second"],
        )
        self.scene_name = CUSTOM_SCENE
        if data["gravity"] is not None:
            self.gravity = data["gravity"]

    def load_scenario(self, data):
        """Load a saved scenario now: it becomes the Custom scene, and the module is rebuilt."""
        self.prepare_scenario(data)
        self.reset_requested = True

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
        # A body on its exact orbit round a hole: the hole stands in for it
        # (it is within the hole's handover sphere).
        if self.close.count:
            for hole_id, meta, _, _ in self.away():
                if meta["name"] == f:
                    return self.index_of(hole_id)
        return None

    def _followed_point(self, target):
        """Where the camera's followed point is. As anchor_point, except that
        for a body on its exact orbit round a hole it is the body itself."""
        f = editor_settings["follow"]
        if target not in (None, "com") and self.scene.names[target] != f:
            close = self._close_sprites()
            if close is not None:
                for meta, pos in zip(close[0], close[1], strict=True):
                    if meta["name"] == f and not meta["debris"]:
                        return pos
        if target not in (None, "com"):
            # A member of a pair whose orbit cannot be seen: the pair's centre
            # of mass, so that the camera does not shake with an orbit that
            # is sampled at random phases.
            for i, j in self.held_pairs():
                if target in (i, j):
                    nb = self.nb
                    total = nb.mass[i] + nb.mass[j]
                    return (nb.mass[i] * nb.pos[i] + nb.mass[j] * nb.pos[j]) / total
        return self.anchor_point(self.nb.pos, self.nb.mass, target)

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
        pos = self._followed_point(target)
        if pos is None:
            self.followed = None
            return

        key = editor_settings["follow"]
        if key != self.followed:
            # Just started following: centre on it (the distance is left alone).
            camera.target[:] = pos
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
        # Bound pairs that cannot be seen are followed analytically (nbody_binary.py).
        pairs = self._hold(self.nb)
        advanced = min(
            step_time(self.nb, sim_dt, self.visible_scale, self.view_zone, self.scene.radii, pairs),
            self._close_pace(sim_dt),
        )
        self.slow_motion = sim_dt / advanced
        self.slowed = not advance_step(
            self.integrator, self.nb, advanced, self._merge_contact, self._hold, self._meet
        )
        # A step cut short by STEP_BUDGET counts for the time it covered, so
        # the clock and the bodies on exact orbits stay in step with the rest.
        covered = self.integrator.last_time if self.slowed else advanced
        self.elapsed += covered
        self.last_step_time = covered
        self.step_index += 1
        self.trails.record(self.nb.pos)
        self.trails.valid[self.nb.gas] = 0  # a cloud of gas is drawn without trails
        self._close_passes()
        self._accrete_unresolved(sim_dt, advanced)
        self._prune_fading(time.perf_counter())
        if self.rebase_pending:
            self._rebase()
        nb = self.nb
        acc = self.integrator.acc_cache
        if acc is not None and acc.shape == nb.pos.shape:
            pairs = self.integrator.held.active(nb.N)
            if pairs is not None:
                # The members of a held pair have their centre's acceleration
                # there: add their pull on each other.
                acc = acc + internal_acc(nb, pairs, G)
            self.sheet.record(quadrupole_ddot(nb.pos, nb.vel, acc, nb.mass), G, C_AU_PER_YEAR)

    def _hold(self, state):
        return hold_pairs(
            self.integrator, state, self.dt * self.scene.years_per_second, self.visible_scale
        )

    def held_pairs(self):
        """The pairs currently followed analytically, as rows of body indices."""
        pairs = self.integrator.held.active(self.nb.N)
        return NO_PAIRS if pairs is None else pairs

    def _merge_contact(self, pair):
        self.contact(pair)
        return self.nb

    def _meet(self, pair):
        """A held pair's orbit has decayed until its members meet (nbody_binary.py)."""
        self.contact(pair, met=True)
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
            + (
                f"\n Unresolved pairs followed analytically: {len(self.held_pairs())}"
                if len(self.held_pairs())
                else ""
            )
            + (f"\n {self.scene.info}" if self.scene.info else "")
            + (f"\n {self.load_message}" if self.load_message else "")
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
        if self.camera is not None:
            W, H = imgui.get_io().display_size
            draw_scale_bar(self.camera, W, H, AU_M, "AU")

    def setup(self, ctx):
        pending = scenario_files.take_request("nbody")  # opened from the home screen
        self.load_message = ""
        if pending is not None:
            # A file that cannot be used must not stop the app: the scene
            # the module would have shown anyway is loaded, with a message.
            try:
                self.prepare_scenario(pending)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                self.load_message = f"Could not open the saved scenario: {error}"
        if self.scene_name not in SCENES:
            self.scene_name = DEFAULT_SCENE
        self.load_scene(self.scene_name)

        self.renderer.setup(ctx, self.nb.N)
        self.sheet.setup(ctx)
        self.trails.setup(ctx)
        self.background.setup(ctx)
        # Same smallest size as the bodies, so that a body does not change
        # size when it is handed to its exact orbit round a hole.
        self.plunge_renderer = NBodyRenderer(
            np.zeros((OVERLAY_CAPACITY, 3)), np.zeros(OVERLAY_CAPACITY)
        )
        self.plunge_renderer.setup(ctx, OVERLAY_CAPACITY)
        self.hole_renderer = self._hole_renderer()
        self.hole_renderer.setup(ctx, int(self.nb.black_hole.sum()))

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
            self.renderer = self._body_renderer()
            self.renderer.setup(ctx, self.nb.N)
            self.hole_renderer.release()
            self.hole_renderer = self._hole_renderer()
            self.hole_renderer.setup(ctx, int(self.nb.black_hole.sum()))
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
        self._draw_bodies(ctx, camera)
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
        self.hole_renderer.release()
