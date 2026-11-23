"""Light bodies very close to a black hole of the N-body module: exact orbits.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22). Review
the physics.

The module's relativity is an expansion in v / c (1PN). Within a few
G M / c^2 of a hole it fails: it cannot say whether a body plunges or swings
round and leaves, and inside 4 G M / c^2 it even repels. A body much lighter
than the hole, on the other hand, follows a geodesic of the hole's
Schwarzschild metric there, which geodesics.py integrates exactly. So when
such a body comes within nbodystate.HANDOVER_RADIUS horizon radii of the hole
(20 G M / c^2 in the module's coordinates, where 1PN is still good to a
quarter of a percent) it is handed over:

- its state relative to the hole is converted (see `enter`) and it is
  followed on the exact geodesic, in the hole's rest frame, in the time of
  a distant observer;
- if the geodesic turns round and leaves the sphere, the body is given
  back to the N-body system with the state it leaves with;
- if it plunges, it winds inward and slows as it nears the horizon
  (r = 2 G M / c^2); when it gets there its mass and momentum are added to
  the hole. Seen from far away its light is redshifted by the factor
  dtau/dt = (1 - 2/r) / E on top of the Doppler shift; it is drawn dimmer
  and redder by that factor relative to where it entered (the Doppler part,
  which depends on the viewing direction, is left out).

Whether a body is swallowed is therefore decided by the exact criterion
(for a body falling from far away: angular momentum below 4 G M / c per unit
mass), not by the sphere's size.

Approximations: the hole is treated as unaccelerated and non-rotating while
the body is inside (it is there for ~100 G M / c^3); the body's own gravity
and that of other bodies are ignored there; the two coordinate systems are
matched to first order (1PN "harmonic" radius = Schwarzschild radius
- G M / c^2). A body heavier than nbodystate.LIGHT_BODY of the hole is not
handed over (a geodesic assumes a light body): it merges at the capture
radius as before.

A body that is not debris and whose tidal radius R (M / m)^(1/3) lies inside
the handover sphere but outside the horizon is still whole when it is handed
over. It is followed like any other until it comes inside that radius, and
is then given back to the module to be torn apart (`advance` reports it).

Display. The module's coordinates put the horizon at G M / c^2, but a hole is
drawn as a disc of its areal radius 2 G M / c^2. So that a body freezes on
the edge of the disc as drawn, and not underneath it, its drawn distance
from the hole is stretched smoothly from the module's radius at the handover
sphere to the areal radius at the horizon (`display_radius`). Its light is
shown redder and dimmer by the redshift factor above, but only down to
FREEZE_LOOK of the way: the real fade e-folds every ~4 G M / c^3, far too
fast to see, so the module finishes it in slow motion (see
nbodymodule.FADE_SECONDS). Both are display only.

When a simulated step is much longer than G M / c^3 (a stellar-mass hole in
a planetary scene: microseconds against days) the whole passage happens
within one step and nothing of it can be seen: the body is swallowed or
comes back out at once.
"""

import numpy as np

from physengine.modules.geodesics import BodySystem, energy

# Leaves when it is this far outside the handover sphere (20 G M / c^2 in the
# module's coordinates = 21 here), so that it is not handed straight back.
HANDOVER = 21.0
EXIT_RADIUS = 1.1 * HANDOVER
MAX_DURATION = 200.0  # most G M / c^3 followed in one step (the rest waits for the next)
CAPACITY = 400
# Display: how far the drawn look goes towards fully redshifted while the
# body is still on its orbit (1 = not at all, 0 = black at the horizon).
FREEZE_LOOK = 0.7
# Display: a body frozen on the horizon is drawn with its centre this many of
# its drawn radii outside the edge of the hole's disc, moved there gradually
# from Schwarzschild radius FROZEN_FROM inward.
FROZEN_OUTSIDE = 0.7
FROZEN_FROM = 6.0
# A body is watched in slow motion (nbodymodule._close_pace) for this many
# G M / c^3 after it is handed over: a plunge or a passage takes 100 to 300.
# One that stays longer is on a bound orbit inside the sphere, which would
# keep the whole scene in slow motion for good.
WATCH_DURATION = 1500.0
TEAR_STEP = 4.0  # G M / c^3 between checks of a whole body against its tidal radius


def redshift_look(color, g):
    """Display colour of a body whose light is redshifted by g (1: as emitted, 0: gone):
    blue and green go first, then the whole dims."""
    g = np.clip(np.asarray(g, dtype=float), 0.0, 1.0)[..., None]
    return np.asarray(color, dtype=float) * np.concatenate([np.ones_like(g), g, g * g], -1) * g


def display_radius(r, size=0.0):
    """Drawn distance from the hole [G M / c^2] of a body at Schwarzschild radius r
    (see the module docstring): r - 1 at the handover sphere, r at the horizon.
    With `size`, the body's drawn radius in the same units, it stops
    FROZEN_OUTSIDE of that short of the disc's edge, so that it is seen
    resting on the horizon and not half hidden by the disc."""
    out = r - np.clip((r - 2.0) / (HANDOVER - 2.0), 0.0, 1.0)
    return out + FROZEN_OUTSIDE * size * np.clip((FROZEN_FROM - r) / (FROZEN_FROM - 2.0), 0.0, 1.0)


class CloseBodies:
    """The bodies currently inside one hole's capture sphere."""

    def __init__(self):
        self.system = BodySystem(np.zeros((0, 3)), np.zeros((0, 3)))
        self.system.OUTER_STEP = 20.0  # no pulls between them: one drift per call
        self.meta = []  # per body: dict (id, name, colour, masses, ...)
        self.exit_radius = []
        self.g0 = []
        self.age = []  # G M / c^3 since each was handed over

    def add(self, x, w, meta):
        self.system.add(x, w)
        self.meta.append(meta)
        self.exit_radius.append(EXIT_RADIUS)
        self.g0.append((1.0 - 2.0 / np.linalg.norm(x)) / max(energy(x[None], w[None])[0], 1e-12))
        self.age.append(0.0)

    def remove(self, gone):
        keep = [k for k in range(len(self.meta)) if k not in gone]
        s = self.system
        s.x, s.w, s.mass = s.x[keep], s.w[keep], s.mass[keep]
        s.alive, s.captured_at = s.alive[keep], s.captured_at[keep]
        s._acc = np.zeros_like(s.x)
        self.meta = [self.meta[k] for k in keep]
        self.exit_radius = [self.exit_radius[k] for k in keep]
        self.g0 = [self.g0[k] for k in keep]
        self.age = [self.age[k] for k in keep]

    def look(self):
        """Redshift factor to draw each body with: 1 where it entered,
        FREEZE_LOOK at the horizon."""
        s = self.system
        r = np.linalg.norm(s.x, axis=1)
        g = (1.0 - 2.0 / r) / np.maximum(energy(s.x, s.w), 1e-12) / np.array(self.g0)
        return FREEZE_LOOK + (1.0 - FREEZE_LOOK) * np.clip(g, 0.0, 1.0)


def to_schwarzschild(rel, vel, r_g, c):
    """(x, w = dx/dtau) in units G = c = M = 1 from a position and velocity relative
    to the hole in the module's coordinates, or None if the velocity is not below light speed.

    Radius: r = r_harmonic + G M / c^2. Coordinate velocity: same radial
    part, same angular velocity. Then dt/dtau from the normalisation of the
    four-velocity, 1 = (dt/dtau)^2 [(1 - 2/r) - v_r^2 / (1 - 2/r) - v_t^2].
    """
    d = np.linalg.norm(rel)
    if not d > 0.0:
        return None
    n = rel / d
    r = d / r_g + 1.0
    v = np.asarray(vel) / c
    v_r = v @ n
    v_t = (v - v_r * n) * (r / (d / r_g))
    lapse = 1.0 - 2.0 / r
    if lapse <= 0.0:
        return None
    norm = lapse - v_r**2 / lapse - v_t @ v_t
    if norm <= 0.0:
        return None
    return r * n, (v_r * n + v_t) / np.sqrt(norm)


def from_schwarzschild(x, w, r_g, c):
    """Inverse of to_schwarzschild: position and velocity relative to the hole."""
    r = np.linalg.norm(x)
    n = x / r
    lapse = 1.0 - 2.0 / r
    dt_dtau = energy(x[None], w[None])[0] / lapse
    v = w / dt_dtau
    v_r = v @ n
    v_t = (v - v_r * n) * ((r - 1.0) / r)
    return (r - 1.0) * r_g * n, (v_r * n + v_t) * c


class ClosePasses:
    """All bodies on close passes. Holes are known by a key that never changes
    (the module's body id), not by their display name."""

    def __init__(self, G, c):
        self.G, self.c = G, c
        self.holes = {}  # hole key -> CloseBodies

    @property
    def count(self):
        return sum(len(h.meta) for h in self.holes.values())

    def enter(self, hole_key, hole_mass, rel, vel, meta):
        """Hand a body over. Returns False if it cannot be followed (then merge it).

        `meta` may hold "tidal": the body's tidal radius [length units of
        rel], if it is to be torn on coming inside it (see `advance`).
        """
        if self.count >= CAPACITY:
            return False
        state = to_schwarzschild(rel, vel, self.G * hole_mass / self.c**2, self.c)
        if state is None or np.linalg.norm(state[0]) <= 2.05:
            return False
        self.holes.setdefault(hole_key, CloseBodies()).add(state[0], state[1], meta)
        return True

    def advance(self, dt, hole_masses):
        """Follow all bodies for `dt` of simulated time. `hole_masses` maps the keys of
        the holes that still exist to their masses.

        Returns (swallowed, left, torn): lists of (hole key, meta, direction
        from the hole where it went in), of (hole key, meta, relative
        position, relative velocity) for bodies that left the sphere, and
        the same for whole bodies that came inside their tidal radius.
        """
        swallowed, left, torn = [], [], []
        for key in list(self.holes):
            close = self.holes[key]
            if key not in hole_masses:
                del self.holes[key]  # its hole is gone (removed, or merged into another)
                continue
            r_g = self.G * hole_masses[key] / self.c**2
            remaining = min(self.c * dt / r_g, MAX_DURATION)
            while remaining > 1e-9 and close.meta:
                tearable = any(m.get("tidal", 0.0) > 0.0 for m in close.meta)
                step = min(remaining, TEAR_STEP if tearable else 20.0)
                close.system.advance(step)
                remaining -= step
                close.age = [a + step for a in close.age]
                s = close.system
                r = np.linalg.norm(s.x, axis=1)
                out = (
                    s.alive & (r > np.array(close.exit_radius)) & (np.sum(s.x * s.w, axis=1) > 0.0)
                )
                gone = []
                for k in np.flatnonzero(~s.alive):
                    swallowed.append((key, close.meta[k], s.x[k] / r[k]))
                    gone.append(int(k))
                for k in np.flatnonzero(out):
                    rel, vel = from_schwarzschild(s.x[k], s.w[k], r_g, self.c)
                    left.append((key, close.meta[k], rel, vel))
                    gone.append(int(k))
                for k in np.flatnonzero(s.alive & ~out):
                    tidal = close.meta[k].get("tidal", 0.0)
                    if tidal > 0.0 and (r[k] - 1.0) * r_g < tidal:
                        rel, vel = from_schwarzschild(s.x[k], s.w[k], r_g, self.c)
                        torn.append((key, close.meta[k], rel, vel))
                        gone.append(int(k))
                if gone:
                    close.remove(gone)
            if not close.meta:
                del self.holes[key]
        return swallowed, left, torn

    def watched(self, hole_key):
        """Whether any body round this hole was handed over less than WATCH_DURATION ago."""
        close = self.holes.get(hole_key)
        return close is not None and any(a < WATCH_DURATION for a in close.age)

    def bodies(self, hole_masses):
        """(hole key, meta, relative position, relative velocity) of every body being
        followed, in the module's coordinates (for saving, and for the camera)."""
        out = []
        for key, close in self.holes.items():
            if key not in hole_masses:
                continue
            r_g = self.G * hole_masses[key] / self.c**2
            s = close.system
            for k, meta in enumerate(close.meta):
                rel, vel = from_schwarzschild(s.x[k], s.w[k], r_g, self.c)
                out.append((key, meta, rel, vel))
        return out

    def sprites(self, hole_pos, hole_masses, sizes=None):
        """(metas, positions, redshift factors) of the bodies being followed, as drawn
        (see display_radius and FREEZE_LOOK). `hole_pos` maps hole keys to positions;
        `sizes(metas)` gives the radii the bodies are drawn with (default: meta["drawn"])."""
        metas, out_pos, out_g = [], [], []
        for key, close in self.holes.items():
            if key not in hole_masses or not close.meta:
                continue
            r_g = self.G * hole_masses[key] / self.c**2
            s = close.system
            r = np.linalg.norm(s.x, axis=1)
            metas.extend(close.meta)
            size = (
                np.array([m["drawn"] for m in close.meta], dtype=float)
                if sizes is None
                else sizes(close.meta)
            )
            out_pos.append(hole_pos[key] + (display_radius(r, size / r_g) * r_g / r)[:, None] * s.x)
            out_g.append(close.look())
        if not metas:
            return None
        return metas, np.concatenate(out_pos), np.concatenate(out_g)
