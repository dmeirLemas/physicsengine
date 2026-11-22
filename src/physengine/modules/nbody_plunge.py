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

When a simulated step is much longer than G M / c^3 (a stellar-mass hole in
a planetary scene: microseconds against days) the whole passage happens
within one step and nothing of it can be seen: the body is swallowed or
comes back out at once.
"""

import numpy as np

from physengine.modules.geodesics import BodySystem, energy

# Leaves when it is this far outside the handover sphere (20 G M / c^2 in the
# module's coordinates = 21 here), so that it is not handed straight back.
EXIT_RADIUS = 1.1 * 21.0
MAX_DURATION = 200.0  # most G M / c^3 followed in one step (the rest waits for the next)
CAPACITY = 400


class CloseBodies:
    """The bodies currently inside one hole's capture sphere."""

    def __init__(self):
        self.system = BodySystem(np.zeros((0, 3)), np.zeros((0, 3)))
        self.system.OUTER_STEP = 20.0  # no pulls between them: one drift per call
        self.meta = []  # per body: dict (name, colour, masses, ...)
        self.exit_radius = []
        self.g0 = []

    def add(self, x, w, meta):
        self.system.add(x, w)
        self.meta.append(meta)
        self.exit_radius.append(EXIT_RADIUS)
        self.g0.append((1.0 - 2.0 / np.linalg.norm(x)) / max(energy(x[None], w[None])[0], 1e-12))

    def remove(self, gone):
        keep = [k for k in range(len(self.meta)) if k not in gone]
        s = self.system
        s.x, s.w, s.mass = s.x[keep], s.w[keep], s.mass[keep]
        s.alive, s.captured_at = s.alive[keep], s.captured_at[keep]
        s._acc = np.zeros_like(s.x)
        self.meta = [self.meta[k] for k in keep]
        self.exit_radius = [self.exit_radius[k] for k in keep]
        self.g0 = [self.g0[k] for k in keep]


def to_schwarzschild(rel, vel, r_g, c):
    """(x, w = dx/dtau) in units G = c = M = 1 from a position and velocity relative
    to the hole in the module's coordinates, or None if the velocity is not below light speed.

    Radius: r = r_harmonic + G M / c^2. Coordinate velocity: same radial
    part, same angular velocity. Then dt/dtau from the normalisation of the
    four-velocity, 1 = (dt/dtau)^2 [(1 - 2/r) - v_r^2 / (1 - 2/r) - v_t^2].
    """
    d = np.linalg.norm(rel)
    n = rel / d
    r = d / r_g + 1.0
    v = np.asarray(vel) / c
    v_r = v @ n
    v_t = (v - v_r * n) * (r / (d / r_g))
    lapse = 1.0 - 2.0 / r
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
    def __init__(self, G, c):
        self.G, self.c = G, c
        self.holes = {}  # hole name -> CloseBodies

    @property
    def count(self):
        return sum(len(h.meta) for h in self.holes.values())

    def enter(self, hole_name, hole_mass, rel, vel, meta):
        """Hand a body over. Returns False if it cannot be followed (then merge it)."""
        if self.count >= CAPACITY:
            return False
        state = to_schwarzschild(rel, vel, self.G * hole_mass / self.c**2, self.c)
        if state is None or np.linalg.norm(state[0]) <= 2.05:
            return False
        self.holes.setdefault(hole_name, CloseBodies()).add(state[0], state[1], meta)
        return True

    def advance(self, dt, hole_masses):
        """Follow all bodies for `dt` of simulated time. `hole_masses` maps the names of
        the holes that still exist to their masses.

        Returns (swallowed, left): lists of (hole name, meta, direction from the hole
        where it went in) and of (hole name, meta, relative position, relative velocity).
        """
        swallowed, left = [], []
        for name in list(self.holes):
            close = self.holes[name]
            if name not in hole_masses:
                del self.holes[name]  # its hole is gone (removed, or merged into another)
                continue
            r_g = self.G * hole_masses[name] / self.c**2
            remaining = min(self.c * dt / r_g, MAX_DURATION)
            while remaining > 1e-9 and close.meta:
                step = min(remaining, 20.0)
                close.system.advance(step)
                remaining -= step
                s = close.system
                r = np.linalg.norm(s.x, axis=1)
                out = (
                    s.alive & (r > np.array(close.exit_radius)) & (np.sum(s.x * s.w, axis=1) > 0.0)
                )
                gone = []
                for k in np.flatnonzero(~s.alive):
                    swallowed.append((name, close.meta[k], s.x[k] / r[k]))
                    gone.append(int(k))
                for k in np.flatnonzero(out):
                    rel, vel = from_schwarzschild(s.x[k], s.w[k], r_g, self.c)
                    left.append((name, close.meta[k], rel, vel))
                    gone.append(int(k))
                if gone:
                    close.remove(gone)
            if not close.meta:
                del self.holes[name]
        return swallowed, left

    def sprites(self, names, pos, hole_masses):
        """(positions, colours, radii, kinds) of the bodies being followed."""
        out_pos, out_col, out_rad, out_kind = [], [], [], []
        for name, close in self.holes.items():
            if name not in hole_masses:
                continue
            centre = pos[names.index(name)]
            r_g = self.G * hole_masses[name] / self.c**2
            s = close.system
            r = np.linalg.norm(s.x, axis=1)
            g = (1.0 - 2.0 / r) / np.maximum(energy(s.x, s.w), 1e-12) / np.array(close.g0)
            g = np.clip(g, 0.0, 1.0)
            for k, meta in enumerate(close.meta):
                out_pos.append(centre + (r[k] - 1.0) * r_g * s.x[k] / r[k])
                # Redshift: blue and green go first, and the whole dims.
                out_col.append(
                    np.array(meta["color"]) * np.array([1.0, g[k], g[k] ** 2]) * g[k] ** 2
                )
                out_rad.append(meta["drawn"])
                out_kind.append(2.0 if meta["gas"] else 0.0)
        if not out_pos:
            return None
        return np.array(out_pos), np.array(out_col), np.array(out_rad), np.array(out_kind)
