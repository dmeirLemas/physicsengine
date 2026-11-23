"""A star that holds together until the hole's tide can pull it apart.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
3). Review the physics.

Outside its tidal radius r_t = R (M / m)^(1/3) a star is held together by
its own gravity and pressure, and moves as one body on the geodesic of its
centre. The particles that draw it (blackhole_scenes.star_on_orbit) are
carried along rigidly: their offsets from the centre and internal velocities
stay as they were made. When the centre comes inside r_t the tide wins: the
particles are released, each on its own geodesic, with the pulls between
them switched on (their share of the star's mass) and, in the scenes with
gas, their pressure (blackhole_sph.py). A star whose tidal radius
is inside the hole's capture radius is never released: it goes in whole.

What holds the released star up against its own gravity is either the random
motions of its particles (`info.dv`, a collisionless star) or, when the
bodies are gas, their pressure (thermal energy of hydrostatic balance,
blackhole_sph.plummer_energy) with the particles at rest in the star. Never
both: a star supported twice swells (measured: its half-mass radius doubled
in 6 dynamical times).

The switch at r_t is sharp; in a real star the distortion grows smoothly as
it approaches r_t (and how deep it must go to be fully torn depends on its
structure, 0.9 to 2 r_t: Guillochon & Ramirez-Ruiz 2013, ApJ 767, 25).
"""

import numpy as np

from physengine.modules.blackhole_sph import plummer_energy
from physengine.modules.geodesics import BodySystem, History
from physengine.modules.kerr_bodies import (
    KerrBodySystem,
    local_velocity_from_schwarzschild,
    states_from_local,
)

INTACT, TORN, SWALLOWED = "intact", "torn apart", "swallowed whole"
# The centre's position is checked against r_t at least every this fraction
# of r_t along its path, so the release happens at r_t and not a whole frame
# later (at a high Speed setting a frame is several M).
RELEASE_TOLERANCE = 0.01


class IntactStar:
    def __init__(self, info, bodies):
        self.info = info
        self.idx = np.arange(info.first, info.first + info.n)
        self.state = INTACT
        self.released_at = None  # coordinate time of the release
        self.released_r = None  # and the centre's radius then
        self.kerr = isinstance(bodies, KerrBodySystem)
        if self.kerr:
            v = local_velocity_from_schwarzschild(info.x, info.w)
            st, c = states_from_local(info.x[None], v[None], bodies.a)
            self.centre = KerrBodySystem(st, c, bodies.a)
        else:
            self.centre = BodySystem(info.x[None], info.w[None])
        self.centre.MAX_STEP = bodies.MAX_STEP
        self.centre.OUTER_STEP = np.inf  # one body: no mutual pulls to apply in between

    def _dv(self, bodies):
        """The particles' velocities in the star: none if pressure holds it up."""
        return 0.0 if bodies.hydro is not None else self.info.dv

    def _particles(self, bodies):
        """A massless system of the star's particles, carried by the centre as it now is."""
        c, info = self.centre, self.info
        x = c.x[0] + info.dx
        if self.kerr:
            v = c.local_velocity(0) + self._dv(bodies) + np.zeros_like(info.dx)
            st, consts = states_from_local(x, v, bodies.a)
            system = KerrBodySystem(st, consts, bodies.a)
        else:
            system = BodySystem(x, c.w[0] + self._dv(bodies) + np.zeros_like(info.dx))
        system.MAX_STEP = bodies.MAX_STEP
        system.OUTER_STEP = np.inf  # massless: no mutual pulls
        return system

    def _place(self, bodies, system, t_start):
        """Put the star's particles in `bodies` where `system` (started at t_start) has them."""
        idx = self.idx
        bodies.alive[idx] = system.alive
        bodies.captured_at[idx] = np.where(system.alive, np.nan, t_start + system.captured_at)
        if self.kerr:
            bodies.st[idx], bodies.consts[idx] = system.st, system.consts
        else:
            bodies.x[idx] = system.x
            bodies.w[idx] = system.w
            bodies._acc_fresh = False

    def fill_past(self, bodies, history):
        """The star's past in `history`: carried rigidly by its centre, as it is now.

        The History fills the time before the start by following each body
        backwards on its own geodesic; for the particles of a star that is
        still whole that is wrong (the star would be seen sheared).
        """
        centre_history = History(self.centre, history.spacing, history.length)
        self.centre.history = None
        dv = self._dv(bodies)
        x = centre_history.x[:, 0][:, None, :] + self.info.dx[None].astype(np.float32)
        w = centre_history.w[:, 0][:, None, :] + np.zeros_like(x)
        if not self.kerr:  # (Kerr: the history holds dx/dt; the internal motions are left out)
            w = w + np.asarray(dv, dtype=np.float32)
        history.x[:, self.idx] = x
        history.w[:, self.idx] = w

    def update(self, bodies, duration):
        """Call after bodies.advance(duration): carries the particles, or releases them."""
        if self.state != INTACT:
            return
        c, info = self.centre, self.info
        t_start = bodies.t - duration
        done = 0.0
        while duration - done > 1e-12 and c.alive[0]:
            # Far from r_t in one piece; near it in pieces along which the
            # centre moves RELEASE_TOLERANCE r_t at most.
            r = c.r()[0]
            speed = max(float(np.linalg.norm(c.dx_dt()[0])), 1e-6)
            reach = max(0.5 * (r - info.tidal_radius), RELEASE_TOLERANCE * info.tidal_radius)
            h = min(duration - done, reach / speed)
            c.advance(h)
            done += h
            if c.alive[0] and c.r()[0] < info.tidal_radius:
                break
        if not c.alive[0]:
            bodies.alive[self.idx] = False
            bodies.captured_at[self.idx] = t_start + c.captured_at[0]
            self.state = SWALLOWED
            return
        # The particles were advanced on their own geodesics; put them back
        # on the star (their captured flags too: the star is still whole).
        system = self._particles(bodies)
        if c.r()[0] >= info.tidal_radius:
            self._place(bodies, system, t_start + done)
            return
        # Released where the centre crossed r_t: from there each particle
        # follows its own geodesic for the rest of this call (its pulls and
        # pressure start with the next one; a frame is short against the
        # star's dynamical time).
        self.released_r = float(c.r()[0])
        self.released_at = t_start + done
        if duration - done > 1e-12:
            system.advance(duration - done)
        self._place(bodies, system, self.released_at)
        idx = self.idx
        bodies.mass[idx] = info.mass / info.n
        if bodies.hydro is not None:
            # The star's gas, in hydrostatic balance until now.
            bodies.hydro.resize(bodies.n)
            scale = info.radius / 2.5
            spacing = info.radius / np.cbrt(info.n)
            bodies.hydro.start(idx, plummer_energy(info.dx, info.mass, scale), spacing)
        self.state = TORN
