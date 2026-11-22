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

The switch at r_t is sharp; in a real star the distortion grows smoothly as
it approaches r_t (and how deep it must go to be fully torn depends on its
structure, 0.9 to 2 r_t: Guillochon & Ramirez-Ruiz 2013, ApJ 767, 25).
"""

import numpy as np

from physengine.modules.blackhole_sph import plummer_energy
from physengine.modules.geodesics import BodySystem
from physengine.modules.kerr_bodies import (
    KerrBodySystem,
    local_velocity_from_schwarzschild,
    states_from_local,
)

INTACT, TORN, SWALLOWED = "intact", "torn apart", "swallowed whole"


class IntactStar:
    def __init__(self, info, bodies):
        self.info = info
        self.idx = np.arange(info.first, info.first + info.n)
        self.state = INTACT
        self.released_at = None  # coordinate time of the release
        if isinstance(bodies, KerrBodySystem):
            v = local_velocity_from_schwarzschild(info.x, info.w)
            st, c = states_from_local(info.x[None], v[None], bodies.a)
            self.centre = KerrBodySystem(st, c, bodies.a)
        else:
            self.centre = BodySystem(info.x[None], info.w[None])

    def update(self, bodies, duration):
        """Call after bodies.advance(duration): carries the particles, or releases them."""
        if self.state != INTACT:
            return
        c, idx, info = self.centre, self.idx, self.info
        c.advance(duration)
        if not c.alive[0]:
            bodies.alive[idx] = False
            bodies.captured_at[idx] = c.captured_at[0]
            self.state = SWALLOWED
            return
        # The particles were advanced on their own geodesics; put them back
        # on the star (their captured flags too: the star is still whole).
        bodies.alive[idx] = True
        bodies.captured_at[idx] = np.nan
        x = c.x[0] + info.dx
        if isinstance(bodies, KerrBodySystem):
            v = c.local_velocity(0) + info.dv
            bodies.st[idx], bodies.consts[idx] = states_from_local(x, v, bodies.a)
        else:
            bodies.x[idx] = x
            bodies.w[idx] = c.w[0] + info.dv
            bodies._acc_fresh = False
        if c.r()[0] < info.tidal_radius:
            bodies.mass[idx] = info.mass / info.n
            if bodies.hydro is not None:
                # The star's gas, in hydrostatic balance until now.
                bodies.hydro.resize(bodies.n)
                scale = info.radius / 2.5
                spacing = info.radius / np.cbrt(info.n)
                bodies.hydro.start(idx, plummer_energy(info.dx, info.mass, scale), spacing)
            self.state = TORN
            self.released_at = bodies.t
