"""Forces when a system holds debris (nbodystate.DEBRIS): cost linear in the number of pieces.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22). Review
the physics.

Pieces of debris do not act on each other, so the forces split into
- the ordinary bodies among themselves (the module's usual force),
- the Newtonian pull of the debris on the ordinary bodies,
- the pull of the ordinary bodies on each piece. With relativity on, each
  ordinary body acts on a piece as on a test body, in that body's rest
  frame (postnewtonian.py, harmonic coordinates):
      a = -G M / r^2 [(1 - 4 G M / (r c^2) + v^2 / c^2) n - 4 (n.v) v / c^2],
  v the piece's velocity relative to the body. This drops the terms of the
  full 1PN equations that couple three bodies, and the pieces' own
  gravitational-wave reaction, both negligible for light debris.
"""

import numba
import numpy as np


@numba.njit(cache=True)
def pull_of_debris(pos_major, pos_debris, mass_debris, G):
    """Newtonian acceleration of each ordinary body due to all the debris."""
    out = np.zeros_like(pos_major)
    for i in range(len(pos_major)):
        for j in range(len(pos_debris)):
            d0 = pos_debris[j, 0] - pos_major[i, 0]
            d1 = pos_debris[j, 1] - pos_major[i, 1]
            d2 = pos_debris[j, 2] - pos_major[i, 2]
            r2 = d0 * d0 + d1 * d1 + d2 * d2
            if r2 == 0.0:
                continue
            f = G * mass_debris[j] / (r2 * np.sqrt(r2))
            out[i, 0] += f * d0
            out[i, 1] += f * d1
            out[i, 2] += f * d2
    return out


@numba.njit(cache=True)
def pull_on_debris(pos_debris, vel_debris, pos_major, vel_major, mass_major, G, c, relativistic):
    """Acceleration of each piece of debris due to the ordinary bodies (see the module docstring)."""
    out = np.zeros_like(pos_debris)
    inv_c2 = 1.0 / (c * c) if relativistic else 0.0
    for i in range(len(pos_debris)):
        for j in range(len(pos_major)):
            if mass_major[j] <= 0.0:
                continue
            n0 = pos_debris[i, 0] - pos_major[j, 0]
            n1 = pos_debris[i, 1] - pos_major[j, 1]
            n2 = pos_debris[i, 2] - pos_major[j, 2]
            r = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
            if r == 0.0:
                continue
            n0, n1, n2 = n0 / r, n1 / r, n2 / r
            v0 = vel_debris[i, 0] - vel_major[j, 0]
            v1 = vel_debris[i, 1] - vel_major[j, 1]
            v2 = vel_debris[i, 2] - vel_major[j, 2]
            gm = G * mass_major[j]
            radial = 1.0 + inv_c2 * (v0 * v0 + v1 * v1 + v2 * v2 - 4.0 * gm / r)
            along = -4.0 * inv_c2 * (n0 * v0 + n1 * v1 + n2 * v2)
            f = -gm / (r * r)
            out[i, 0] += f * (radial * n0 + along * v0)
            out[i, 1] += f * (radial * n1 + along * v1)
            out[i, 2] += f * (radial * n2 + along * v2)
    return out


def split_acc(pos, vel, mass, debris, major_acc, G, c, relativistic):
    """Accelerations of all bodies; `major_acc(pos, vel, mass)` is the force among the ordinary ones."""
    major = ~debris
    out = np.empty_like(pos)
    pm, vm, mm = pos[major], vel[major], mass[major]
    pd, vd = np.ascontiguousarray(pos[debris]), np.ascontiguousarray(vel[debris])
    out[major] = major_acc(pm, vm, mm) + pull_of_debris(pm, pd, mass[debris], G)
    out[debris] = pull_on_debris(pd, vd, pm, vm, mm, G, c, relativistic)
    return out
