"""What happens when bodies of the N-body module touch: merging, shattering, tidal disruption.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22). Review
the physics: these are scaling laws and simple recipes, not simulations of
the impacts themselves.

Units: AU, years, solar masses.

Collisions of two ordinary bodies
---------------------------------
The outcome is set by the specific impact energy Q_R = mu v^2 / (2 M)
(mu the reduced mass, M the total mass, v the impact speed) against the
catastrophic-disruption energy Q*_RD, at which half the mass is dispersed.
For bodies held together by gravity (Leinhardt & Stewart 2012, ApJ 745, 79,
eqs. 5, 23, 28, 44):

    Q*_RD = C_STAR (4/5) pi RHO1 G R_C1^2 * ((1 + g)^2 / (4 g))^(2 / (3 MU_BAR) - 1),

with R_C1 the radius of the total mass at density RHO1 = 1000 kg/m^3, g the
mass ratio (<= 1), C_STAR = 1.9 and MU_BAR = 0.36 (their fit for planets).
The largest remnant has

    M_lr / M = 1 - Q_R / (2 Q*_RD)                     for Q_R < 1.8 Q*_RD,
             = 0.1 (Q_R / (1.8 Q*_RD))^(-1.5)          above (super-catastrophic).

Two Earths: Q*_RD = 2.85 (AU/yr)^2, i.e. half is dispersed at 4.8 AU/yr =
23 km/s, twice their mutual escape speed. The impact is taken as head-on
(a grazing impact disperses less: their "hit-and-run" regime is not
modelled). The dispersed mass is split into N_FRAGMENTS equal pieces, thrown
out in random directions from just outside the remnant, with speeds between
escape and well above it, limited so that the fragments carry at most
(1 - HEAT_SHARE) of the impact's kinetic energy; the rest is heat. Their total
momentum is zero in the centre-of-mass frame, so momentum is conserved.
Fragments that cannot escape fall back onto the remnant.

A black hole and a body
-----------------------
Inside r_t = R (M_hole / m)^(1/3) the hole's tide exceeds the body's own
gravity (Rees 1988, Nature 333, 523) and pulls it apart: the body is
replaced by N_TIDAL pieces spread through its volume, all with its velocity.
From there gravity alone stretches them into a stream ("spaghettification"),
roughly half of it bound to the hole. If r_t is inside the hole's capture
radius, the body goes in whole: for the Sun that needs a hole above ~1e8
solar masses (Hills 1975, Nature 254, 295); denser bodies need less.
"""

import numpy as np

RHO1 = 1.683e6  # 1000 kg/m^3 in solar masses per AU^3
C_STAR = 1.9
MU_BAR = 0.36
N_FRAGMENTS = 8
N_TIDAL = 300
HEAT_SHARE = 0.5  # of the impact's kinetic energy
# A merger that keeps more than this fraction is treated as a clean merger.
MERGE_FRACTION = 0.9


def disruption_energy(m1, m2, G):
    """Q*_RD of a collision of masses m1 and m2 [(AU/yr)^2]."""
    total = m1 + m2
    r_c1 = (3.0 * total / (4.0 * np.pi * RHO1)) ** (1.0 / 3.0)
    g = min(m1, m2) / max(m1, m2)
    ratio = ((1.0 + g) ** 2 / (4.0 * g)) ** (2.0 / (3.0 * MU_BAR) - 1.0)
    return C_STAR * 0.8 * np.pi * RHO1 * G * r_c1**2 * ratio


def largest_remnant_fraction(m1, m2, speed, G):
    """M_lr / (m1 + m2) after a head-on collision at relative speed `speed`."""
    if m1 <= 0.0 or m2 <= 0.0:
        return 1.0
    total = m1 + m2
    q = 0.5 * (m1 * m2 / total) * speed**2 / total
    x = q / disruption_energy(m1, m2, G)
    if x < 1.8:
        return 1.0 - 0.5 * x
    return 0.1 * (x / 1.8) ** -1.5


def fragments(remnant_mass, remnant_radius, debris_mass, impact_energy, G, rng, n=N_FRAGMENTS):
    """Positions and velocities of n equal fragments relative to the remnant.

    Launched from 2.5 (remnant radius + fragment radius), with speeds there
    of sqrt(v_esc^2 + v_inf^2), v_inf drawn up to twice the escape speed,
    then scaled down if together they would carry more than
    (1 - HEAT_SHARE) of `impact_energy`. Returns (offsets, velocities,
    fragment mass, fragment radius); the velocities' mean is zero.
    """
    m = debris_mass / n
    radius = remnant_radius * (m / remnant_mass) ** (1.0 / 3.0)  # same density
    start = 2.5 * (remnant_radius + radius)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    v_esc = np.sqrt(2.0 * G * (remnant_mass + debris_mass) / start)
    speed = v_esc * np.sqrt(1.0 + rng.uniform(0.0, 2.0, n) ** 2)
    vel = speed[:, None] * d
    vel -= vel.mean(axis=0)
    kinetic = 0.5 * m * np.sum(vel**2)
    budget = (1.0 - HEAT_SHARE) * impact_energy
    if kinetic > budget:
        vel *= np.sqrt(budget / kinetic)
    return start * d, vel, m, radius


def tidal_pieces(radius, rng, n=N_TIDAL):
    """Offsets of n pieces spread uniformly through a sphere of `radius`."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return radius * np.cbrt(rng.uniform(0.0, 1.0, n))[:, None] * d
