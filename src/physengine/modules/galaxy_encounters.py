"""Two-galaxy encounters: the approach orbit and the orientation of the discs.

Written by Claude under one-off overrides of CLAUDE.md (2026-11-20: the
parabolic encounters; later the same day: any eccentricity, Toomre angles,
editable scenes, the orbit in the galaxies' own potential). Review the
physics.

The galaxies' relative motion is set up as an orbit with the chosen
pericentre and eccentricity, each galaxy's centre falling in the field of the
other's mass inside their separation (galaxy_orbit.py; for point masses that
is the Kepler orbit, kepler_orbit below, kept for reference and tests).
Parabolic orbits (e = 1, just unbound at infinity) are the usual choice for
merger simulations, because galaxies falling together for the first time
come from far away with little energy (Toomre & Toomre 1972,
ApJ 178, 623; Barnes & Hernquist 1992, ARA&A 30, 705). Bound orbits (e < 1)
return sooner after the first passage and merge faster. Once the dark haloes
overlap the galaxies lose orbital energy to them (dynamical friction) and
merge. The haloes are extended (about 70% of a halo's mass lies within
100 kpc) and already overlap at the start, so the galaxies attract each other
less than point masses would. Set up on the point-mass orbit they passed at
18.7 kpc after 275 Myr instead of the nominal 10 kpc after 239 Myr, which is
why the orbit is computed from the galaxies' mass profiles (galaxy_orbit.py).
Measured with the module's settings, pericentre set to 10 kpc: the bulges
pass at 8.6 kpc after 272 Myr (Prograde, merged after 1.3 Gyr), 8.3
(Inclined), 7.2 (Retrograde), 7.1 kpc after 284 Myr (Bound prograde, merged
after 0.5 Gyr), and 1.0 kpc for Head-on.

Disc orientations use the angles of Toomre & Toomre (1972, sec. II): for each
galaxy, take its disc as the reference plane and describe the companion's
orbit around it. The inclination i is the angle between the disc plane and
that orbit (0: the disc spins the same way as the orbit, prograde; 180:
retrograde). The argument of pericentre omega is the angle, measured in the
orbit plane in the direction of motion, from the ascending node (where the
companion crosses the disc plane going to the side the disc's spin points to)
to the pericentre.

Model units: G = 1, kpc, 1e10 M_sun, 4.715 Myr.
"""

from dataclasses import dataclass, field

import numpy as np

from physengine.modules.galaxy_components import GalaxyModel
from physengine.modules.galaxy_orbit import encounter_orbit

# ------------------------------------------------------------------ the orbit


def apocentre(r_p, e):
    """Largest distance on the orbit: r_p (1 + e) / (1 - e); infinite for e >= 1."""
    return np.inf if e >= 1.0 else r_p * (1.0 + e) / (1.0 - e)


def kepler_orbit(M, r_p, e, r0, G=1.0):
    """Relative position and velocity on an incoming Kepler orbit.

    Body 2 relative to body 1, total mass M, pericentre distance r_p,
    eccentricity e, current distance r0 (r_p <= r0 <= apocentre). The orbit
    lies in the xy-plane with angular momentum along +z, pericentre on the +x
    axis. With semi-latus rectum p = r_p (1 + e), r = p / (1 + e cos f) with
    true anomaly f, negative before pericentre. Velocity: radial
    sqrt(G M / p) e sin f, transverse sqrt(G M / p) (1 + e cos f) (Murray &
    Dermott 1999, eqs. 2.20, 2.31-2.32).

    r_p = 0 is the radial parabolic orbit (only with e = 1): body 2 starts on
    the -x axis and falls straight in with the escape speed. That is the limit
    of the orbit above as r_p -> 0, so the disc angles mean the same thing on
    both sides of it.
    """
    if r_p < 0.0 or e < 0.0:
        raise ValueError("pericentre and eccentricity must be >= 0")
    if r_p == 0.0:
        if e != 1.0:
            raise ValueError("a radial orbit (pericentre 0) needs eccentricity 1")
        return np.array([-r0, 0.0, 0.0]), np.array([np.sqrt(2.0 * G * M / r0), 0.0, 0.0])

    tol = 1e-9
    r_a = apocentre(r_p, e)
    if r0 < r_p * (1.0 - tol) or r0 > r_a * (1.0 + tol):
        raise ValueError(f"start distance {r0} is outside the orbit's range [{r_p}, {r_a}]")

    p = r_p * (1.0 + e)
    cos_f = 1.0 if e == 0.0 else np.clip((p / r0 - 1.0) / e, -1.0, 1.0)
    f = -np.arccos(cos_f)
    c, s = np.cos(f), np.sin(f)
    k = np.sqrt(G * M / p)
    pos = r0 * np.array([c, s, 0.0])
    v_r = k * e * s
    v_t = k * (1.0 + e * c)
    vel = np.array([v_r * c - v_t * s, v_r * s + v_t * c, 0.0])
    return pos, vel


def time_to_pericentre(M, r_p, r0, G=1.0):
    """Time from distance r0 to pericentre on the parabola (Barker's equation).

    t = sqrt(2 r_p^3 / (G M)) (D + D^3 / 3), D = tan(|f| / 2).
    """
    p = 2.0 * r_p
    f = np.arccos(np.clip(p / r0 - 1.0, -1.0, 1.0))
    D = np.tan(f / 2.0)
    return np.sqrt(2.0 * r_p**3 / (G * M)) * (D + D**3 / 3.0)


# ----------------------------------------------------------- disc orientation


def rotation_x(angle):
    """Rotation by `angle` (radians) about the x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_z(angle):
    """Rotation by `angle` (radians) about the z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def disc_rotation(inclination, argument):
    """Rotation from a galaxy's own frame to the frame of its companion's orbit.

    Own frame: disc in the xy-plane, spinning about +z. Orbit frame: the
    companion's orbit around this galaxy in the xy-plane, angular momentum
    along +z, pericentre on +x. Angles in degrees (Toomre & Toomre 1972).

    With the disc as reference plane and the node line on its x-axis, the
    orbit frame's axes are R_x(i) R_z(omega) applied to the own frame's axes
    (the usual orbital-element rotation, longitude of the node 0, which the
    disc's symmetry about its axis makes arbitrary). This function returns the
    inverse, R_z(-omega) R_x(-i). The disc's spin axis then comes out as
    (sin i sin omega, sin i cos omega, cos i) in the orbit frame.
    """
    return rotation_z(-np.radians(argument)) @ rotation_x(-np.radians(inclination))


# Orbit frame of each galaxy in the simulation frame. Galaxy 1 sees the
# companion's orbit exactly as kepler_orbit returns it. Galaxy 2 sees galaxy 1
# at minus the relative position and velocity: the same orbital angular
# momentum (+z), pericentre on -x.
ORBIT_FRAMES = (np.eye(3), rotation_z(np.pi))


# -------------------------------------------------------------------- scenes


@dataclass
class DiscAngles:
    """Toomre & Toomre angles of one disc, in degrees (see the module docstring)."""

    inclination: float = 0.0
    argument: float = 0.0


@dataclass
class Encounter:
    pericentre: float = 10.0  # [kpc]; 0 means head-on
    eccentricity: float = 1.0
    start_distance: float = 100.0  # [kpc]
    discs: list[DiscAngles] = field(default_factory=lambda: [DiscAngles(), DiscAngles()])


@dataclass
class GalaxyScene:
    """What load() builds: one galaxy, or two on an encounter orbit."""

    galaxies: list[GalaxyModel]
    encounter: Encounter | None = None


def _pair(**encounter):
    return GalaxyScene(galaxies=[GalaxyModel(), GalaxyModel()], encounter=Encounter(**encounter))


SINGLE = "Single galaxy"

# The parabolic encounters start 100 kpc apart; the off-centre ones pass about
# 10 kpc apart at closest approach (a few disc scale lengths: close enough to raise strong
# tides, far enough that the discs do not collide head-on on the first pass).
SCENES = {
    # Both discs in the orbital plane, both prograde: the longest tails.
    "Prograde": _pair(),
    # Second disc tilted 60 degrees out of the orbital plane: one long tail,
    # one shorter and fanned out, as in the Antennae (NGC 4038/9).
    "Inclined": _pair(discs=[DiscAngles(), DiscAngles(60.0, 0.0)]),
    # Second disc flipped over: it spins against the orbit, and the brief,
    # misaligned tidal pull barely raises a tail on it.
    "Retrograde": _pair(discs=[DiscAngles(), DiscAngles(180.0, 0.0)]),
    # Straight at each other along y (see encounter_orbit), spin axes along
    # the collision axis: expanding ring-like ripples.
    "Head-on": _pair(pericentre=0.0, discs=[DiscAngles(90.0, 0.0), DiscAngles(90.0, 0.0)]),
    # Bound orbit (apocentre 90 kpc): falls back soon after the first passage
    # and merges after about 0.5 Gyr instead of 1.3. Starting at 70 kpc,
    # inbound, it reaches pericentre after about 270 Myr, like the parabolic
    # scenes (260 Myr).
    "Bound prograde": _pair(eccentricity=0.8, start_distance=70.0),
    SINGLE: GalaxyScene(galaxies=[GalaxyModel()]),
}


def place_pair(galaxy1, galaxy2, encounter, pair):
    """Put two galaxies on the encounter orbit, in the centre-of-mass frame.

    `pair` is their interaction (galaxy_orbit.GalaxyPair of the two models,
    or PointMasses), which fixes the orbit.

    Each galaxy is (pos, vel, mass, kind) centred and at rest in its own frame
    (disc in the xy-plane). Its disc is turned by its Toomre angles, then it
    is moved to its point on the orbit. Returns (pos, vel, mass, kind,
    galaxy_index).
    """
    m1, m2 = galaxy1[2].sum(), galaxy2[2].sum()
    M = m1 + m2
    enc = encounter
    rel_pos, rel_vel, _ = encounter_orbit(
        pair, enc.pericentre, enc.eccentricity, enc.start_distance
    )

    shifts = [(-m2 / M * rel_pos, -m2 / M * rel_vel), (m1 / M * rel_pos, m1 / M * rel_vel)]
    pos, vel, mass, kind, index = [], [], [], [], []
    for g, ((p, v, m, k), frame, angles, (dp, dv)) in enumerate(
        zip((galaxy1, galaxy2), ORBIT_FRAMES, enc.discs, shifts, strict=True)
    ):
        rot = frame @ disc_rotation(angles.inclination, angles.argument)
        pos.append(p @ rot.T + dp)
        vel.append(v @ rot.T + dv)
        mass.append(m)
        kind.append(k)
        index.append(np.full(len(m), g))
    return tuple(map(np.concatenate, (pos, vel, mass, kind, index)))
