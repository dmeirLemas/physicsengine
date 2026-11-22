"""Scenes for the black-hole module: which bodies orbit the hole at the start.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). The orbits come from geodesics.py; review the choices there.

Units G = c = M = 1. The disc lies in the xz-plane and spins about +y; an
orbit's inclination is measured from it (0: same plane and sense).
"""

from dataclasses import dataclass

import numpy as np

from physengine.modules.geodesics import (
    bound_orbit,
    circular_orbit,
    effective_potential,
    orbit_normal,
    orbit_state,
)


@dataclass
class BodySpec:
    """One body at the start of a scene."""

    name: str
    x: np.ndarray  # position [M]
    w: np.ndarray  # dx/dtau
    radius: float = 0.5  # [M], for its apparent size
    temperature: float = 6000.0  # [K]: colour and surface brightness
    mass: float = 0.0  # [M], for the pull on other bodies (0: a test body)


def orbiting(name, r_p, r_a, inclination_deg=0.0, node_deg=0.0, start="apocentre", **kw):
    """A body on the bound orbit between r_p and r_a, starting at its apocentre (or pericentre)."""
    E, L = bound_orbit(r_p, r_a)
    r = max(r_p, r_a) if start == "apocentre" else min(r_p, r_a)
    normal = orbit_normal(np.radians(inclination_deg), np.radians(node_deg))
    x, w = orbit_state(r, L, E, normal=normal, direction=(1.0, 0.0, 0.0))
    return BodySpec(name, x, w, **kw)


def circular(name, r, inclination_deg=0.0, nudge=0.0, **kw):
    """A body on the circular orbit at r, its angular momentum lowered by the fraction `nudge`."""
    E, L = circular_orbit(r)
    normal = orbit_normal(np.radians(inclination_deg))
    x, w = orbit_state(r, L * (1.0 - nudge), E, normal=normal, direction=(1.0, 0.0, 0.0))
    return BodySpec(name, x, w, **kw)


def infalling(name, r0, L, E=1.0, inclination_deg=0.0, **kw):
    """A body at r0 moving inward with energy E (1: at rest at infinity) and angular momentum L.

    A body from far away (E = 1) is captured if L < 4 M: the potential
    barrier V(r) never reaches E^2 = 1 (Chandrasekhar 1983, sec. 19).
    """
    normal = orbit_normal(np.radians(inclination_deg))
    x, w = orbit_state(r0, L, E, inward=True, normal=normal, direction=(1.0, 0.0, 0.0))
    return BodySpec(name, x, w, **kw)


def at_rest(name, r0, direction=(1.0, 0.0, 0.0), **kw):
    """A body let go from rest at r0: it falls straight in."""
    d = np.asarray(direction, dtype=float)
    return BodySpec(name, r0 * d / np.linalg.norm(d), np.zeros(3), **kw)


# ----------------------------------------------------------- a star of particles


def plummer_star(n, mass, scale, rng, r_max_factor=5.0):
    """Positions and velocities (Newtonian, centre-of-mass frame) of a Plummer sphere in equilibrium.

    Aarseth, Henon & Wielen (1974, A&A 37, 183): radii from the cumulative
    mass M(<r) / M = r^3 / (r^2 + a^2)^(3/2); speeds q v_esc with q drawn
    from g(q) = q^2 (1 - q^2)^(7/2) by rejection, v_esc = sqrt(2 G M) (r^2 + a^2)^(-1/4);
    directions isotropic. Cut at r_max_factor a.
    """
    r = np.empty(0)
    while len(r) < n:
        x1 = rng.uniform(1e-10, 1.0, 2 * n)
        cand = scale / np.sqrt(x1 ** (-2.0 / 3.0) - 1.0)
        r = np.concatenate([r, cand[cand < r_max_factor * scale]])
    r = r[:n]

    def directions(k):
        d = rng.normal(size=(k, 3))
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    q = np.empty(0)
    while len(q) < n:
        x4 = rng.uniform(0.0, 1.0, 4 * n)
        x5 = rng.uniform(0.0, 0.1, 4 * n)
        q = np.concatenate([q, x4[x5 < x4**2 * (1.0 - x4**2) ** 3.5]])
    q = q[:n]
    v_esc = np.sqrt(2.0 * mass) * (r**2 + scale**2) ** -0.25
    pos = r[:, None] * directions(n)
    vel = (q * v_esc)[:, None] * directions(n)
    return pos - pos.mean(axis=0), vel - vel.mean(axis=0)


def tidal_radius(star_radius, star_mass):
    """r_t = R (M / m)^(1/3): inside it the hole's tide beats the star's own gravity (Rees 1988)."""
    return star_radius * star_mass ** (-1.0 / 3.0)


def parabolic_pericentre_L(r_p):
    """L of the orbit with E = 1 (from rest at infinity) and pericentre r_p: L^2 = 2 r_p^2 / (r_p - 2)."""
    return np.sqrt(2.0 * r_p**2 / (r_p - 2.0))


# A star in physical units, and the hole's mass in solar masses, fix the
# star's mass and radius in the module's units (G = c = M = 1):
#     m = m_star / M,   R = R_star / (G M / c^2),
# with G M_sun / c^2 = 1.4766 km. Its tidal radius is then
#     r_t = R (M / m_star)^(1/3) = 47 M (M / 1e6 M_sun)^(-2/3)  for the Sun:
# it shrinks, in units of the hole's own size, as the hole grows. Above the
# Hills mass (Hills 1975, Nature 254, 295), ~1e8 solar masses for the Sun,
# r_t is inside the horizon and the star is swallowed whole.
GM_SUN_KM = 1.476625
STARS = {
    # name: mass [solar masses], radius [km], surface temperature [K]
    "Sun-like": (1.0, 6.957e5, 5800.0),
    "White dwarf": (0.6, 8.8e3, 12000.0),
}
# 3e5 solar masses: the Sun then has a radius of 1.5 M, large enough next to
# the hole (horizon radius 2 M) to see what happens to it.
DEFAULT_HOLE_MASS = 10.0**5.5  # solar masses
DEFAULT_STAR = "Sun-like"


def star_in_hole_units(star, hole_mass):
    """(mass, radius) of star `star` (a key of STARS) in units of a hole of `hole_mass` solar masses."""
    m, radius_km, _ = STARS[star]
    return m / hole_mass, radius_km / (GM_SUN_KM * hole_mass)


@dataclass
class StarInfo:
    """A star made of particles (see blackhole_star.IntactStar)."""

    first: int  # index of its first particle among the scene's bodies
    n: int
    dx: np.ndarray  # particle positions relative to the centre
    dv: np.ndarray  # and velocities (the star's internal motions)
    x: np.ndarray  # the centre's position
    w: np.ndarray  # and dx/dtau
    mass: float  # of the whole star [M]
    radius: float  # [M]
    tidal_radius: float  # [M]


def star_on_orbit(
    name,
    n,
    star_mass,
    star_radius,
    r0,
    L,
    inclination_deg=0.0,
    temperature=5800.0,
    seed=1,
    direction=(1.0, 0.0, 0.0),
    E=1.0,
):
    """A Plummer star of n particles whose centre is at r0, inbound with energy E (1: parabolic) and angular momentum L.

    Returns the particles (BodySpec, massless: the star holds together as
    one body until it is released, see blackhole_star.IntactStar) and a
    StarInfo. The star's own velocities (~sqrt(m/R), far below the orbital
    speed) are added to the centre's w = dx/dtau Newtonianly. Each particle
    has a radius R / sqrt(n), so that together they show the star's disc.
    """
    rng = np.random.default_rng(seed)
    scale = star_radius / 2.5  # Plummer scale: 80% of the mass inside star_radius
    dx, dv = plummer_star(n, star_mass, scale, rng)
    normal = orbit_normal(np.radians(inclination_deg))
    xc, wc = orbit_state(r0, L, E, inward=True, normal=normal, direction=direction)
    size = star_radius / np.sqrt(n)
    specs = [BodySpec(f"{name} {k}", xc + dx[k], wc + dv[k], size, temperature) for k in range(n)]
    info = StarInfo(
        0, n, dx, dv, xc, wc, star_mass, star_radius, tidal_radius(star_radius, star_mass)
    )
    return specs, info


@dataclass
class SceneOptions:
    """How a scene is run (not physics: step and storage sizes)."""

    outer_step: float = 1.0  # [M] between kicks of the bodies' mutual pull
    softening: float = 0.05  # [M]
    history: tuple = (0.5, 1600)  # (spacing [M], samples) of past states
    trails: bool = True
    info: str = ""
    gas: bool = False  # the torn star's debris is gas (blackhole_sph.py)
    # A scene whose bodies are gas from the start (the gas ring): thermal
    # energy per unit mass of each body, and their typical separation.
    gas_energy: np.ndarray | None = None
    gas_spacing: float = 0.0
    hide_disc: bool = False  # do not draw the painted accretion disc
    # Display: light of the bodies x this (a star of 2000 sub-pixel particles
    # is too faint to see next to the disc), and the camera at the start.
    brightness: float = 1.0
    view: tuple = (40.0, 0.25)  # (distance [M], pitch [rad])


STAR_PARTICLES = 2000
# "Star torn apart": a bound, very eccentric orbit (apocentre TDE_APOCENTRE)
# with L = 3.8, just under the capture limit of such an orbit (between 3.75
# and 3.9: measured, 90% of the star is swallowed on the first passage at
# L = 3.75 and 34% at 3.9). The star is drawn out into a strand that winds
# round the hole; most of it goes in, the rest swings out and comes back,
# because the orbit is bound. A star whose tidal radius is inside the
# capture radius goes in whole.
TDE_L = 3.8
TDE_APOCENTRE = 60.0  # [M]
TDE_START = 40.0  # [M] already on the way in
# "... with gas": an eccentric orbit (e = 0.6), so all the debris stays bound
# and comes back within a few hundred M (eccentric disruptions: Hayasaki,
# Stone & Loeb 2013, MNRAS 434, 909). The debris is gas (blackhole_sph.py):
# its streams collide and shock, and it settles around the hole.
GAS_PERICENTRE, GAS_APOCENTRE = 8.0, 32.0
STAR_SCENES = ("Star torn apart", "Star torn apart, with gas")


def star_scene(name, hole_mass=DEFAULT_HOLE_MASS, star=DEFAULT_STAR, plunge_L=TDE_L):
    """(particles, StarInfo) of star scene `name` for a hole of `hole_mass` solar masses.

    `plunge_L` is the angular momentum of the plunging orbit (a spinning
    hole's capture limit differs from 4).
    """
    m, radius = star_in_hole_units(star, hole_mass)
    common = dict(
        n=STAR_PARTICLES,
        star_mass=m,
        star_radius=radius,
        inclination_deg=25.0,
        # From the far right, clear of the settings panel on the left; the
        # star then swings round the far side and passes in front of the hole.
        direction=(1.0, 0.0, -0.6),
        temperature=STARS[star][2],
    )
    if name == "Star torn apart":
        E = np.sqrt(effective_potential(TDE_APOCENTRE, plunge_L))
        return star_on_orbit("Star", r0=TDE_START, L=plunge_L, E=E, **common)
    E, L = bound_orbit(GAS_PERICENTRE, GAS_APOCENTRE)
    return star_on_orbit("Star", r0=GAS_APOCENTRE, L=L, E=E, **common)


def _scenes():
    return {
        "Disc only": [],
        # Pericentre advance of 114 degrees per orbit (weak-field formula: 76).
        "Precessing orbit": [
            orbiting("Star", 10.0, 25.0, inclination_deg=25.0, temperature=9000.0)
        ],
        # Either side of the innermost stable circular orbit (r = 6): the
        # outer body stays, the inner one, nudged by 1e-4, spirals in.
        "Inside and outside the ISCO": [
            circular("r = 7", 7.0, inclination_deg=12.0, nudge=1e-4, temperature=8000.0),
            circular("r = 5.5", 5.5, inclination_deg=-12.0, nudge=1e-4, temperature=4500.0),
        ],
        # From far away (E = 1) with L just below 4: it whirls round the hole
        # a few times near r = 4, then plunges, slowing and fading as it
        # nears the horizon.
        "Plunging body": [infalling("Star", 30.0, 3.995, inclination_deg=20.0, temperature=9000.0)],
        # The star scenes depend on the masses: built by star_scene().
        "Star torn apart": [],
        "Star torn apart, with gas": [],
        # A ring of gas that slowly drains into the hole: built by gas_ring().
        "Gas ring": [],
        "Custom": [],
    }


SCENES = _scenes()


# "Gas ring": a ring of gas on circular orbits at RING_RADIUS, RING_WIDTH
# wide (standard deviation) and RING_HEIGHT thick, with the sound speed that
# holds that thickness up against the hole's vertical pull, c_s = H Omega
# (thin-disc balance), so u = c_s^2 / (gamma (gamma - 1)). Its mass is far
# too small for its own gravity to matter.
RING_PARTICLES = 2000
RING_RADIUS, RING_WIDTH, RING_HEIGHT = 12.0, 0.8, 0.35
RING_MASS = 1.0e-6
RING_ASPECT = RING_HEIGHT / RING_RADIUS  # H / r
RING_TEMPERATURE = 6500.0


def gas_ring(seed=3):
    """(particles, thermal energies, spacing) of the gas ring."""
    rng = np.random.default_rng(seed)
    n = RING_PARTICLES
    r = RING_RADIUS + RING_WIDTH * rng.normal(size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    height = RING_HEIGHT * rng.normal(size=n)
    axis = np.array([0.0, 1.0, 0.0])  # the disc's spin axis
    specs, energies = [], np.empty(n)
    for k in range(n):
        x = np.array([r[k] * np.cos(phi[k]), height[k], -r[k] * np.sin(phi[k])])
        dist = np.linalg.norm(x)
        E, L = circular_orbit(dist)
        # The circular orbit through x that is closest to the ring's plane.
        normal = axis - (axis @ x) * x / dist**2
        pos, w = orbit_state(dist, L, E, normal=normal, direction=x)
        specs.append(BodySpec(f"Gas {k}", pos, w, 0.03, RING_TEMPERATURE, RING_MASS / n))
        energies[k] = (RING_HEIGHT / dist) ** 2 / dist / (5.0 / 3.0 * 2.0 / 3.0)
    volume = 2.0 * np.pi * RING_RADIUS * (2.0 * RING_WIDTH) * (2.0 * RING_HEIGHT)
    return specs, energies, float(np.cbrt(volume / n))


def scene_options(name, hole_mass=DEFAULT_HOLE_MASS, star=DEFAULT_STAR):
    if name == "Gas ring":
        _, energies, spacing = gas_ring()
        return SceneOptions(
            outer_step=0.3,
            softening=0.1,
            history=(2.0, 250),
            trails=False,
            brightness=40.0,
            view=(48.0, 0.8),
            gas=True,
            gas_energy=energies,
            gas_spacing=spacing,
            hide_disc=True,
            info=f"Ring of gas at {RING_RADIUS:.0f} M: viscosity makes it spread and drain",
        )
    if name not in STAR_SCENES:
        return SceneOptions()
    m, radius = star_in_hole_units(star, hole_mass)
    r_t = tidal_radius(radius, m)
    t_dyn = np.sqrt((radius / 2.5) ** 3 / m)  # of the Plummer star
    common = dict(
        # 1% of the star's dynamical time, but not so short that the pulls
        # between 2000 particles take over the frame.
        outer_step=float(np.clip(0.01 * t_dyn, 0.2, 1.0)),
        softening=0.1 * radius / 2.5,
        history=(2.0, 250),
        trails=False,
        brightness=8.0,
    )
    if name == "Star torn apart":
        return SceneOptions(view=(55.0, 0.9), gas=True, info=f"Tidal radius {r_t:.3g} M", **common)
    return SceneOptions(
        view=(70.0, 0.9),
        gas=True,
        info=f"Tidal radius {r_t:.3g} M",
        **common,
    )
