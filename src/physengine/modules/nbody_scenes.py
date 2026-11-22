"""Scenes for the N-body module: initial conditions plus how each body looks.

Units: AU, years, solar masses. Every builder takes G in those units.

All orbits here start circular and coplanar (in the xy-plane), with phases
chosen only to spread the planets out. They are not real ephemeris positions.

Data sources (NASA planetary fact sheets, nssdc.gsfc.nasa.gov/planetary/factsheet):
    masses in kg divided by the solar mass 1.98847e30 kg,
    semi-major axes in AU, Moon distance 384 400 km / 149 597 870.7 km.
"""

from dataclasses import dataclass

import numpy as np

from physengine.modules.nbodystate import NBodyState, sun_earth_scenario

# Colours (display only)
SUN = (1.0, 0.85, 0.4)
MERCURY = (0.65, 0.6, 0.55)
VENUS = (0.95, 0.8, 0.55)
EARTH = (0.3, 0.55, 1.0)
MOON = (0.8, 0.8, 0.8)
MARS = (0.9, 0.4, 0.25)

# Masses in solar masses
M_SUN = 1.0
M_MERCURY = 1.6601e-7
M_VENUS = 2.4478e-6
M_EARTH = 3.0035e-6
M_MOON = 3.6923e-8
M_MARS = 3.2272e-7

# Orbit radii in AU (semi-major axes)
A_MERCURY = 0.387098
A_VENUS = 0.723332
A_EARTH = 1.0
A_MARS = 1.523679
A_MOON = 0.0025696  # from Earth

# Physical (mean) radii in AU, for collisions: km / 149 597 870.7.
R_SUN = 0.0046505  # 695 700 km
R_MERCURY = 1.6308e-5  # 2 439.7 km
R_VENUS = 4.0454e-5  # 6 051.8 km
R_EARTH = 4.2587e-5  # 6 371.0 km
R_MOON = 1.1614e-5  # 1 737.4 km
R_MARS = 2.2657e-5  # 3 389.5 km
R_JUPITER = 4.6733e-4  # 69 911 km


def star_radius(mass):
    """Approximate main-sequence radius in AU: R = R_sun (M / M_sun)^0.8.

    A rough empirical fit, good to tens of percent between about 0.1 and
    1.5 solar masses (see e.g. Demircan & Kahraman 1991, Ap&SS 181, 313).
    """
    return R_SUN * mass**0.8


def rocky_radius(mass):
    """Radius in AU of a body of Earth's mean density: R = R_earth (m / m_earth)^(1/3)."""
    return R_EARTH * (mass / M_EARTH) ** (1.0 / 3.0)


@dataclass
class Scene:
    state: NBodyState
    names: list[str]
    colors: list[tuple[float, float, float]]
    # Drawn radius of each body in AU. Exaggerated so bodies are visible; the
    # true radii (Sun 0.00465 AU, Earth 0.0000426 AU) would be sub-pixel.
    radii: list[float]
    # Camera distance in AU when the scene starts.
    view_distance: float = 5.0
    # Camera pitch in radians when the scene starts (0: face-on to the xy-plane).
    view_pitch: float = 0.0
    # Show the spacetime sheet (nbody_sheet.py) when the sheet setting is "Auto".
    sheet: bool = False
    info: str = ""
    # Simulated years per second at time scale 1 (default: one year). A
    # scene whose action takes hours sets it small.
    years_per_second: float = 1.0


def make_state(pos, vel, mass, radius=None, centre_of_mass_frame=True, black_hole=None):
    """Pack arrays into an NBodyState, by default shifted to the centre-of-mass
    frame. `radius` is the physical radius per body (default 0, no collisions);
    `black_hole` flags the bodies that are black holes (default none)."""
    pos = np.array(pos, dtype=np.float64).reshape(-1, 3)
    vel = np.array(vel, dtype=np.float64).reshape(-1, 3)
    mass = np.array(mass, dtype=np.float64).reshape(-1)

    if centre_of_mass_frame and mass.sum() > 0.0:
        pos -= mass @ pos / mass.sum()
        vel -= mass @ vel / mass.sum()

    state = NBodyState(len(mass))
    state.pos[:, :] = pos
    state.vel[:, :] = vel
    state.mass[:] = mass
    if radius is not None:
        state.radius[:] = radius
    if black_hole is not None:
        state.black_hole[:] = black_hole
    return state


def _circular(G, m_central, m, r, phase):
    """Position and velocity, relative to the central body, of a body of mass m
    on a circular orbit of radius r: speed sqrt(G (M + m) / r), counter-clockwise."""
    v = np.sqrt(G * (m_central + m) / r)
    c, s = np.cos(phase), np.sin(phase)
    return np.array([r * c, r * s, 0.0]), np.array([-v * s, v * c, 0.0])


def sun_earth(G):
    # G is unused: the existing scenario function carries its own.
    state = sun_earth_scenario()
    state.radius[:] = [R_SUN, R_EARTH]
    return Scene(
        state=state,
        names=["Sun", "Earth"],
        colors=[SUN, EARTH],
        radii=[0.08, 0.032],
    )


def inner_solar_system(G):
    planets = [
        # name, mass, orbit radius, phase (rad), colour, drawn radius, physical radius
        ("Mercury", M_MERCURY, A_MERCURY, 0.0, MERCURY, 0.02, R_MERCURY),
        ("Venus", M_VENUS, A_VENUS, 2.0, VENUS, 0.03, R_VENUS),
        ("Earth", M_EARTH, A_EARTH, 4.0, EARTH, 0.032, R_EARTH),
        ("Mars", M_MARS, A_MARS, 5.5, MARS, 0.025, R_MARS),
    ]

    pos, vel, mass = [np.zeros(3)], [np.zeros(3)], [M_SUN]
    for _, m, a, phase, _, _, _ in planets:
        p, v = _circular(G, M_SUN, m, a, phase)
        pos.append(p)
        vel.append(v)
        mass.append(m)

    return Scene(
        state=make_state(pos, vel, mass, [R_SUN] + [p[6] for p in planets]),
        names=["Sun"] + [p[0] for p in planets],
        colors=[SUN] + [p[4] for p in planets],
        radii=[0.08] + [p[5] for p in planets],
    )


def sun_earth_moon(G):
    # The Earth-Moon barycentre goes round the Sun on a circle of 1 AU.
    bary_pos, bary_vel = _circular(G, M_SUN, M_EARTH + M_MOON, A_EARTH, 0.0)

    # The Moon goes round the Earth on a circle; split the relative position
    # and velocity by mass ratio so the pair's barycentre is where it should be.
    rel_pos, rel_vel = _circular(G, M_EARTH, M_MOON, A_MOON, 0.0)
    f_moon = M_EARTH / (M_EARTH + M_MOON)
    f_earth = -M_MOON / (M_EARTH + M_MOON)

    pos = [np.zeros(3), bary_pos + f_earth * rel_pos, bary_pos + f_moon * rel_pos]
    vel = [np.zeros(3), bary_vel + f_earth * rel_vel, bary_vel + f_moon * rel_vel]

    return Scene(
        state=make_state(pos, vel, [M_SUN, M_EARTH, M_MOON], [R_SUN, R_EARTH, R_MOON]),
        names=["Sun", "Earth", "Moon"],
        colors=[SUN, EARTH, MOON],
        # Earth and Moon must be drawn smaller than their 0.00257 AU separation.
        radii=[0.08, 0.0005, 0.00015],
    )


# ------------------------------------------------------------ star systems

STAR_A = (1.0, 0.85, 0.45)
STAR_B = (1.0, 0.55, 0.3)
STAR_C = (0.6, 0.75, 1.0)
PLANET = (0.4, 0.8, 0.7)
ASTEROID = (0.75, 0.7, 0.65)
JUPITER = (0.9, 0.75, 0.55)
M_JUPITER = 9.5479e-4  # solar masses (NASA fact sheet: 1.89813e27 kg)
A_JUPITER = 5.2026  # AU


def _kepler_pair(G, m1, m2, a, e):
    """Two bodies on a Kepler orbit with semi-major axis a and eccentricity e,
    starting at periapsis, in their centre-of-mass frame, orbiting
    counter-clockwise in the xy-plane.

    Periapsis distance a (1 - e); relative speed there from the vis-viva
    equation, v^2 = G (m1 + m2) (2 / r - 1 / a) = G (m1 + m2) (1 + e) / (a (1 - e)).
    """
    M = m1 + m2
    r = a * (1.0 - e)
    v = np.sqrt(G * M * (1.0 + e) / (a * (1.0 - e)))
    pos = [[-m2 / M * r, 0.0, 0.0], [m1 / M * r, 0.0, 0.0]]
    vel = [[0.0, -m2 / M * v, 0.0], [0.0, m1 / M * v, 0.0]]
    return np.array(pos), np.array(vel)


def binary_star(G):
    """Two stars on an eccentric orbit: 1.0 and 0.6 solar masses, a = 1 AU, e = 0.5."""
    m1, m2 = 1.0, 0.6
    pos, vel = _kepler_pair(G, m1, m2, a=1.0, e=0.5)
    return Scene(
        state=make_state(pos, vel, [m1, m2], [star_radius(m1), star_radius(m2)]),
        names=["Star A", "Star B"],
        colors=[STAR_A, STAR_B],
        radii=[0.06, 0.045],
        view_distance=4.0,
    )


def circumbinary_planet(G):
    """A planet circling both stars of a close binary (a 'Tatooine' orbit).

    Binary: 1.0 and 0.5 solar masses, a = 0.5 AU, e = 0.1. Planet: 1e-5 solar
    masses on a circular orbit of 2 AU around the binary's centre of mass,
    with the binary's total mass as the central mass. The stability limit for
    such orbits (Holman & Wiegert 1999, AJ 117, 621, eq. 3) is about 1.4 AU
    for these values, so 2 AU is stable.
    """
    m1, m2, m_p = 1.0, 0.5, 1.0e-5
    pos, vel = _kepler_pair(G, m1, m2, a=0.5, e=0.1)
    p_pos, p_vel = _circular(G, m1 + m2, m_p, 2.0, np.pi / 2)
    return Scene(
        state=make_state(
            [*pos, p_pos],
            [*vel, p_vel],
            [m1, m2, m_p],
            [star_radius(m1), star_radius(m2), rocky_radius(m_p)],
        ),
        names=["Star A", "Star B", "Planet"],
        colors=[STAR_A, STAR_B, PLANET],
        radii=[0.05, 0.04, 0.025],
        view_distance=6.0,
    )


def hierarchical_triple(G):
    """A close binary with a third star orbiting the pair from far out.

    Inner binary: two 1.0 solar-mass stars, a = 0.5 AU, circular. Outer star:
    0.5 solar masses on a circular orbit of 5 AU around the inner pair's
    centre of mass. The ratio of the orbits, 10, is far above the
    Mardling & Aarseth (2001, MNRAS 321, 398) stability limit of about 3
    for these masses, so the system is stable.
    """
    m1, m2, m3 = 1.0, 1.0, 0.5
    pos, vel = _kepler_pair(G, m1, m2, a=0.5, e=0.0)
    o_pos, o_vel = _circular(G, m1 + m2, m3, 5.0, np.pi / 2)
    return Scene(
        state=make_state(
            [*pos, o_pos],
            [*vel, o_vel],
            [m1, m2, m3],
            [star_radius(m1), star_radius(m2), star_radius(m3)],
        ),
        names=["Star A", "Star B", "Star C"],
        colors=[STAR_A, STAR_A, STAR_C],
        radii=[0.06, 0.06, 0.05],
        view_distance=14.0,
    )


def figure_eight(G):
    """Three equal masses chasing each other along one figure-eight curve.

    Initial conditions of Chenciner & Montgomery (2000, Annals of Mathematics
    152, 881), in the numbers of Simo, for G = 1 and unit masses:
        x1 = -x2 = (0.97000436, -0.24308753), x3 = 0,
        v3 = (-0.93240737, -0.86473146), v1 = v2 = -v3 / 2,
    period 6.32591398. Converted to 1 solar mass per body and a length unit
    of 1 AU: the time unit becomes sqrt(L^3 / (G M)) = 1 / (2 pi) yr, so
    velocities are multiplied by L / T = 2 pi AU/yr and the period is about
    1.007 yr. The orbit is linearly stable.
    """
    x1 = np.array([0.97000436, -0.24308753, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    v_scale = np.sqrt(G * 1.0 / 1.0)  # L / T for M = 1, L = 1 AU
    pos = [x1, -x1, np.zeros(3)]
    vel = [-v3 / 2 * v_scale, -v3 / 2 * v_scale, v3 * v_scale]
    return Scene(
        state=make_state(pos, vel, [1.0, 1.0, 1.0], [R_SUN] * 3),
        names=["Star A", "Star B", "Star C"],
        colors=[STAR_A, STAR_B, STAR_C],
        radii=[0.05, 0.05, 0.05],
        view_distance=3.5,
    )


def trojans(G):
    """Sun, Jupiter and two asteroids at Jupiter's L4 and L5 points.

    Lagrange's equilateral solution: every body sits at a corner of an
    equilateral triangle with the Sun and Jupiter and the whole configuration
    rotates rigidly about the centre of mass with Jupiter's angular speed
    w = sqrt(G (M_sun + M_J) / a^3). It is stable when one mass dominates
    (Routh's criterion, satisfied here by a factor of about 40).
    """
    m_ast = 1.0e-10
    masses = [M_SUN, M_JUPITER, m_ast, m_ast]
    a = A_JUPITER
    angles = [None, 0.0, np.pi / 3, -np.pi / 3]  # Jupiter, L4 (ahead), L5 (behind)
    pos = [np.zeros(3)] + [a * np.array([np.cos(t), np.sin(t), 0.0]) for t in angles[1:]]

    # Rigid rotation about the centre of mass: v = w z x (r - r_com).
    pos = np.array(pos)
    com = np.array(masses) @ pos / sum(masses)
    w = np.sqrt(G * (M_SUN + M_JUPITER) / a**3)
    vel = w * np.cross([0.0, 0.0, 1.0], pos - com)

    return Scene(
        state=make_state(
            pos, vel, masses, [R_SUN, R_JUPITER, rocky_radius(m_ast), rocky_radius(m_ast)]
        ),
        names=["Sun", "Jupiter", "Trojan (L4)", "Trojan (L5)"],
        colors=[SUN, JUPITER, ASTEROID, ASTEROID],
        radii=[0.3, 0.18, 0.1, 0.1],
        view_distance=16.0,
    )


# ------------------------------------------------------------ black holes
# Added by Claude under the override for the 1PN work (2026-11-20). These
# scenes are meant for the 1PN gravity setting. A black hole's physical
# radius is its horizon, 2 G M / c^2: bodies touching it are absorbed.

BLACK_HOLE = (0.35, 0.15, 0.5)  # display colour
M_SGR_A = 4.30e6  # Sagittarius A*, solar masses (GRAVITY Collaboration 2020, A&A 636, L5)
S2_A, S2_E = 1032.0, 0.8847  # S2: semi-major axis [AU] (0.1255" at 8.25 kpc) and eccentricity


def _black_hole_radius(G, m):
    from physengine.modules.postnewtonian import schwarzschild_radius

    return schwarzschild_radius(m, G)


def s2_sgr_a(G):
    """S2 on its 16-year orbit around Sagittarius A*, starting at pericentre (120 AU).

    General relativity advances its pericentre by
    6 pi G M / (c^2 a (1 - e^2)) = 12.1 arcminutes per orbit, measured by the
    GRAVITY Collaboration (2020): 1.10 +- 0.19 times the GR value.
    """
    m_s2 = 14.0
    pos, vel = _kepler_pair(G, M_SGR_A, m_s2, a=S2_A, e=S2_E)
    return Scene(
        state=make_state(
            pos,
            vel,
            [M_SGR_A, m_s2],
            [_black_hole_radius(G, M_SGR_A), 0.0],
            black_hole=[True, False],
        ),
        names=["Sgr A*", "S2"],
        colors=[BLACK_HOLE, (0.75, 0.85, 1.0)],
        radii=[25.0, 12.0],
        view_distance=4000.0,
    )


def star_near_black_hole(G):
    """A star 20 times closer to Sgr A* than S2: a = 50 AU, e = 0.67 (pericentre 16.5 AU).

    Pericentre advance 6 pi G M / (c^2 a (1 - e^2)) = 0.029 rad = 1.7 degrees
    per orbit of 0.17 years, so the orbit visibly turns. 1PN is still
    accurate here: the pericentre is 390 G M / c^2 from the hole. (The 1PN
    energy is conserved only up to 2PN terms, which on this orbit make it
    wobble by ~6e-4 of the binding energy over each orbit.)
    """
    m_star = 1.0
    pos, vel = _kepler_pair(G, M_SGR_A, m_star, a=50.0, e=0.67)
    return Scene(
        state=make_state(
            pos,
            vel,
            [M_SGR_A, m_star],
            [_black_hole_radius(G, M_SGR_A), R_SUN],
            black_hole=[True, False],
        ),
        names=["Black hole", "Star"],
        colors=[BLACK_HOLE, SUN],
        radii=[2.0, 1.0],
        view_distance=180.0,
    )


def black_hole_flyby(G):
    """A 5 solar-mass black hole passing by the inner solar system.

    It comes in from 30 AU at 50 AU/yr (237 km/s), aimed to pass 5 AU from
    the Sun. The impulse approximation gives the Sun a kick of
    2 G M / (b v) = 1.6 AU/yr, a quarter of the Earth's orbital speed: the
    planets' orbits are stretched and tilted, the outer ones most. (At 1 AU
    and 30 AU/yr a 10 solar-mass hole would fling the Sun away at 20 AU/yr.)
    The solar system starts at rest (this is not the centre-of-mass frame).
    """
    inner = inner_solar_system(G)
    m_bh = 5.0
    start = np.array([-30.0, 5.0, 0.5])
    vel_bh = np.array([50.0, 0.0, 0.0])
    st = inner.state
    pos = np.vstack([st.pos, start])
    vel = np.vstack([st.vel, vel_bh])
    mass = np.append(st.mass, m_bh)
    radius = np.append(st.radius, _black_hole_radius(G, m_bh))
    return Scene(
        state=make_state(
            pos, vel, mass, radius, centre_of_mass_frame=False, black_hole=[False] * st.N + [True]
        ),
        names=inner.names + ["Black hole"],
        colors=inner.colors + [BLACK_HOLE],
        radii=inner.radii + [0.06],
        view_distance=14.0,
    )


def black_hole_merger(G):
    """Two black holes of 5 billion solar masses each, spiralling together.

    They start 20 G M / c^2 apart (M the total mass: 1974 AU; each horizon
    has a radius of 99 AU) on the 1PN circular orbit. Gravitational
    waves (the 2.5PN force in postnewtonian.py) shrink the orbit: Peters'
    formula gives 19.5 years to merge; with the 1PN terms it takes 18.8
    years and 28 orbits (measured). The holes merge when they come within
    6 G M / c^2, and 4.8% of the mass leaves as waves.

    Post-Newtonian gravity is an expansion in v / c, here 0.22 at the start
    and 0.4 at the end: the last orbits are indicative only (the real ones
    need numerical relativity). Masses like these exist: the binary
    candidate OJ 287 is put at 1.8e10 and 1.5e8 solar masses.
    """
    from physengine.modules.postnewtonian import C_AU_PER_YEAR, circular_speed_1pn

    m = 5.0e9
    separation = 20.0 * G * 2.0 * m / C_AU_PER_YEAR**2
    v = circular_speed_1pn(m, m, separation, G)
    pos = [[-separation / 2, 0.0, 0.0], [separation / 2, 0.0, 0.0]]
    vel = [[0.0, -v / 2, 0.0], [0.0, v / 2, 0.0]]
    horizon = _black_hole_radius(G, m)
    return Scene(
        state=make_state(pos, vel, [m, m], [horizon, horizon], black_hole=[True, True]),
        names=["Black hole A", "Black hole B"],
        colors=[(0.55, 0.35, 0.95), (0.95, 0.45, 0.35)],
        radii=[horizon, horizon],  # drawn at their true size
        view_distance=11000.0,
        view_pitch=0.85,
        sheet=True,
        info="Inspiral by gravitational waves: merger after ~19 years (28 orbits)",
    )


def head_on_planets(G):
    """Two Earth-mass planets on the same orbit (4 AU) going opposite ways.

    They meet head-on after a quarter of an orbit (2 years) at twice the
    orbital speed, 2 * 3.14 = 6.3 AU/yr = 30 km/s. That is 1.3 times the
    speed at which two Earths lose half their mass (nbody_collisions.py):
    they shatter, leaving a remnant of about a tenth of the mass, which,
    having no orbital speed left, falls towards the star, and fragments
    scattered onto their own orbits.
    """
    r = 4.0
    v = np.sqrt(G * M_SUN / r)
    pos = [[0.0, 0.0, 0.0], [r, 0.0, 0.0], [-r, 0.0, 0.0]]
    vel = [[0.0, 0.0, 0.0], [0.0, v, 0.0], [0.0, v, 0.0]]  # the second one is retrograde
    return Scene(
        state=make_state(pos, vel, [M_SUN, M_EARTH, M_EARTH], [R_SUN, R_EARTH, R_EARTH]),
        names=["Sun", "Planet A", "Planet B"],
        colors=[SUN, (0.35, 0.6, 1.0), (0.9, 0.55, 0.35)],
        radii=[0.3, 0.13, 0.13],
        view_distance=13.0,
        view_pitch=0.5,
        info="Head-on collision at 30 km/s after 2 years",
    )


def star_torn_apart(G):
    """The Sun falling past a black hole of 316 000 solar masses, and being eaten.

    Lengths in units of r_g = G M / c^2 = 0.0031 AU (467 000 km): the Sun's
    radius is 1.5 r_g (the hole's horizon: 2 r_g), its tidal radius
    R (M / m)^(1/3) = 102 r_g. It starts as one body at the far end (106 r_g,
    outside the tidal radius) of an orbit that would bring its centre to
    TORN_PERICENTRE r_g. On crossing the tidal radius it becomes gas, which
    at first keeps the star's shape, then is drawn into a strand as the tide
    grows. Whatever comes within 6 r_g is followed on its exact orbit round
    the hole (nbody_plunge.py): the part with too little angular momentum
    winds in and is swallowed, the rest swings round and out again, and
    returns, because the orbit is bound. Everything is drawn at its true size.

    All this takes under an hour: r_g / c = 1.56 s, the fall 35 minutes. The
    scene therefore runs at 1e-5 years (5 minutes) per second.
    """
    from physengine.modules.postnewtonian import C_AU_PER_YEAR

    m_hole, m_star = 10.0**5.5, 1.0
    r_g = G * m_hole / C_AU_PER_YEAR**2
    apocentre, pericentre = 106.0 * r_g, TORN_PERICENTRE * r_g
    # Kepler ellipse: speed at the apocentre.
    v = np.sqrt(2.0 * G * (m_hole + m_star) * pericentre / (apocentre * (apocentre + pericentre)))
    # From the lower right (clear of the panels), anticlockwise round the hole.
    angle = np.radians(-28.0)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    sideways = np.array([-np.sin(angle), np.cos(angle), 0.0])
    pos = [[0.0, 0.0, 0.0], apocentre * direction]
    vel = [[0.0, 0.0, 0.0], v * sideways]
    horizon = _black_hole_radius(G, m_hole)
    return Scene(
        state=make_state(pos, vel, [m_hole, m_star], [horizon, R_SUN], black_hole=[True, False]),
        names=["Black hole", "Sun"],
        colors=[BLACK_HOLE, (1.0, 0.85, 0.55)],
        radii=[horizon, R_SUN],
        view_distance=140.0 * r_g,
        view_pitch=0.35,
        years_per_second=1.0e-5,
        info="The Sun and a 316 000 solar-mass hole, true sizes; 5 minutes per second",
    )


# Pericentre of the Sun's centre in "Star torn by a black hole", in units of
# G M / c^2. Chosen by trial so that most of the star is swallowed on the
# first passage (see the test for the measured fractions).
TORN_PERICENTRE = 7.8


CUSTOM = "Custom"


def empty(G):
    """No bodies: the Custom scene, built by hand in the scene editor."""
    return Scene(state=NBodyState(0), names=[], colors=[], radii=[])


SCENES = {
    "Sun-Earth": sun_earth,
    "Inner solar system": inner_solar_system,
    "Sun-Earth-Moon": sun_earth_moon,
    "Binary star": binary_star,
    "Circumbinary planet": circumbinary_planet,
    "Hierarchical triple": hierarchical_triple,
    "Figure-eight": figure_eight,
    "Jupiter Trojans": trojans,
    "S2 around Sgr A*": s2_sgr_a,
    "Star near a black hole": star_near_black_hole,
    "Black hole flyby": black_hole_flyby,
    "Black hole merger": black_hole_merger,
    "Star torn by a black hole": star_torn_apart,
    "Planets colliding": head_on_planets,
    CUSTOM: empty,
}
