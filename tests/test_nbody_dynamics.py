"""Tests of the N-body dynamics: integrator, adaptive substeps, collisions,
the preset scenes, and the orbit helpers of the scene editor.

Units: AU, years, solar masses, so G = 4 pi^2.

Reference values and their sources:

- Kepler's third law: a two-body orbit with semi-major axis a and total mass
  M has period P = 2 pi sqrt(a^3 / (G M)). For Sun-Earth (a = 1,
  M = 1 + 3e-6) that is 1 / sqrt(1 + 3e-6) yr.
- Vis-viva: at periapsis r_p = a (1 - e) the relative speed is
  sqrt(G M (1 + e) / (a (1 - e))); the orbit returns to r_p after one period.
- Leapfrog (Stormer-Verlet) is symplectic: for a fixed step its energy error
  stays bounded and does not grow over long runs (Hairer, Lubich & Wanner,
  Geometric Numerical Integration, 2nd ed., 2006, ch. IX). It also conserves
  linear and angular momentum up to rounding for pairwise central forces
  (same book, ch. IV.1 and VI.4).
- Free-fall time sqrt(r^3 / (G (m1 + m2))) and crossing time r / |v_rel|:
  definitions used by the adaptive step (nbodystate.min_timescale).
- Perfectly inelastic merger: total mass and total momentum are conserved,
  the merged body is at the centre of mass, and with equal density its
  volume is the sum, R = (R1^3 + R2^3)^(1/3).
- Hill radius r_H = a (m / (3 M))^(1/3) (Murray & Dermott, Solar System
  Dynamics, 1999, eq. 3.75); Make orbit places bodies at r_H / 3.
- Circular-orbit speed sqrt(G (M + m) / r), perpendicular to the separation.
- Scene stability bounds are those claimed in the scene docstrings
  (nbody_scenes.py): Hill-stable Moon, Trojans held at L4/L5, circumbinary
  planet beyond the Holman & Wiegert limit, hierarchical triple beyond the
  Mardling & Aarseth limit, figure-eight linearly stable.
"""

import time
from copy import deepcopy
from functools import partial

import numpy as np
import pytest

from physengine.modules import scenario_files
from physengine.modules.nbody_binary import invariants
from physengine.modules.nbody_collisions import (
    N_FRAGMENTS,
    N_TIDAL,
    disruption_energy,
    is_torn_before_horizon,
    largest_remnant_fraction,
    tidal_radius,
)
from physengine.modules.nbody_editor import (
    HILL_FRACTION,
    body_radius_without_horizon,
    circular_orbit_velocity,
    hill_limited_position,
    new_body_radius,
)
from physengine.modules.nbody_editor import settings as editor_settings
from physengine.modules.nbody_plunge import FREEZE_LOOK
from physengine.modules.nbody_scenes import (
    CUSTOM,
    M_EARTH,
    R_EARTH,
    R_SUN,
    SCENES,
    _kepler_pair,
    make_state,
    rocky_radius,
    star_radius,
)
from physengine.modules.nbody_trails import TrailRenderer
from physengine.modules.nbodymodule import (
    DEFAULT_SCENE,
    FAST_ORBIT,
    MAX_SLOW_MOTION,
    NEWTON,
    PIXELS_PER_STEP,
    RELATIVITY,
    SLOW_MOTION,
    VISIBLE_PIXELS,
    NBodyModule,
    make_integrator,
    step_time,
    validate_scenario,
)
from physengine.modules.nbodystate import (
    ETA,
    Integrator,
    acc,
    energy,
    first_contact,
    merge_pair,
    min_timescale,
    sun_earth_scenario,
    total_angular_momentum,
    total_momentum,
)
from physengine.modules.postnewtonian import C_AU_PER_YEAR, schwarzschild_radius

G = 4.0 * np.pi**2
force = partial(acc, G=G, eps=0.0)
timescale = partial(min_timescale, G=G)


def total_energy(state):
    return energy(state.pos, state.vel, state.mass, G, 0.0)


def binary(m1, m2, a, e):
    pos, vel = _kepler_pair(G, m1, m2, a, e)
    return make_state(pos, vel, [m1, m2])


def separation(state):
    return np.linalg.norm(state.pos[1] - state.pos[0])


# ----------------------------------------------------------------- timescale


def test_min_timescale_free_fall():
    # Two 1 M_sun bodies at rest 2 AU apart: sqrt(2^3 / (4 pi^2 * 2)) = 1 / pi.
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vel = np.zeros((2, 3))
    assert min_timescale(pos, vel, np.array([1.0, 1.0]), G) == pytest.approx(1.0 / np.pi, rel=1e-12)


def test_min_timescale_crossing():
    # Nearly massless bodies 1 AU apart closing at 4 AU/yr: crossing time 0.25 yr.
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [-4.0, 0.0, 0.0]])
    assert min_timescale(pos, vel, np.array([1e-12, 1e-12]), G) == pytest.approx(0.25, rel=1e-12)


def test_substep_level_follows_the_timescale():
    # k = ceil(log2(dt / (ETA * tau))), clamped to [0, MAX_LEVEL].
    # Earth around the Sun: free-fall and crossing times are both
    # 1 / (2 pi sqrt(1 + 3e-6)) ~ 0.159 yr, so with ETA = 0.02 and
    # dt = 1/240 yr: k = ceil(log2(0.00417 / 0.00318)) = 1.
    integrator = Integrator(force, timescale)
    dt = 1.0 / 240.0
    tau = 1.0 / (2.0 * np.pi * np.sqrt(1.0 + 3.0e-6))
    expected = int(np.ceil(np.log2(dt / (ETA * tau))))
    assert expected == 1
    assert integrator.level(sun_earth_scenario(), dt) == expected

    # A Jupiter-like orbit (5.2 AU, tau ~ 1.9 yr) needs no substeps.
    wide = binary(1.0, 1e-3, 5.2, 0.0)
    assert integrator.level(wide, dt) == 0

    # Without a timescale function the integrator is the plain leapfrog.
    assert Integrator(force).level(sun_earth_scenario(), dt) == 0


# --------------------------------------------------------------- integrator


def test_circular_orbit_closes_after_one_period():
    state = sun_earth_scenario()
    start = state.pos.copy()
    period = 1.0 / np.sqrt(1.0 + 3.0e-6)
    n = 1000

    Integrator(force, timescale).step(state, period / n, n)

    # Leapfrog phase error after one orbit is ~ 2 pi (w dt)^2 / 24 ~ 1e-5 rad.
    np.testing.assert_allclose(state.pos, start, atol=1e-4)


def test_eccentric_orbit_returns_to_periapsis():
    m1, m2, a, e = 1.0, 0.6, 1.0, 0.5
    state = binary(m1, m2, a, e)
    assert separation(state) == pytest.approx(a * (1 - e), rel=1e-12)

    period = 2.0 * np.pi * np.sqrt(a**3 / (G * (m1 + m2)))
    n = 2000
    Integrator(force, timescale).step(state, period / n, n)

    assert separation(state) == pytest.approx(a * (1 - e), rel=1e-3)


def test_leapfrog_energy_error_does_not_grow():
    # Plain leapfrog (no adaptive substeps), 200 orbits of an e = 0.5 binary
    # at 200 steps per orbit: the energy error over the last 10 orbits is no
    # larger than over the first 10, i.e. there is no secular drift.
    m1, m2, a, e = 1.0, 0.6, 1.0, 0.5
    state = binary(m1, m2, a, e)
    period = 2.0 * np.pi * np.sqrt(a**3 / (G * (m1 + m2)))
    steps_per_orbit = 200
    integrator = Integrator(force)

    e0 = total_energy(state)
    errors = []
    for _ in range(200 * steps_per_orbit):
        integrator.step(state, period / steps_per_orbit, 1)
        errors.append(abs(total_energy(state) - e0) / abs(e0))

    first = max(errors[: 10 * steps_per_orbit])
    last = max(errors[-10 * steps_per_orbit :])
    assert last < 1.5 * first
    assert last < 1e-2


def test_adaptive_substeps_resolve_close_periapsis():
    # e = 0.9 binary of two 1 M_sun stars, a = 1 AU, at the module's step of
    # 1/240 yr. Periapsis is 0.1 AU, where the free-fall time is ~0.0036 yr,
    # shorter than the step itself: the plain leapfrog fails there.
    m, a, e, dt = 1.0, 1.0, 0.9, 1.0 / 240.0
    period = 2.0 * np.pi * np.sqrt(a**3 / (G * 2 * m))
    n_steps = int(10 * period / dt)

    def energy_error(integrator):
        state = binary(m, m, a, e)
        e0 = total_energy(state)
        worst, levels = 0.0, []
        for _ in range(n_steps):
            integrator.step(state, dt, 1)
            levels.append(integrator.last_level)
            worst = max(worst, abs(total_energy(state) - e0) / abs(e0))
        return worst, levels

    plain, _ = energy_error(Integrator(force))
    adaptive, levels = energy_error(Integrator(force, timescale))

    # The adaptive scheme must be at least 100 times better than the plain
    # leapfrog and stay below 5e-3 worst case (at periapsis).
    assert adaptive < plain / 100
    assert adaptive < 5e-3
    # Substeps only near periapsis: level 0 most of the time, > 0 at periapsis.
    assert max(levels) >= 4
    assert np.mean(np.array(levels) == 0) > 0.5


def test_adaptive_energy_error_does_not_drift():
    # The figure-eight switches between substep levels all the time. With
    # time-symmetric level selection the energy error stays bounded: the
    # worst error in years 15-20 is no larger than in year 1. (Without the
    # symmetric selection it grew by ~8e-6 per year.)
    state = SCENES["Figure-eight"](G).state
    integrator = Integrator(force, timescale)
    e0 = total_energy(state)
    yearly = []
    for _ in range(20):
        worst = 0.0
        for _ in range(240):
            integrator.step(state, 1.0 / 240.0, 1)
            worst = max(worst, abs(total_energy(state) - e0) / abs(e0))
        yearly.append(worst)
    assert max(yearly[15:]) < 1.5 * yearly[0]


def test_momentum_and_angular_momentum_conserved():
    rng = np.random.default_rng(seed=2026)
    n = 6
    state = make_state(
        rng.uniform(-2.0, 2.0, size=(n, 3)),
        rng.uniform(-3.0, 3.0, size=(n, 3)),
        rng.uniform(1e-3, 1.0, size=n),
    )
    p0, l0 = total_momentum(state), total_angular_momentum(state)
    p_scale = np.sum(state.mass[:, None] * np.abs(state.vel))
    l_scale = np.sum(state.mass * np.linalg.norm(np.cross(state.pos, state.vel), axis=1))

    Integrator(force, timescale).step(state, 1.0 / 240.0, 500)

    np.testing.assert_allclose(total_momentum(state), p0, atol=1e-12 * p_scale)
    np.testing.assert_allclose(total_angular_momentum(state), l0, atol=1e-11 * l_scale)


# ------------------------------------------------------------------ contact


def test_contact_caught_when_bodies_pass_through_each_other():
    # Bodies at x = -1 and x = +1 swap places within one drift: the ends do
    # not overlap, but the paths do.
    pos0 = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    radius = np.array([0.01, 0.01])
    assert first_contact(pos0, v, radius, 0.02) == (0, 1)


def test_no_contact_on_a_near_miss():
    # Same, but offset by 0.05 AU sideways; the radii only add up to 0.02.
    pos0 = np.array([[-1.0, 0.0, 0.0], [1.0, 0.05, 0.0]])
    v = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    assert first_contact(pos0, v, np.array([0.01, 0.01]), 0.02) is None


def test_point_masses_never_touch():
    pos0 = np.zeros((2, 3))
    assert first_contact(pos0, np.zeros((2, 3)), np.array([0.0, 0.1]), 0.01) is None


def test_merge_conserves_mass_momentum_and_volume():
    state = make_state(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [1.0, 3.0],
        radius=[1.0, 2.0],
        centre_of_mass_frame=False,
    )
    merge_pair(state, 1, 0)

    assert state.mass[1] == 4.0
    np.testing.assert_allclose(state.pos[1], [0.75, 0.0, 0.0])  # (1*0 + 3*1) / 4
    np.testing.assert_allclose(state.vel[1], [0.25, 1.5, 0.0])  # (1*(1,0) + 3*(0,2)) / 4
    assert state.radius[1] == pytest.approx(9.0 ** (1.0 / 3.0))


def _two_bodies_head_on(speed):
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = CUSTOM
    m.load_scene(CUSTOM)
    for x, vx in [(-0.25, speed), (0.25, -speed)]:
        m.add_body(
            {
                "mass": 1e-6,
                "pos": [x, 0.0, 0.0],
                "vel": [vx, 0.0, 0.0],
                "color": (1, 1, 1),
                "radius": 0.02,
                "phys_radius": 0.01,
            }
        )

    for _ in range(240):
        m.step(m.dt)
        if m.nb.N != 2:
            break
    return m


def test_head_on_collision_merges_in_the_module():
    # Relative speed 1 AU/yr (4.7 km/s), far below the 3.3 AU/yr at which two
    # bodies of 1e-6 solar masses lose half their mass: a clean merger.
    m = _two_bodies_head_on(0.5)
    assert m.nb.N == 1
    assert m.nb.mass[0] == pytest.approx(2e-6, rel=1e-12)
    np.testing.assert_allclose(total_momentum(m.nb), 0.0, atol=1e-18)
    np.testing.assert_allclose(m.nb.pos[0], 0.0, atol=1e-12)
    assert m.nb.radius[0] == pytest.approx(0.01 * 2 ** (1 / 3))
    assert len(m.scene.names) == 1


# ------------------------------------------------------------------- scenes


def run_scene(name, years):
    m = NBodyModule()
    m.scene_name = name
    m.load_scene(name)
    n0 = m.nb.N
    e0 = total_energy(m.nb)
    for _ in range(int(round(years / m.dt))):
        m.step(m.dt)
    assert m.nb.N == n0, "bodies merged in a scene that should have no collisions"
    return m, abs(total_energy(m.nb) - e0) / abs(e0)


@pytest.mark.parametrize(
    "name, years, max_energy_error",
    [
        ("Sun-Earth", 10, 1e-10),
        ("Inner solar system", 10, 1e-5),
        ("Sun-Earth-Moon", 2, 1e-6),
        ("Binary star", 10, 1e-4),
        ("Circumbinary planet", 10, 1e-4),
        ("Hierarchical triple", 10, 1e-4),
        ("Figure-eight", 10, 1e-4),
        ("Jupiter Trojans", 12, 1e-10),
    ],
)
def test_scene_conserves_energy(name, years, max_energy_error):
    _, err = run_scene(name, years)
    assert err < max_energy_error


@pytest.mark.parametrize(
    "name, steps, max_energy_error",
    [
        # Measured 2026-11-23 (and unchanged by that day's pace changes):
        # 1.95e-5, 6.6e-4 and 9.0e-10. S2: 32 years, two orbits. "Star near
        # a black hole" wobbles by ~6e-4 with the orbit, the size of the
        # 2PN terms the 1PN energy leaves out (scene docstring).
        ("S2 around Sgr A*", 32 * 240, 1e-4),
        ("Star near a black hole", 10 * 240, 2e-3),
        ("Black hole flyby", 10 * 240, 1e-8),
    ],
)
def test_hole_scene_conserves_energy(name, steps, max_energy_error):
    # The scenes with a black hole and no collision, under the module's
    # default gravity (1PN): the conserved quantity is the 1PN energy
    # (postnewtonian.eih_energy), which m.energy() returns.
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = name
    m.load_scene(name)
    n0, e0 = m.nb.N, m.energy()
    for _ in range(steps):
        m.step(m.dt)
    assert m.nb.N == n0
    assert abs(m.energy() - e0) / abs(e0) < max_energy_error


def test_moon_stays_bound_to_earth():
    m, _ = run_scene("Sun-Earth-Moon", 2)
    assert np.linalg.norm(m.nb.pos[2] - m.nb.pos[1]) < 0.005  # starts at 0.00257


def test_trojans_stay_at_lagrange_points():
    m, _ = run_scene("Jupiter Trojans", 12)
    for k in (2, 3):
        d = np.linalg.norm(m.nb.pos[k] - m.nb.pos[1])
        assert d == pytest.approx(5.2026, rel=0.02)


def test_circumbinary_planet_stays_in_orbit():
    m, _ = run_scene("Circumbinary planet", 10)
    nb = m.nb
    binary_com = nb.mass[:2] @ nb.pos[:2] / nb.mass[:2].sum()
    assert 1.6 < np.linalg.norm(nb.pos[2] - binary_com) < 2.4  # starts at 2 AU


def test_hierarchical_triple_stays_hierarchical():
    m, _ = run_scene("Hierarchical triple", 10)
    nb = m.nb
    assert separation(nb) < 0.7  # inner pair, starts at 0.5 AU
    inner_com = nb.mass[:2] @ nb.pos[:2] / nb.mass[:2].sum()
    assert 4.0 < np.linalg.norm(nb.pos[2] - inner_com) < 6.0  # starts at 5 AU


def test_figure_eight_stays_bounded():
    m, _ = run_scene("Figure-eight", 10)
    assert np.abs(m.nb.pos).max() < 1.3  # the curve spans |x| < ~1.08


def test_every_scene_loads():
    for name, build in SCENES.items():
        scene = build(G)
        n = scene.state.N
        assert len(scene.names) == len(scene.colors) == len(scene.radii) == n, name


# -------------------------------------------------------------- orbit helper


def test_hill_rule_moves_mars_into_earths_hill_sphere():
    nb = SCENES["Inner solar system"](G).state
    earth, mars, sun = 3, 4, 0
    new = hill_limited_position(nb, mars, earth)

    a = np.linalg.norm(nb.pos[earth] - nb.pos[sun])
    r_hill = a * (nb.mass[earth] / (3 * nb.mass[sun])) ** (1 / 3)  # ~0.0100 AU
    assert np.linalg.norm(new - nb.pos[earth]) == pytest.approx(HILL_FRACTION * r_hill)
    # Same direction from Earth as before.
    before = (nb.pos[mars] - nb.pos[earth]) / np.linalg.norm(nb.pos[mars] - nb.pos[earth])
    after = (new - nb.pos[earth]) / np.linalg.norm(new - nb.pos[earth])
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_hill_rule_leaves_close_bodies_and_star_orbits_alone():
    nb = SCENES["Sun-Earth-Moon"](G).state
    assert hill_limited_position(nb, 2, 1) is None  # Moon is well inside Earth's Hill sphere
    nb = SCENES["Inner solar system"](G).state
    assert hill_limited_position(nb, 4, 0) is None  # nothing outweighs the Sun


def test_circular_orbit_velocity():
    pos, central = np.array([2.0, 0.0, 0.0]), np.zeros(3)
    v = circular_orbit_velocity(G, pos, 1e-6, np.zeros(3), central, np.zeros(3), 1.0)
    assert np.linalg.norm(v) == pytest.approx(np.sqrt(G * (1.0 + 1e-6) / 2.0), rel=1e-12)
    assert v @ pos == pytest.approx(0.0, abs=1e-12)
    assert v[1] > 0  # counter-clockwise seen from +z when there is no prior motion


def test_disruption_energy_of_two_earths():
    # Leinhardt & Stewart (2012): Q*_RD = c* (4/5) pi rho_1 G R_C1^2 with
    # c* = 1.9, rho_1 = 1000 kg/m^3 and R_C1 the radius of both masses at
    # that density. Two Earths (5.972e24 kg each): R_C1 = 1.418e7 m,
    # Q*_RD = 1.9 * 0.8 pi * 1000 * 6.674e-11 * (1.418e7)^2 = 6.41e7 J/kg
    # = 2.85 (AU/yr)^2. Half the mass is lost at Q = v^2 / 8 = Q*_RD:
    # v = 22.6 km/s = 4.78 AU/yr.
    m_earth = 3.003e-6
    assert disruption_energy(m_earth, m_earth, G) == pytest.approx(2.85, rel=0.01)
    assert largest_remnant_fraction(m_earth, m_earth, 4.78, G) == pytest.approx(0.5, abs=0.01)
    assert largest_remnant_fraction(m_earth, m_earth, 0.5, G) > 0.99


def test_fast_collision_shatters_and_conserves_mass_and_momentum():
    # Relative speed 20 AU/yr (95 km/s), six times the half-mass speed:
    # super-catastrophic. A remnant and N_FRAGMENTS pieces of debris.
    m = _two_bodies_head_on(10.0)
    nb = m.nb
    assert nb.N == 1 + N_FRAGMENTS
    assert nb.debris.sum() == N_FRAGMENTS
    assert nb.mass.sum() == pytest.approx(2e-6, rel=1e-12)
    kept = largest_remnant_fraction(1e-6, 1e-6, 20.0, G)
    assert nb.mass[~nb.debris][0] == pytest.approx(kept * 2e-6, rel=1e-3)
    np.testing.assert_allclose(total_momentum(nb), 0.0, atol=1e-15)


def _hole_and_star(hole_mass):
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = CUSTOM
    m.load_scene(CUSTOM)
    common = {"color": (1, 1, 1), "radius": 0.02}
    m.add_body(
        {
            "mass": hole_mass,
            "pos": [0.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "phys_radius": schwarzschild_radius(hole_mass, G),
            "black_hole": True,
            **common,
        }
    )
    # A Sun dropped from rest, 20 capture radii or tidal radii out.
    r_t = R_SUN * hole_mass ** (1.0 / 3.0)
    start = 20.0 * max(r_t, 3.0 * schwarzschild_radius(hole_mass, G))
    m.add_body(
        {
            "mass": 1.0,
            "pos": [start, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "phys_radius": R_SUN,
            **common,
        }
    )
    events = []
    for _ in range(100000):
        n = m.nb.N
        m.step(m.dt)
        if m.nb.N != n:
            events.append(m.nb.N)
        if m.nb.N == 1 and m.close.count == 0:
            break
    return m, events


def test_star_is_torn_by_a_small_hole_and_swallowed_whole_by_a_huge_one():
    # Tidal radius of the Sun: R_sun (M / M_sun)^(1/3). At 1e6 solar masses
    # it is 0.47 AU, 8 times the capture radius 6 G M / c^2 = 0.059 AU: the
    # star is torn into N_TIDAL pieces first, which then fall in (dropped
    # from rest, so all of it ends in the hole). At 1e9 solar masses the
    # capture radius (59 AU) is outside the tidal radius (4.7 AU): the star
    # goes in whole.
    m, events = _hole_and_star(1e6)
    assert events[0] == 1 + N_TIDAL
    assert m.nb.N == 1 and m.nb.black_hole[0]
    assert m.nb.mass[0] == pytest.approx(1e6 + 1.0, rel=1e-12)
    m, events = _hole_and_star(1e9)
    assert events == [1]
    assert m.nb.mass[0] == pytest.approx(1e9 + 1.0, rel=1e-12)


def test_star_torn_scene_is_mostly_swallowed():
    # The Sun on an orbit whose centre would pass 7.8 G M / c^2 from a hole
    # of 3.2e5 solar masses (the Sun's radius: 1.5 G M / c^2). It starts as
    # one body just outside its tidal radius (102 G M / c^2), becomes
    # N_TIDAL pieces on crossing it, and the pieces that come within
    # 20 G M / c^2 are followed on exact geodesics (nbody_plunge.py): those
    # that plunge are swallowed, the others come out again. Measured: about
    # two thirds swallowed on the first passage; the hole gains their mass.
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = "Star torn by a black hole"
    m.load_scene(m.scene_name)
    assert m.nb.N == 2  # one star, not yet torn
    hole_mass = m.nb.mass[0]
    for _ in range(15 * 240):
        m.step(m.dt)
    assert m.close.count == 0
    left = int(m.nb.gas.sum())
    # Measured 190 of 300 pieces (0.633) with the module's fixed random
    # seed; the window allows for the few pieces whose fate can change with
    # how a step was cut short (STEP_BUDGET is wall-clock time).
    assert 0.56 < 1.0 - left / N_TIDAL < 0.70
    assert m.nb.mass[0] == pytest.approx(hole_mass + (N_TIDAL - left) / N_TIDAL, rel=1e-9)


def test_exact_capture_criterion_near_a_hole():
    # A light body falling from far away (at rest at infinity) is captured
    # by a Schwarzschild hole if its angular momentum per unit mass is below
    # 4 G M / c, and swings round and escapes above it (e.g. Misner, Thorne
    # & Wheeler 1973, sec. 25.5). Bodies with L = 3.9 and 4.1 G M / c both
    # come within 20 G M / c^2 and are handed to the exact orbit there: the
    # first must be swallowed, the second come back out.
    from physengine.modules.postnewtonian import C_AU_PER_YEAR

    hole_mass = 1.0e6
    r_g = G * hole_mass / C_AU_PER_YEAR**2
    outcome = {}
    for L in (3.9, 4.1):
        m = NBodyModule()
        m.discard_edits()
        m.scene_name = CUSTOM
        m.load_scene(CUSTOM)
        common = {"color": (1, 1, 1), "radius": 0.02}
        m.add_body(
            {
                "mass": hole_mass,
                "pos": [0.0, 0.0, 0.0],
                "vel": [0.0, 0.0, 0.0],
                "phys_radius": schwarzschild_radius(hole_mass, G),
                "black_hole": True,
                **common,
            }
        )
        start = 60.0 * r_g
        speed = np.sqrt(2.0 * G * hole_mass / start)
        # L = r^2 dphi/dt (1 + v^2 / (2 c^2) + 3 G M / (r c^2)) at 1PN order.
        correction = 1.0 + 0.5 * speed**2 / C_AU_PER_YEAR**2 + 3.0 * r_g / start
        sideways = L * r_g * C_AU_PER_YEAR / start / correction
        m.add_body(
            {
                "mass": 1e-9,
                "pos": [start, 0.0, 0.0],
                "vel": [-np.sqrt(speed**2 - sideways**2), sideways, 0.0],
                "phys_radius": 1e-12,  # too compact to be torn
                **common,
            }
        )
        entered = False
        for _ in range(4000):
            m.step(m.dt)
            entered |= m.close.count > 0
            if entered and m.close.count == 0:
                break
        outcome[L] = (entered, m.nb.N)
    assert outcome[3.9] == (True, 1)  # swallowed
    assert outcome[4.1] == (True, 2)  # came back out


# ------------------------------------------------- black holes in Custom mode
#
# Helpers: a Custom scene as the editor builds it, seen through a view. The
# module's render() sets visible_scale to the world size of VISIBLE_PIXELS
# (2 d tan(fov / 2) / H per pixel for camera distance d, fov = 1 rad,
# window height H) and view_zone to d; the tests set the same numbers.

VIEW_DISTANCE, VIEW_HEIGHT = 5.0, 1000
AU_PER_PIXEL = 2.0 * VIEW_DISTANCE * np.tan(0.5) / VIEW_HEIGHT  # 0.00546 AU


def custom_module(view=True):
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = CUSTOM
    m.load_scene(CUSTOM)
    if view:
        m.visible_scale = VISIBLE_PIXELS * AU_PER_PIXEL
        m.view_zone = VIEW_DISTANCE
    return m


def place(m, kind, mass, pos, vel=(0.0, 0.0, 0.0), phys_radius=None):
    """Add a body the way the editor does (kind: "Body", "Star" or "Black hole")."""
    radius = new_body_radius(kind, mass, G) if phys_radius is None else phys_radius
    hole = kind == "Black hole"
    m.add_body(
        {
            "mass": mass,
            "pos": list(pos),
            "vel": list(vel),
            "color": (1.0, 1.0, 1.0),
            "radius": radius if hole else 0.02,
            "phys_radius": radius,
            "black_hole": hole,
        }
    )
    return m.ids[-1]


def circular(mass, r):
    return (0.0, float(np.sqrt(G * mass / r)), 0.0)


def gas_pieces(m):
    """Pieces of gas, in the state and on exact orbits round a hole."""
    away = sum(meta["gas"] for close in m.close.holes.values() for meta in close.meta)
    return int(m.nb.gas.sum()) + away


def count_fades(m):
    """Record the calls of m._fade: returns the list they are appended to."""
    calls, original = [], m._fade

    def fade(*args, **kwargs):
        calls.append(kwargs.get("look", 1.0))
        return original(*args, **kwargs)

    m._fade = fade
    return calls


@pytest.mark.parametrize("trigger", ["star dropped", "planet torn on its orbit"])
def test_scene_returns_to_its_pace_after_a_tear_near_a_small_hole(trigger):
    # A 10 solar-mass hole with a planet on a circular orbit at 1 AU, in the
    # default view (1 px = 0.00546 AU). Then either a Sun dropped from rest
    # at 0.1 AU, or a rocky planet on a circular orbit at 0.005 AU: both are
    # torn (tidal radii 0.010 and 0.0064 AU, 1 to 2 px) and their debris is
    # bound to the hole on orbits of hours, below a pixel. Before 2026-11-23
    # that debris kept the scene x540 to x1800 slower for good and the outer
    # planet moved 0.2 to 1.7 px per 60 steps.
    #
    # Reference: the outer planet's orbit, P = 2 pi sqrt(r^3 / (G M)) =
    # 1 / sqrt(10) yr = 75.9 steps of 1/240 yr. In 60 steps it turns by
    # 2 pi 60 / 75.9 = 4.97 rad, a chord of 2 sin(4.97 / 2) = 1.226 AU =
    # 224 px. Bounds: the star needs ~10 steps of slow motion to fall
    # 0.1 AU = 18 px (PIXELS_PER_STEP), measured 60 steps in all until
    # nothing is left; 240 steps (one second) is allowed.
    m = custom_module()
    hole = place(m, "Black hole", 10.0, (0.0, 0.0, 0.0))
    planet = place(m, "Body", 1e-6, (1.0, 0.0, 0.0), circular(10.0, 1.0))
    if trigger == "star dropped":
        place(m, "Star", 1.0, (0.1, 0.0, 0.0))
    else:
        place(m, "Body", 1e-6, (0.005, 0.0, 0.0), circular(10.0, 0.005))
    fades = count_fades(m)
    settled = None
    for step in range(1, 241):
        m.step(m.dt)
        quiet = m.slow_motion == 1.0 and not m.slowed
        if quiet and m.nb.N == 2 and m.close.count == 0:
            settled = step
            break
    assert settled is not None, "debris still holds the scene after 240 steps"
    assert len(fades) == N_TIDAL  # every piece was shown going into the hole
    assert m.nb.mass[m.index_of(hole)] == pytest.approx(
        10.0 + (1.0 if trigger == "star dropped" else 1e-6), rel=1e-9
    )

    def planet_position():
        return m.nb.pos[m.index_of(planet)] - m.nb.pos[m.index_of(hole)]

    before = planet_position()
    for _ in range(60):
        m.step(m.dt)
        # (The star's momentum left the planet on an eccentric orbit, whose
        # pericentre may be slowed by a few percent: the timescale rule.)
        assert m.slow_motion < 1.5 and not m.slowed
    moved = np.linalg.norm(planet_position() - before)
    if trigger == "planet torn on its orbit":
        assert moved == pytest.approx(1.226, rel=0.02)
    else:
        # The hole now has 11 solar masses and the star's momentum, so the
        # planet is on another orbit; it must move at an orbit's pace.
        assert moved > 0.3


def test_slow_motion_is_only_for_events_that_can_be_seen():
    # step_time for a 10 solar-mass hole and one body, in the default view:
    # visible = 4 px = 0.0218 AU, pixel rule 1.5 px per step. G M = 394.8.
    visible = VISIBLE_PIXELS * AU_PER_PIXEL
    sim_dt = 1.0 / 240.0

    def pace(pos, vel, view=True, radius=None):
        m = custom_module(view=False)
        place(m, "Black hole", 10.0, (0.0, 0.0, 0.0))
        place(m, "Body", 1e-6, pos, vel, phys_radius=radius)
        if not view:
            return step_time(m.nb, sim_dt)
        return step_time(m.nb, sim_dt, visible, VIEW_DISTANCE, m.scene.radii)

    gm = G * (10.0 + 1e-6)
    # A planet on a circular orbit at 1 AU: P = 75.9 steps, dynamical time
    # P / (2 pi) = 12.1 steps, and SLOW_MOTION of that exceeds a step: not
    # slowed. (Before, the pixel rule held every body near a hole to 1.5 px
    # per step: x13 for this one.)
    assert pace((1.0, 0.0, 0.0), circular(10.0, 1.0)) == sim_dt
    # A circular orbit at 0.1 AU takes 2.4 steps, fewer than FAST_ORBIT: a
    # fast orbit, left at the scene's pace (it used to be x26, for good).
    assert 2.0 * np.pi * np.sqrt(0.1**3 / gm) / sim_dt < FAST_ORBIT
    assert pace((0.1, 0.0, 0.0), circular(10.0, 0.1)) == sim_dt
    # The same body at rest there is falling into the hole: an event.
    # Free-fall time sqrt(r^3 / (G M)) = 1.59e-3 yr, times SLOW_MOTION.
    assert pace((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)) == pytest.approx(
        SLOW_MOTION * np.sqrt(0.1**3 / gm), rel=1e-9
    )
    # At rest 0.01 AU from the hole it is under the hole's dot (2 px from
    # its centre; holes are drawn at least 4 px): not slowed for.
    assert pace((0.01, 0.0, 0.0), (0.0, 0.0, 0.0)) == sim_dt
    # Without a view nothing is hidden: slowed, by the free-fall time.
    assert pace((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), view=False) == pytest.approx(
        SLOW_MOTION * np.sqrt(0.01**3 / gm), rel=1e-9
    )
    # ... but never beyond MAX_SLOW_MOTION (free-fall time from 1e-5 AU:
    # 1.6e-9 yr; the floor is the handover radius, 2e-6 AU: 1.4e-10 yr).
    assert pace((1e-5, 0.0, 0.0), (0.0, 0.0, 0.0), view=False) == sim_dt / MAX_SLOW_MOTION
    # A passage at 100 AU/yr (escape speed at 1 AU: 28 AU/yr): the pixel
    # rule, PIXELS_PER_STEP pixels of travel per step.
    assert pace((1.0, 0.0, 0.0), (0.0, 100.0, 0.0)) == pytest.approx(
        PIXELS_PER_STEP * AU_PER_PIXEL / 100.0, rel=1e-9
    )
    # A point mass (radius 0) is never captured or torn: on the fast orbit
    # it is not an event whatever its pericentre.
    assert pace((0.1, 0.0, 0.0), (0.0, 1.0, 0.0), radius=0.0) == sim_dt


def test_hills_mass():
    # A body is torn only if its tidal radius R (M / m)^(1/3) is outside
    # the horizon 2 G M / c^2 (Hills 1975): M < (R c^2 / (2 G))^(3/2) m^(-1/2).
    # With G / c^2 = 9.871e-9 AU per solar mass: the Sun (R = 0.0046505 AU)
    # 1.14e8 solar masses; a body of Earth's density, 5.8e7 whatever its mass.
    g_c2 = G / C_AU_PER_YEAR**2
    hills_sun = (R_SUN / (2.0 * g_c2)) ** 1.5
    assert hills_sun == pytest.approx(1.14e8, rel=0.01)
    for factor, torn in ((0.9, True), (1.1, False)):
        mass = factor * hills_sun
        assert is_torn_before_horizon(R_SUN, 1.0, mass, schwarzschild_radius(mass, G)) is torn
    hills_rock = (R_EARTH / (2.0 * g_c2)) ** 1.5 / np.sqrt(M_EARTH)
    assert hills_rock == pytest.approx(5.8e7, rel=0.02)
    for m_body in (1e-9, 1e-3):
        for factor, torn in ((0.9, True), (1.1, False)):
            mass = factor * hills_rock
            horizon = schwarzschild_radius(mass, G)
            assert is_torn_before_horizon(rocky_radius(m_body), m_body, mass, horizon) is torn


@pytest.mark.parametrize("distance", [0.10, 0.15, 0.30, 0.44])
def test_star_inside_its_tidal_radius_is_torn_on_both_sides_of_the_handover_radius(distance):
    # A million solar masses: handover radius 20 G M / c^2 = 0.197 AU, the
    # Sun's tidal radius 0.465 AU. A Sun placed at rest anywhere inside the
    # tidal radius is torn at once, inside the handover radius as well as
    # outside it. (Before 2026-11-23 it was swallowed whole inside.)
    m = custom_module()
    place(m, "Black hole", 1.0e6, (0.0, 0.0, 0.0))
    place(m, "Star", 1.0, (distance, 0.0, 0.0))
    assert distance < tidal_radius(R_SUN, 1.0, 1.0e6) == pytest.approx(0.465, rel=1e-3)
    m.step(m.dt)
    assert gas_pieces(m) == N_TIDAL
    for _ in range(400):
        m.step(m.dt)
        if m.nb.N == 1 and m.close.count == 0:
            break
    assert m.nb.mass[0] == pytest.approx(1.0e6 + 1.0, rel=1e-12)  # dropped from rest: all of it


def test_light_body_is_torn_at_its_tidal_radius_inside_the_handover_sphere():
    # 5e7 solar masses, below the Sun's Hills mass (1.14e8): horizon
    # 0.987 AU, handover radius 9.87 AU, the Sun's tidal radius
    # R_sun (5e7)^(1/3) = 1.713 AU, between the two. A Sun dropped from
    # rest at 15 AU is handed to its exact orbit whole and torn when it
    # gets to the tidal radius (checked every 4 G M / c^3 of its fall
    # there, at up to light speed: within 4 G M / c^2 = 2 AU inside it).
    m = custom_module()
    hole = place(m, "Black hole", 5.0e7, (0.0, 0.0, 0.0))
    place(m, "Star", 1.0, (15.0, 0.0, 0.0))
    r_t = tidal_radius(R_SUN, 1.0, 5.0e7)
    assert r_t == pytest.approx(1.713, rel=1e-3)
    whole_inside, torn_at = False, None
    last = 15.0
    for _ in range(3000):
        m.step(m.dt)
        away = m.away()
        if away:
            whole_inside = True
            last = float(np.linalg.norm(away[0][2]))
        if torn_at is None and gas_pieces(m) > 0:
            torn_at = last
        if m.nb.N == 1 and m.close.count == 0:
            break
    assert whole_inside, "the star was never followed whole inside the handover sphere"
    assert torn_at is not None and 0.0 < torn_at < 1.3 * r_t + 2.0
    assert m.nb.mass[m.index_of(hole)] == pytest.approx(5.0e7 + 1.0, rel=1e-12)


@pytest.mark.parametrize("distance", [10.0, 100.0, 300.0])
def test_star_goes_whole_into_a_hole_above_its_hills_mass(distance):
    # 1e9 solar masses, above the Sun's Hills mass: horizon 19.7 AU,
    # handover radius 197 AU, tidal radius 4.65 AU (inside the horizon).
    # Placed inside the horizon, inside the handover sphere or outside it,
    # the star is never torn; it ends in the hole, with one fade.
    m = custom_module()
    place(m, "Black hole", 1.0e9, (0.0, 0.0, 0.0))
    place(m, "Star", 1.0, (distance, 0.0, 0.0))
    fades = count_fades(m)
    for _ in range(3000):
        m.step(m.dt)
        assert gas_pieces(m) == 0
        if m.nb.N == 1 and m.close.count == 0:
            break
    assert m.nb.N == 1 and m.nb.mass[0] == pytest.approx(1.0e9 + 1.0, rel=1e-12)
    assert len(fades) == 1


def test_every_way_into_a_hole_ends_in_a_fade():
    # The swallowed body is shown frozen on the horizon, reddening and
    # fading (display). Each path must make exactly one fade per body:
    r_s = schwarzschild_radius(1.0e9, G)

    def run(build, steps=3000):
        m = custom_module()
        place(m, "Black hole", 1.0e9, (0.0, 0.0, 0.0))
        build(m)
        fades, looks = count_fades(m), []
        for _ in range(steps):
            for close in m.close.holes.values():
                looks.extend(close.look())
            m.step(m.dt)
            if m.nb.N == 1 and m.close.count == 0:
                break
        assert m.nb.N == 1 and m.close.count == 0
        return fades, looks

    # 1. On its exact orbit (falls from outside the handover sphere, with a
    #    quarter of the circular speed: angular momentum 1.3 G M / c, far
    #    below the 4 G M / c that would let it out again). The fade starts
    #    from the look the body froze with, FREEZE_LOOK, so its brightness
    #    does not jump; on the way in the look only ever goes down to there.
    fades, looks = run(
        lambda m: place(
            m, "Body", 1e-6, (260.0, 0.0, 0.0), (0.0, 0.25 * circular(1e9, 260.0)[1], 0.0)
        )
    )
    assert fades == [FREEZE_LOOK]
    assert looks[0] == pytest.approx(1.0, abs=0.02) and min(looks) >= FREEZE_LOOK
    assert all(b <= a + 1e-9 for a, b in zip(looks[-10:], looks[-9:], strict=False))
    # 2. Placed inside the horizon: no exact orbit starts there, the
    #    handover is refused and the hole takes it at once.
    fades, _ = run(lambda m: place(m, "Body", 1e-6, (0.3 * r_s, 0.0, 0.0)))
    assert fades == [1.0]
    # 3. At 2.5 G M / c^2 with the Newtonian circular speed, 0.63 c, which
    #    no body can have there (the four-velocity cannot be normalised).
    fades, _ = run(
        lambda m: place(m, "Body", 1e-6, (1.25 * r_s, 0.0, 0.0), circular(1e9, 1.25 * r_s))
    )
    assert fades == [1.0]
    # 4. Too heavy for an exact orbit (5% of the hole): merges at the
    #    capture radius.
    fades, _ = run(lambda m: place(m, "Body", 5.0e7, (150.0, 0.0, 0.0), phys_radius=1e-3))
    assert fades == [1.0]


# ------------------------------------------------------------------ identity


def test_body_keeps_its_identity_across_close_passes():
    # A compact body on an orbit between 8 and 30 G M / c^2 round a 10
    # solar-mass hole goes inside the handover sphere (20 G M / c^2) on
    # every orbit and comes out again. It must come back as the same body:
    # same id, same place in the list (a planet is listed after it), same
    # name, still selected and followed. (Zoomed in until the orbit, 3e-6 AU
    # across, can be seen, 1 px = 1e-7 AU: in the default view it is far
    # below a pixel and is followed analytically without being handed
    # over, see test_unseen_tight_pair_does_not_hold_the_scene.)
    m = custom_module()
    m.visible_scale, m.view_zone = VISIBLE_PIXELS * 1e-7, 1e-4
    r_g = G * 10.0 / C_AU_PER_YEAR**2
    place(m, "Black hole", 10.0, (0.0, 0.0, 0.0))
    a, e = 19.0 * r_g, 11.0 / 19.0
    v_apo = np.sqrt(G * 10.0 * (1.0 - e) / (a * (1.0 + e)))  # vis-viva at the apocentre
    probe = place(m, "Body", 1e-9, (30.0 * r_g, 0.0, 0.0), (0.0, v_apo, 0.0), phys_radius=1e-12)
    place(m, "Body", 1e-6, (1.0, 0.0, 0.0), circular(10.0, 1.0))
    m.rename_body(1, "Probe")
    editor_settings["follow"] = "Probe"
    m.editor.selected = 1
    passes, was_away = 0, False
    try:
        for _ in range(150):
            m.step(m.dt)
            away = probe not in m.ids
            passes += away and not was_away
            was_away = away
            assert editor_settings["follow"] == "Probe"
            if away:
                assert m.away_names() == ["Probe"]
                assert m.follow_target() == 0  # the hole stands in for it
            else:
                assert m.index_of(probe) == 1 and m.scene.names == ["Body 1", "Probe", "Body 3"]
                assert m.editor.selected == 1 and m.follow_target() == 1
    finally:
        editor_settings["follow"] = "None"
    assert passes >= 1


def test_names_stay_unique_and_bodies_are_not_found_by_name():
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = "Planets colliding"
    m.load_scene(m.scene_name)
    assert m.scene.names == ["Sun", "Planet A", "Planet B"]
    # Renaming to a name in use gives a numbered name; an empty one is refused.
    assert m.rename_body(0, "Planet A") == "Planet A (2)"
    assert m.rename_body(0, "  ") == "Planet A (2)"
    assert m.rename_body(2, "Planet A") == "Planet A (3)"
    assert len(set(m.scene.names)) == 3
    # Even with names forced equal, the collision must change the planets
    # and not the Sun: the module finds bodies by id. (Before 2026-11-23 it
    # looked the remnant up by name and found the Sun, whose mass went from
    # 1.0 to 0.087.) The planets meet after a quarter orbit, 2 years.
    m.scene.names[:] = ["Planet A", "Planet A", "Planet A"]
    sun = m.ids[0]
    for _ in range(int(2.2 * 240)):
        m.step(m.dt)
    assert m.nb.debris.sum() == N_FRAGMENTS  # they did collide and shatter
    assert m.nb.mass[m.index_of(sun)] >= 1.0  # the Sun can only have gained debris
    assert m.nb.mass.sum() == pytest.approx(1.0 + 2.0 * M_EARTH, rel=1e-12)


def test_renaming_a_hole_does_not_lose_the_bodies_falling_into_it():
    # The torn star's pieces are on exact orbits round the hole from about
    # step 1500. They are kept by the hole's id, so renaming the hole
    # leaves them where they are. (They used to be kept by its name, and
    # were deleted when it changed.)
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = "Star torn by a black hole"
    m.load_scene(m.scene_name)
    total = m.nb.mass.sum()
    for _ in range(2000):
        m.step(m.dt)
        if m.close.count > 100:
            break
    count = m.close.count
    assert count > 100
    m.rename_body(0, "Sagittarius B")
    m.step(m.dt)
    assert m.close.count > count - 50  # a step's worth may be swallowed or leave
    for _ in range(2000):
        m.step(m.dt)
        if m.close.count == 0:
            break
    assert m.nb.mass.sum() == pytest.approx(total, rel=1e-9)


# ------------------------------------------------------------- save and load


def test_scenario_includes_bodies_on_close_passes_and_keeps_their_flags():
    # Saved while pieces of the torn star are on exact orbits round the
    # hole (not in the state): they must be in the file, as gas and debris,
    # and the loaded scene must hold the same number of bodies and mass.
    m = NBodyModule()
    m.discard_edits()
    m.scene_name = "Star torn by a black hole"
    m.load_scene(m.scene_name)
    for _ in range(2000):
        m.step(m.dt)
        if m.close.count > 0 and m.nb.gas.any():
            break
    in_state, away = m.nb.N, m.close.count
    assert away > 0 and in_state > 1
    data = m.scenario()
    assert len(data["bodies"]) == in_state + away
    assert sum(b["gas"] and b["debris"] for b in data["bodies"]) == gas_pieces(m)
    assert sum(b["black_hole"] for b in data["bodies"]) == 1
    assert sum(b["mass"] for b in data["bodies"]) == pytest.approx(
        m.nb.mass.sum() + away / N_TIDAL, rel=1e-12
    )

    other = NBodyModule()
    other.prepare_scenario(data)
    other.load_scene(other.scene_name)
    assert other.nb.N == in_state + away
    assert int(other.nb.gas.sum()) == int(other.nb.debris.sum()) == gas_pieces(m)
    assert list(other.nb.black_hole) == [True] + [False] * (in_state + away - 1)
    # The pieces that were on exact orbits are inside the handover sphere
    # and are handed over again on the first step.
    other.step(other.dt)
    assert other.close.count > 0
    m.discard_edits()


GOOD_BODY = {"pos": [1.0, 0.0, 0.0], "vel": [0.0, 6.28, 0.0], "mass": 3e-6}


@pytest.mark.parametrize(
    "data, message",
    [
        (None, "not an N-body scenario"),
        ([GOOD_BODY], "not an N-body scenario"),
        ({}, "no list of bodies"),
        ({"bodies": None}, "no list of bodies"),
        ({"scene": "Single galaxy", "galaxies": []}, "no list of bodies"),  # a galaxy save
        ({"bodies": [None]}, "body 1 is not a dictionary"),
        ({"bodies": [{"pos": [0, 0, 0], "vel": [0, 0, 0]}]}, "body 1 has no 'mass'"),
        ({"bodies": [GOOD_BODY, {**GOOD_BODY, "mass": "heavy"}]}, "mass of body 2 is not a number"),
        ({"bodies": [{**GOOD_BODY, "mass": float("nan")}]}, "not finite"),
        ({"bodies": [{**GOOD_BODY, "mass": -1.0}]}, "negative"),
        ({"bodies": [{**GOOD_BODY, "pos": [0.0, 0.0]}]}, "position of body 1"),
        ({"bodies": [{**GOOD_BODY, "vel": None}]}, "velocity of body 1"),
        ({"bodies": [{**GOOD_BODY, "black_hole": "yes"}]}, "'black_hole' of body 1"),
        ({"bodies": [GOOD_BODY], "years_per_second": 0}, "years_per_second must be positive"),
        ({"bodies": [GOOD_BODY], "years_per_second": -2.0}, "years_per_second must be positive"),
        ({"bodies": [GOOD_BODY], "years_per_second": "fast"}, "years_per_second is not a number"),
        ({"bodies": [GOOD_BODY], "view_distance": 0.0}, "view_distance must be positive"),
    ],
)
def test_malformed_scenario_is_refused_with_a_message(data, message):
    with pytest.raises(ValueError, match=message):
        validate_scenario(data)
    # ... and leaves the module as it was.
    m = NBodyModule()
    m.discard_edits()
    with pytest.raises(ValueError):
        m.prepare_scenario(data)
    assert m.scene_name == DEFAULT_SCENE
    with pytest.raises(ValueError):
        m.load_scenario(data)
    assert not m.reset_requested


def test_scenario_defaults_and_flags_round_trip():
    # Only pos, vel and mass are required. A missing or null colour is a
    # look, not a reason to refuse the file; equal names are numbered.
    data = {
        "bodies": [
            {**GOOD_BODY, "name": "A", "color": None},
            {**GOOD_BODY, "name": "A", "debris": True, "gas": True, "radius": 1e-5},
            {**GOOD_BODY, "mass": 10.0, "black_hole": True, "radius": 2e-7, "drawn_radius": None},
        ]
    }
    clean = validate_scenario(data)
    assert [b["name"] for b in clean["bodies"]] == ["A", "A (2)", "Body 3"]
    assert clean["bodies"][0]["color"] == [0.8, 0.8, 0.8]
    assert clean["years_per_second"] == 1.0 and clean["view_distance"] == 5.0
    m = NBodyModule()
    m.discard_edits()
    m.prepare_scenario(data)
    m.load_scene(m.scene_name)
    assert list(m.nb.debris) == [False, True, False]
    assert list(m.nb.gas) == [False, True, False]
    assert list(m.nb.black_hole) == [False, False, True]
    again = validate_scenario(m.scenario())
    assert [(b["debris"], b["gas"], b["black_hole"]) for b in again["bodies"]] == [
        (False, False, False),
        (True, True, False),
        (False, False, True),
    ]
    m.discard_edits()


def test_setup_survives_a_malformed_request_from_the_home_screen():
    # An exception leaving setup() ends the app. With a file that cannot be
    # used, setup() must load the scene it would have shown anyway and say
    # what was wrong (in the editor window and the diagnostics).
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as error:
        pytest.skip(f"no OpenGL 4.3 context: {error}")
    try:
        for bad in ({"bodies": None}, {"bodies": [{"pos": 1}]}, "text", {"galaxies": []}):
            scenario_files.request("nbody", bad)
            m = NBodyModule()
            m.discard_edits()
            m.setup(ctx)
            assert m.loaded_scene_name == DEFAULT_SCENE and m.nb.N == 2
            assert m.load_message.startswith("Could not open the saved scenario")
            assert m.load_message in m.diagnostics()
            m.teardown()
        # A good one still opens, with no message.
        scenario_files.request("nbody", {"bodies": [GOOD_BODY]})
        m = NBodyModule()
        m.setup(ctx)
        assert m.loaded_scene_name == CUSTOM and m.nb.N == 1 and m.load_message == ""
        m.teardown()
        m.discard_edits()
    finally:
        scenario_files.take_request("nbody")
        ctx.release()


# -------------------------------------------------------------------- editor


def test_radius_of_a_body_that_stops_being_a_black_hole():
    # Unticking "Black hole": a main-sequence star's radius above the
    # hydrogen-burning limit (0.08 solar masses), a rocky body's below.
    # (It used to be the rocky radius always: 0.003 AU for the Sun, whose
    # radius is 0.00465 AU.)
    assert body_radius_without_horizon(1.0) == pytest.approx(R_SUN)
    assert body_radius_without_horizon(1.0) == star_radius(1.0)
    assert body_radius_without_horizon(M_EARTH) == pytest.approx(R_EARTH)


# ------------------------------------------- tight pairs that cannot be seen
#
# nbody_binary.py: a bound pair whose orbit is below a pixel (round a black
# hole: below VISIBLE_PIXELS, under the hole's dot) and takes less than
# UNRESOLVED_PERIOD steps is carried as one body by the leapfrog and its own
# orbit is advanced analytically. Reference values:
#
# - Kepler: P = 2 pi sqrt(a^3 / (G M)). A 10 solar-mass hole has
#   G M / c^2 = 9.87e-8 AU; an orbit of a = 19 G M / c^2 takes 8.3e-10 yr,
#   5 million orbits per step of 1/240 yr.
# - Peters (1964, Phys. Rev. 136, B1224): a circular pair merges after
#   5 c^5 a^4 / (256 G^3 m1 m2 M) (eq. 5.10); an eccentric one after
#   12/19 c0^4 / beta int_0^e0 x^(29/19) (1 + 121/304 x^2)^(1181/2299)
#   / (1 - x^2)^(3/2) dx, beta = 64/5 G^3 m1 m2 M / c^5, with c0 from
#   a0 = c0 e0^(12/19) / (1 - e0^2) (1 + 121/304 e0^2)^(870/2299) (eq. 5.14).
# - Merger of two equal holes: 4.8% of the mass radiated (postnewtonian.py).
# - The outer planet of the first test: circular at 1 AU round 10 solar
#   masses, P = 1 / sqrt(10) yr = 75.9 steps; in 60 steps a chord of
#   2 sin(pi 60 / 75.9) = 1.226 AU.


def tight_pair(m, m2, a, e, hole=False, hole_mass=10.0):
    """A 10 solar-mass hole and a companion of mass m2 (a compact body, or a second
    hole) at the apocentre of the Kepler orbit (a, e), centre of mass at rest
    at the origin. Returns the two ids."""
    total = hole_mass + m2
    r = a * (1.0 + e)
    v = np.sqrt(G * total * (1.0 - e) / r)
    first = place(
        m, "Black hole", hole_mass, (m2 / total * r, 0.0, 0.0), (0.0, m2 / total * v, 0.0)
    )
    second = place(
        m,
        "Black hole" if hole else "Body",
        m2,
        (-hole_mass / total * r, 0.0, 0.0),
        (0.0, -hole_mass / total * v, 0.0),
        phys_radius=None if hole else 1e-12,  # too compact to be torn
    )
    return first, second


def peters_time(m1, m2, a0, e0):
    """Time to merge by gravitational waves (Peters 1964, eqs. 5.10 and 5.14)."""
    beta = 64.0 / 5.0 * G**3 * m1 * m2 * (m1 + m2) / C_AU_PER_YEAR**5
    if e0 == 0.0:
        return a0**4 / (4.0 * beta)
    c0 = a0 * (1.0 - e0**2) / (e0 ** (12 / 19) * (1.0 + 121 / 304 * e0**2) ** (870 / 2299))
    x = np.linspace(0.0, e0, 200001)[1:]
    y = x ** (29 / 19) * (1.0 + 121 / 304 * x**2) ** (1181 / 2299) / (1.0 - x**2) ** 1.5
    return 12 / 19 * c0**4 / beta * float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def test_unseen_tight_pair_does_not_hold_the_scene():
    # The trigger of 2026-11-23: a compact body between 8 and 30 G M / c^2
    # of a 10 solar-mass hole (3e-6 AU, 1 px = 0.0055 AU), with a planet at
    # 1 AU. Before: 97 of 600 steps cut by the step budget, 15 ms per step,
    # the body handed to its exact orbit and back on every step. Now the
    # pair is followed analytically on every step, no step is cut or
    # slowed, and a step costs 0.4 ms (measured; bound 3 ms, the machine is
    # shared).
    m = custom_module()
    r_g = G * 10.0 / C_AU_PER_YEAR**2
    hole = place(m, "Black hole", 10.0, (0.0, 0.0, 0.0))
    a, e = 19.0 * r_g, 11.0 / 19.0
    v_apo = np.sqrt(G * 10.0 * (1.0 - e) / (a * (1.0 + e)))
    probe = place(m, "Body", 1e-9, (30.0 * r_g, 0.0, 0.0), (0.0, v_apo, 0.0), phys_radius=1e-12)
    planet = place(m, "Body", 1e-6, (1.0, 0.0, 0.0), circular(10.0, 1.0))
    m.step(m.dt)  # (compiles on the first call)
    start = m.nb.pos[2] - m.nb.pos[0]
    times, elapsed = [], m.elapsed
    for _ in range(60):
        t = time.perf_counter()
        m.step(m.dt)
        times.append(time.perf_counter() - t)
        assert not m.slowed and m.slow_motion == 1.0
        assert m.held_pairs().tolist() == [[0, 1]]
        assert m.close.count == 0 and m.ids == [hole, probe, planet]
    assert m.elapsed - elapsed == pytest.approx(60 / 240, rel=1e-12)
    assert np.median(times) < 3e-3
    # The rest of the scene moves at its true pace.
    assert np.linalg.norm(m.nb.pos[2] - m.nb.pos[0] - start) == pytest.approx(1.226, rel=0.01)
    # The probe is still on its orbit: between 7 (the 1PN periapsis of this
    # orbit; Newtonian 8) and 30 G M / c^2 from the hole.
    assert 6.9 * r_g < np.linalg.norm(m.nb.pos[1] - m.nb.pos[0]) < 30.01 * r_g
    # Zoomed in until the orbit can be seen, it is integrated directly again.
    m.visible_scale = VISIBLE_PIXELS * 1e-7
    m.step(m.dt)
    assert len(m.held_pairs()) == 0


def binary_and_planet(gravity, planet_distance=2.0):
    """Stars of 1 and 0.1 solar masses on an orbit of a = 0.02 AU, e = 0.3 (period
    2.7e-3 yr = 0.65 steps), tilted by 0.4 rad, and a planet on a circular
    orbit round them. Seen from 30 AU (1 px = 0.033 AU: the pair, 0.026 AU at its
    widest, is below a pixel)."""
    m = custom_module()
    m.gravity = gravity
    m.integrator = make_integrator(gravity)
    m.visible_scale, m.view_zone = VISIBLE_PIXELS * 6.0 * AU_PER_PIXEL, 30.0
    m1, m2, a, e, tilt = 1.0, 0.1, 0.02, 0.3, 0.4
    total = m1 + m2
    r = a * (1.0 + e)
    v = np.sqrt(G * total * (1.0 - e) / r) * np.array([0.0, np.cos(tilt), np.sin(tilt)])
    place(m, "Star", m1, (m2 / total * r, 0.0, 0.0), m2 / total * v)
    place(m, "Star", m2, (-m1 / total * r, 0.0, 0.0), -m1 / total * v)
    if planet_distance:
        place(m, "Body", 3e-6, (planet_distance, 0.0, 0.0), circular(total, planet_distance))
    return m


@pytest.mark.parametrize("gravity", [RELATIVITY, NEWTON])
def test_held_pair_matches_direct_integration(gravity):
    # The same system integrated directly (the leapfrog with its substeps,
    # 700 per step, no view so nothing is held) for half a year: 185 orbits
    # of the pair, a third of the planet's. The pair's centre of mass must
    # agree to 1e-6 AU (measured 7e-8) and the planet to 2e-4 AU of the
    # 2.2 AU it moves (measured 4e-5: the pair's quadrupole pull, which is
    # left out, nbody_binary.QUADRUPOLE). The pair itself stays the same
    # orbit: separation within its range of 0.014 to 0.026 AU.
    m = binary_and_planet(gravity)
    ref = deepcopy(m.nb)
    direct = make_integrator(gravity)
    for _ in range(120):
        m.step(m.dt)
        assert len(m.held_pairs()) == 1 and not m.slowed
        direct.advance(ref, m.sim_dt)
    assert direct.last_substeps > 300

    def centre(state):
        return state.mass[:2] @ state.pos[:2] / state.mass[:2].sum()

    assert np.linalg.norm(centre(m.nb) - centre(ref)) < 1e-6
    assert np.linalg.norm(m.nb.pos[2] - ref.pos[2]) < 2e-4
    assert 0.0139 < separation(m.nb) < 0.0261


def test_held_pair_conserves_energy_and_momenta():
    # Ten years (2400 steps, 3700 orbits of the pair), 1PN gravity: the 1PN
    # energy of the whole system (m.energy) within 1e-9 (measured 1e-11:
    # the pair's own energy is kept to rounding and dominates the total),
    # the total momentum to rounding, and the Newtonian angular momentum
    # within 1e-5 (measured 1.4e-6; it is only conserved to 1PN order,
    # v^2 / c^2 = 1e-6 here).
    m = binary_and_planet(RELATIVITY)
    e0, p0 = m.energy(), total_momentum(m.nb)
    l0 = total_angular_momentum(m.nb)
    worst_e = worst_l = 0.0
    for _ in range(2400):
        m.step(m.dt)
        worst_e = max(worst_e, abs(m.energy() / e0 - 1.0))
        worst_l = max(
            worst_l, np.linalg.norm(total_angular_momentum(m.nb) - l0) / np.linalg.norm(l0)
        )
    assert len(m.held_pairs()) == 1
    assert worst_e < 1e-9
    assert worst_l < 1e-5
    assert np.linalg.norm(total_momentum(m.nb) - p0) < 1e-10
    # With Newtonian gravity the pair's orbit is the exact Kepler orbit, and
    # the angular momentum is conserved too (measured 3e-13).
    m = binary_and_planet(NEWTON)
    e0, l0 = m.energy(), total_angular_momentum(m.nb)
    for _ in range(600):
        m.step(m.dt)
    assert abs(m.energy() / e0 - 1.0) < 1e-9
    assert np.linalg.norm(total_angular_momentum(m.nb) - l0) / np.linalg.norm(l0) < 1e-10


def test_energy_of_a_decaying_pair_changes_only_by_what_the_waves_take():
    # The trigger's pair radiates: its own 1PN energy (reduced mass times
    # nbody_binary.invariants' eps) falls. The energy of the whole system
    # must change by exactly that (measured 1e-10 of the total).
    m = custom_module()
    r_g = G * 10.0 / C_AU_PER_YEAR**2
    tight_pair(m, 1e-9, 19.0 * r_g, 11.0 / 19.0)
    place(m, "Body", 1e-6, (1.0, 0.0, 0.0), circular(10.0, 1.0))
    nb = m.nb
    total = nb.mass[0] + nb.mass[1]
    mu = nb.mass[0] * nb.mass[1] / total

    def pair_energy():
        nb = m.nb
        return (
            mu
            * invariants(
                nb.pos[0] - nb.pos[1],
                nb.vel[0] - nb.vel[1],
                G * total,
                mu / total,
                C_AU_PER_YEAR**-2,
            )[0]
        )

    e0, pair0 = m.energy(), pair_energy()
    for _ in range(480):
        m.step(m.dt)
        assert len(m.held_pairs()) == 1
    radiated = pair_energy() - pair0
    assert radiated < -0.01 * abs(pair0)  # it did decay (measured 7%)
    assert abs(m.energy() - e0 - radiated) < 1e-8 * abs(e0)


@pytest.mark.parametrize(
    "m2, a_rg, e, hole, final_mass",
    [
        # Peters: 0.2466 yr (a = 2005.5 G M / c^2, see below) and 10.07 yr.
        (10.0, 2000.0, 0.0, True, 20.0 * (1.0 - 0.048)),
        (1.4, 4000.0, 0.6, False, 11.4),
    ],
)
def test_unseen_pair_spirals_in_and_merges_when_peters_says(m2, a_rg, e, hole, final_mass):
    # A second hole of 10 solar masses 2000 G M / c^2 from the first
    # (2e-4 AU, 0.04 px; before: every step cut, the scene x3800 slower),
    # and a neutron star on an eccentric orbit. They must merge at Peters'
    # time, within 1% (the 1PN terms change the orbit's size and shape by
    # G M / (a c^2) = 5e-4) plus one step, with the scene at its own pace
    # throughout; the holes' merger radiates 4.8% of the mass.
    m = custom_module()
    r_g = G * 10.0 / C_AU_PER_YEAR**2
    tight_pair(m, m2, a_rg * r_g, e, hole)
    a = a_rg * r_g
    if e == 0.0:
        # Started with the Newtonian circular speed, the 1PN orbit is not
        # quite circular: it reaches from 2000 to 2011 G M / c^2 (measured).
        a *= 2005.5 / 2000.0
    expected = peters_time(10.0, m2, a, e)
    flashes = 0
    while m.nb.N == 2 and m.elapsed < 1.5 * expected:
        m.step(m.dt)
        assert not m.slowed and m.slow_motion == 1.0
        assert m.nb.N == 1 or len(m.held_pairs()) == 1
        flashes = max(flashes, len(m.flashes))
    assert m.nb.N == 1 and m.nb.black_hole[0]
    assert m.elapsed == pytest.approx(expected, rel=0.01, abs=1.01 / 240)
    assert m.nb.mass[0] == pytest.approx(final_mass, rel=1e-12)
    assert flashes == (1 if hole else 0)  # the merger of two holes is shown
    np.testing.assert_allclose(total_momentum(m.nb), 0.0, atol=1e-9)


def test_perturbed_pair_goes_back_to_direct_integration():
    # A star of 0.5 solar masses flies past the pair of stars at 0.34 AU
    # with 30 AU/yr. The tide it raises, 2 m / M (r_max / d)^3, reaches
    # nbody_binary.TIDE = 1e-6 at d = r_max (2 m / (M TIDE))^(1/3) =
    # 0.026 * 97 = 2.5 AU: inside that (less the 0.125 AU it moves per
    # step) the pair is integrated directly, outside it analytically, on
    # the way in and on the way out alike. Taking and releasing the pair
    # does not change the state; the energy changes only by the direct
    # integration's own error for the pair (measured 1.3e-4, bounded).
    m = binary_and_planet(RELATIVITY, planet_distance=None)
    place(m, "Star", 0.5, (-6.0, 0.4, 0.0), (30.0, 0.0, 0.0))
    nb = m.nb
    nb.pos -= nb.mass @ nb.pos / nb.mass.sum()
    nb.vel -= nb.mass @ nb.vel / nb.mass.sum()
    m.state_edited()

    def distance():
        nb = m.nb
        return np.linalg.norm(nb.pos[2] - nb.mass[:2] @ nb.pos[:2] / nb.mass[:2].sum())

    # The switch itself leaves the state as it is.
    before = (m.nb.pos.copy(), m.nb.vel.copy())
    assert len(m._hold(m.nb)) == 1
    assert np.array_equal(m.nb.pos, before[0]) and np.array_equal(m.nb.vel, before[1])

    e0 = m.energy()
    released = retaken = None
    closest, worst = np.inf, 0.0
    for _ in range(400):
        held = len(m.held_pairs()) if m.step_index else 1
        d = distance()
        m.step(m.dt)
        now = len(m.held_pairs())
        if held and not now:
            released = d
        if now and not held:
            retaken = distance()
            break
        closest = min(closest, distance())
        worst = max(worst, abs(m.energy() / e0 - 1.0))
    assert released == pytest.approx(2.6, abs=0.2)
    assert retaken == pytest.approx(2.6, abs=0.3)
    assert closest == pytest.approx(0.34, abs=0.02)
    assert m.nb.N == 3
    assert worst < 5e-4
    for _ in range(20):
        m.step(m.dt)
        assert len(m.held_pairs()) == 1 and not m.slowed


def test_members_of_a_held_pair_do_not_touch_each_other():
    # Two bodies of radius 0.1 at distance 0.1 touch, unless they are each
    # other's partner; a third body touching one of them still counts.
    pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]])
    v = np.zeros((3, 3))
    radius = np.array([0.1, 0.1, 0.1])
    assert first_contact(pos, v, radius, 0.01) == (0, 1)
    partner = np.array([1, 0, -1])
    assert first_contact(pos, v, radius, 0.01, partner=partner) is None
    pos[2] = [0.25, 0.0, 0.0]
    assert first_contact(pos, v, radius, 0.01, partner=partner) == (1, 2)


def test_trail_stays_behind_when_its_body_is_gone():
    # nbody_trails.py: a body that leaves the list (swallowed, torn, handed
    # to its exact orbit) leaves its trail, which goes on fading from its
    # old end for as many steps as it had samples; before, it vanished.
    trails = TrailRenderer(2, 100)
    for k in range(10):
        trails.record(np.array([[k, 0.0, 0.0], [0.0, k, 0.0]]))
    trails.remove_body(0, (1.0, 0.5, 0.0), key=7)
    assert trails.history.shape[1] == 1 and trails.valid.tolist() == [10]
    assert len(trails.left) == 1 and trails.left[0]["valid"] == 10
    assert trails.left[0]["color"] == (1.0, 0.5, 0.0)
    np.testing.assert_array_equal(trails.left[0]["history"][:10, 0], np.arange(10.0))
    for _ in range(9):
        trails.record(np.zeros((1, 3)))
    assert len(trails.left) == 1
    trails.record(np.zeros((1, 3)))
    assert trails.left == []  # all of it has scrolled out
    # A body removed without a colour (deleted by the user), or brought back
    # with its trail, leaves none.
    trails.remove_body(0)
    assert trails.left == []

    # In the module: a planet swallowed by a hole leaves its trail.
    m = custom_module()
    place(m, "Black hole", 1e6, (0.0, 0.0, 0.0))
    place(m, "Body", 1e-6, (0.5, 0.0, 0.0), (-40.0, 0.0, 0.0), phys_radius=1e-12)
    for _ in range(400):
        m.step(m.dt)
        if m.nb.N == 1 and m.close.count == 0:
            break
    assert m.nb.N == 1
    assert len(m.trails.left) == 1 and m.trails.left[0]["valid"] >= 2
