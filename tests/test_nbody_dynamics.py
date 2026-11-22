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

from functools import partial

import numpy as np
import pytest

from physengine.modules.nbody_collisions import (
    N_FRAGMENTS,
    N_TIDAL,
    disruption_energy,
    largest_remnant_fraction,
)
from physengine.modules.nbody_editor import (
    HILL_FRACTION,
    circular_orbit_velocity,
    hill_limited_position,
)
from physengine.modules.nbody_scenes import CUSTOM, R_SUN, SCENES, _kepler_pair, make_state
from physengine.modules.nbodymodule import NBodyModule
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
from physengine.modules.postnewtonian import schwarzschild_radius

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
    assert 0.4 < 1.0 - left / N_TIDAL < 0.9
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
