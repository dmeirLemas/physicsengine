"""Tests of the two-galaxy encounter setup (modules/galaxy_encounters.py) and
of building scenes from edited setups (modules/galaxy_editor.py).

Model units: G = 1, kpc, 1e10 M_sun.

Reference values and their sources (Murray & Dermott 1999, Solar System
Dynamics, ch. 2, for the Kepler orbit):

- Kepler orbit with total mass M, pericentre r_p, eccentricity e:
  semi-latus rectum p = r_p (1 + e); specific energy
  v^2/2 - G M / r = G M (e - 1) / (2 r_p) (that is -G M / (2 a) with
  a = r_p / (1 - e), and 0 for the parabola); specific angular momentum
  h = sqrt(G M p); eccentricity vector (v x h) / (G M) - r / |r|, of
  length e, pointing at the pericentre; apocentre r_p (1 + e) / (1 - e).
- Time from true anomaly f to pericentre on a parabola (Barker's equation):
  t = sqrt(2 r_p^3 / (G M)) (D + D^3 / 3), D = tan(f / 2). For M = 112,
  r_p = 10, r0 = 100: cos f = -0.8, D = 3, t = 50.71 time units = 239 Myr.
- Time to pericentre on an ellipse (Kepler's equation): r = a (1 - e cos E),
  t = (E - e sin E) / n, n = sqrt(G M / a^3). For M = 112, r_p = 10,
  e = 0.8, r0 = 70: a = 50, cos E = -0.5, E = 2 pi / 3,
  t = (2 pi / 3 - 0.8 sqrt(3) / 2) / sqrt(112 / 125000) = 46.82.
- Toomre & Toomre (1972) angles of a disc with spin axis s, for a companion
  orbit with angular momentum along +z and pericentre on +x: inclination
  i = arccos(s . z); ascending node N = s x z (the standard node vector with
  the disc as reference plane), where the companion crosses to the side s
  points to; argument of pericentre omega = angle from N to +x, measured
  about +z.
- A rotation by 60 degrees about x turns the +z axis into (0, -sin 60, cos 60).
- Orbit of two extended galaxies (modules/galaxy_orbit.py):
  - A test body at distance d from the centre of a spherical mass
    distribution is pulled by the mass inside d only (Newton's shell
    theorem), so the centres of two galaxies approach each other with
    g = G (M1(<d) + M2(<d)) / d^2. Checked here against the summed pull of
    the sampled particles of each galaxy on the other's centre. Beyond both
    galaxies g = G (m1 + m2) / d^2.
  - Motion in a central potential W with (reduced) mass mu:
    E = mu v^2 / 2 + W(r) and L = mu |r x v| are conserved (e.g. Goldstein,
    Classical Mechanics, ch. 3). Bound orbits: the radial speed vanishes at
    r_p and at r_a = r_p (1 + e) / (1 - e), from e = (r_a - r_p) / (r_a + r_p)
    (Binney & Tremaine 2008, sec. 3.1). Unbound: E = (e - 1) / 2 |W(r_p)|,
    zero for the parabolic orbit.
  - For point masses all of this is the Kepler orbit above.
"""

import numpy as np
import pytest

from physengine.modules.galaxy_components import DISC, GalaxyModel
from physengine.modules.galaxy_editor import clear_edits, save_edits, scene_for
from physengine.modules.galaxy_encounters import (
    SCENES,
    SINGLE,
    DiscAngles,
    Encounter,
    apocentre,
    disc_rotation,
    kepler_orbit,
    place_pair,
    rotation_x,
    time_to_pericentre,
)
from physengine.modules.galaxy_orbit import (
    GalaxyPair,
    PointMasses,
    encounter_orbit,
    orbit_apocentre,
    orbit_constants,
)
from physengine.modules.galaxymodule import GalaxyModule, make_galaxy
from physengine.modules.nbodystate import Integrator, NBodyState, acc

G = 1.0
ENCOUNTER_NAMES = [name for name in SCENES if name != SINGLE]


# ------------------------------------------------------------------ the orbit


@pytest.mark.parametrize(
    "M, r_p, e, r0",
    [
        (112.0, 10.0, 1.0, 100.0),
        (2.0, 1.0, 1.0, 50.0),
        (10.0, 3.0, 1.0, 3.0),
        (112.0, 10.0, 0.8, 70.0),
        (112.0, 10.0, 0.5, 25.0),
        (112.0, 10.0, 1.5, 200.0),
        (5.0, 2.0, 0.0, 2.0),
    ],
)
def test_kepler_orbit_invariants(M, r_p, e, r0):
    pos, vel = kepler_orbit(M, r_p, e, r0, G)
    assert np.linalg.norm(pos) == pytest.approx(r0)
    energy = 0.5 * vel @ vel - G * M / r0
    assert energy == pytest.approx(G * M * (e - 1.0) / (2.0 * r_p), abs=1e-12 * G * M / r0)
    h = np.cross(pos, vel)
    assert np.linalg.norm(h) == pytest.approx(np.sqrt(G * M * r_p * (1.0 + e)))
    assert h[2] > 0  # angular momentum along +z
    e_vec = np.cross(vel, h) / (G * M) - pos / r0
    assert np.linalg.norm(e_vec) == pytest.approx(e, abs=1e-12)
    if e > 0:
        np.testing.assert_allclose(e_vec / e, [1.0, 0.0, 0.0], atol=1e-12)  # pericentre on +x
    if r_p < r0 < apocentre(r_p, e):
        assert pos @ vel < 0  # still approaching


def test_start_at_apocentre():
    pos, vel = kepler_orbit(112.0, 10.0, 0.8, 90.0, G)
    np.testing.assert_allclose(pos, [-90.0, 0.0, 0.0], atol=1e-9)
    assert abs(pos @ vel) < 1e-9 * 90.0 * np.linalg.norm(vel)


def test_orbit_range_is_checked():
    with pytest.raises(ValueError):
        kepler_orbit(112.0, 10.0, 1.0, 5.0)  # inside the pericentre
    with pytest.raises(ValueError):
        kepler_orbit(112.0, 10.0, 0.8, 95.0)  # beyond the apocentre (90)
    with pytest.raises(ValueError):
        kepler_orbit(112.0, 0.0, 0.8, 50.0)  # radial orbits are parabolic only


def test_radial_orbit_is_the_limit_of_small_pericentre():
    pos, vel = kepler_orbit(112.0, 0.0, 1.0, 100.0, G)
    np.testing.assert_allclose(pos, [-100.0, 0.0, 0.0])
    np.testing.assert_allclose(vel, [np.sqrt(2 * 112.0 / 100.0), 0.0, 0.0])
    p_small, v_small = kepler_orbit(112.0, 1e-8, 1.0, 100.0, G)
    np.testing.assert_allclose(p_small, pos, atol=1e-2)
    np.testing.assert_allclose(v_small, vel, atol=1e-4)


def test_barker_time_for_the_encounter_orbit():
    # cos f = 2 r_p / r0 - 1 = -0.8, so D = tan(f/2) = 3 and t = sqrt(2000/112) * 12.
    assert time_to_pericentre(112.0, 10.0, 100.0) == pytest.approx(np.sqrt(2000.0 / 112.0) * 12.0)


@pytest.mark.parametrize(
    "e, r0, t_peri",
    [
        (1.0, 100.0, np.sqrt(2000.0 / 112.0) * 12.0),
        (0.8, 70.0, (2 * np.pi / 3 - 0.8 * np.sqrt(3) / 2) / np.sqrt(112.0 / 125000.0)),
    ],
)
def test_point_masses_reach_pericentre_on_time(e, r0, t_peri):
    # Integrate the orbit itself with module 1's adaptive leapfrog: the
    # closest approach must be r_p, on the +x side, at the predicted time.
    M, r_p = 112.0, 10.0
    rel_pos, rel_vel = kepler_orbit(M, r_p, e, r0, G)
    s = NBodyState(2)
    s.mass[:] = [M / 2, M / 2]
    s.pos[:] = [-rel_pos / 2, rel_pos / 2]
    s.vel[:] = [-rel_vel / 2, rel_vel / 2]
    integ = Integrator(lambda p, m: acc(p, m, G, 0.0))

    dt, t, best = 0.005, 0.0, (np.inf, 0.0, None)
    while t < t_peri + 20.0:
        integ.step(s, dt, 1)
        t += dt
        rel = s.pos[1] - s.pos[0]
        d = np.linalg.norm(rel)
        if d < best[0]:
            best = (d, t, rel)
    assert best[0] == pytest.approx(r_p, rel=1e-3)
    assert best[1] == pytest.approx(t_peri, abs=2 * dt)
    np.testing.assert_allclose(best[2] / best[0], [1.0, 0.0, 0.0], atol=1e-2)


# ----------------------------------------------------------- disc orientation


@pytest.mark.parametrize(
    "i, omega", [(0.0, 0.0), (60.0, 0.0), (45.0, 90.0), (120.0, 250.0), (90.0, 300.0)]
)
def test_toomre_angles(i, omega):
    R = disc_rotation(i, omega)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0)
    s = R @ [0.0, 0.0, 1.0]  # the disc's spin axis in the orbit frame
    z = np.array([0.0, 0.0, 1.0])
    assert np.degrees(np.arccos(np.clip(s @ z, -1, 1))) == pytest.approx(i, abs=1e-9)
    if 0.0 < i < 180.0:
        N = np.cross(s, z)
        # The companion, moving counter-clockwise about +z, crosses the disc
        # plane at N going to the side the spin points to: the ascending node.
        assert np.cross(z, N) @ s > 0
        measured = np.degrees(np.arctan2(np.cross(N, [1.0, 0.0, 0.0]) @ z, N[0]))
        assert (measured - omega + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-9)


def test_rotation_x():
    expected = [0, -np.sin(np.radians(60)), 0.5]
    np.testing.assert_allclose(rotation_x(np.radians(60)) @ [0, 0, 1], expected, atol=1e-15)


# ---------------------------------------------------------------- placing


def _blob(rng, n, m):
    p = rng.normal(0, 1, (n, 3))
    p -= p.mean(axis=0)
    return p, np.zeros((n, 3)), np.full(n, m / n), np.full(n, DISC)


@pytest.mark.parametrize("e, r0", [(1.0, 100.0), (0.8, 70.0), (1.3, 150.0)])
def test_place_pair_puts_galaxies_on_the_orbit(e, r0):
    rng = np.random.default_rng(0)
    g1, g2 = _blob(rng, 200, 30.0), _blob(rng, 300, 60.0)
    enc = Encounter(pericentre=10.0, eccentricity=e, start_distance=r0)
    pos, vel, mass, kind, index = place_pair(g1, g2, enc, PointMasses(30.0, 60.0, G))
    np.testing.assert_allclose(mass @ pos, 0.0, atol=1e-10)
    np.testing.assert_allclose(mass @ vel, 0.0, atol=1e-10)
    c1 = mass[index == 0] @ pos[index == 0] / 30.0
    c2 = mass[index == 1] @ pos[index == 1] / 60.0
    v1 = mass[index == 0] @ vel[index == 0] / 30.0
    v2 = mass[index == 1] @ vel[index == 1] / 60.0
    rel_pos, rel_vel = kepler_orbit(90.0, 10.0, e, r0, G)
    # (the orbit is integrated numerically; the Kepler formula is exact)
    np.testing.assert_allclose(c2 - c1, rel_pos, atol=2e-3)
    np.testing.assert_allclose(v2 - v1, rel_vel, atol=1e-4)


def test_both_galaxies_measure_their_angles_against_the_companion_orbit():
    # The same angles for both discs must give each the same orientation
    # relative to the orbit of its companion around it. Galaxy 2 sees the
    # companion with pericentre on -x, so its frame is turned 180 deg about z.
    angles = DiscAngles(50.0, 70.0)
    enc = Encounter(discs=[angles, angles])
    # Probe with a single "particle" on the disc's spin axis.
    probe = (np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.array([1.0]), np.array([DISC]))
    pair = PointMasses(1.0, 1.0, G)
    pos, *_ = place_pair(probe, probe, enc, pair)
    rel_pos, _, _ = encounter_orbit(pair, 10.0, 1.0, 100.0)
    s1 = pos[0] - (-0.5 * rel_pos)
    s2 = pos[1] - (0.5 * rel_pos)
    expected = disc_rotation(50.0, 70.0) @ [0.0, 0.0, 1.0]
    np.testing.assert_allclose(s1, expected, atol=1e-12)
    np.testing.assert_allclose(s2, expected * [-1.0, -1.0, 1.0], atol=1e-12)


# ------------------------------------------- orbit of two extended galaxies


@pytest.fixture(scope="module")
def galaxy_pair():
    return GalaxyPair(GalaxyModel(), GalaxyModel(), G)


def test_point_mass_limit_is_the_kepler_orbit():
    pm = PointMasses(40.0, 72.0, G)
    for e, r0, t_ref in [
        (1.0, 100.0, np.sqrt(2000.0 / 112.0) * 12.0),
        (0.8, 70.0, (2 * np.pi / 3 - 0.8 * np.sqrt(3) / 2) / np.sqrt(112.0 / 125000.0)),
        (1.5, 200.0, None),
    ]:
        pos, vel, t = encounter_orbit(pm, 10.0, e, r0)
        ref_pos, ref_vel = kepler_orbit(112.0, 10.0, e, r0, G)
        np.testing.assert_allclose(pos, ref_pos, atol=2e-3)
        np.testing.assert_allclose(vel, ref_vel, atol=1e-4)
        if t_ref is not None:
            assert t == pytest.approx(t_ref, rel=1e-4)


def test_far_apart_the_galaxies_attract_like_point_masses(galaxy_pair):
    for d in (700.0, 2000.0, 10000.0):  # the haloes end at 300 kpc
        assert float(galaxy_pair.dW(d)) == pytest.approx(G * 112.0 / d**2, rel=1e-6)
        assert float(galaxy_pair.W(d)) == pytest.approx(-G * 112.0 / d, rel=1e-3)


def test_attraction_of_the_centres_matches_the_particles():
    # Pull of galaxy 2's sampled particles on galaxy 1's centre, plus that of
    # galaxy 1's on galaxy 2's centre, at separation d, averaged over the six
    # directions +-x, +-y, +-z. Measured ratio to the model: 1.008, 1.007,
    # 0.991 at 30, 60, 150 kpc (single directions scatter by up to 9%:
    # particle noise). At 10 kpc the flat disc, taken as spherical in the
    # model, pulls 30% harder in its own plane and 9% less along its axis
    # (compare Binney & Tremaine 2008, fig. 2.17): mean 1.17. The point-mass
    # value G (m1 + m2) / d^2 is far outside all of these.
    lighter = GalaxyModel()
    lighter.halo.mass, lighter.halo.a = 20.0, 20.0
    models = (GalaxyModel(), lighter)
    pair = GalaxyPair(*models, G)
    samples = [make_galaxy(mod, np.random.default_rng(3 + k), 0) for k, mod in enumerate(models)]
    total = models[0].total_mass() + models[1].total_mass()
    for d, tolerance in ((10.0, 0.25), (30.0, 0.03), (60.0, 0.03), (150.0, 0.03)):
        pulls = []
        for axis in np.vstack([np.eye(3), -np.eye(3)]):
            g = 0.0
            for p, _, m, _ in samples:
                rel = p - d * axis  # the particles as seen from a point at distance d
                g += -G * (m / np.linalg.norm(rel, axis=1) ** 3) @ rel @ axis
            pulls.append(g)
        assert np.mean(pulls) == pytest.approx(float(pair.dW(d)), rel=tolerance), d
        if d <= 60.0:
            assert np.mean(pulls) < 0.75 * G * total / d**2


@pytest.mark.parametrize(
    "r_p, e, r0",
    [
        (10.0, 1.0, 100.0),
        (10.0, 0.8, 70.0),
        (25.0, 0.5, 40.0),
        (10.0, 1.5, 100.0),
        (10.0, 0.0, 10.0),
    ],
)
def test_orbit_has_its_energy_and_angular_momentum(galaxy_pair, r_p, e, r0):
    pair = galaxy_pair
    pos, vel, _ = encounter_orbit(pair, r_p, e, r0)
    E, L = orbit_constants(pair, r_p, e)
    assert np.linalg.norm(pos) == pytest.approx(r0)
    energy = 0.5 * pair.mu * vel @ vel + float(pair.W(r0))
    assert energy == pytest.approx(E, abs=1e-9 * abs(E) + 1e-9)
    assert pair.mu * np.cross(pos, vel)[2] == pytest.approx(L)
    if r0 > r_p:
        assert pos @ vel <= 1e-9  # approaching

    # The radial speed vanishes at the pericentre, and for bound orbits at the apocentre.
    def radial(r):
        return E - float(pair.W(r)) - L**2 / (2 * pair.mu * r**2)

    scale = abs(float(pair.W(r_p)))
    assert radial(r_p) == pytest.approx(0.0, abs=1e-9 * scale)
    if e >= 1.0:
        assert E == pytest.approx(0.5 * (e - 1.0) * scale)
        assert radial(1.5 * r_p) > 0 and radial(500.0) > 0
    elif e > 0.0:
        r_a = orbit_apocentre(r_p, e)
        assert r_a == pytest.approx(r_p * (1 + e) / (1 - e))
        assert radial(r_a) == pytest.approx(0.0, abs=1e-9 * scale)
        assert radial(0.5 * (r_p + r_a)) > 0  # allowed in between


@pytest.mark.parametrize("r_p, e, r0", [(10.0, 1.0, 100.0), (10.0, 0.8, 70.0), (20.0, 1.3, 150.0)])
def test_extended_orbit_reaches_its_pericentre_on_time(galaxy_pair, r_p, e, r0):
    # Independent check of the set-up: follow the one-body problem forwards
    # with a plain leapfrog; closest approach must be r_p, on +x, at t_peri.
    pair = galaxy_pair
    pos, vel, t_peri = encounter_orbit(pair, r_p, e, r0)
    x, v = pos[:2].copy(), vel[:2].copy()

    def a(x):
        r = np.hypot(*x)
        return -float(pair.dW(r)) / (pair.mu * r) * x

    dt, t, best = 0.002, 0.0, (np.inf, 0.0, None)
    acc_now = a(x)
    while t < t_peri + 5.0:
        v += 0.5 * dt * acc_now
        x += dt * v
        acc_now = a(x)
        v += 0.5 * dt * acc_now
        t += dt
        r = np.hypot(*x)
        if r < best[0]:
            best = (r, t, x / r)
    assert best[0] == pytest.approx(r_p, rel=2e-3)
    assert best[1] == pytest.approx(t_peri, abs=0.05)
    np.testing.assert_allclose(best[2], [1.0, 0.0], atol=2e-2)


def test_extended_galaxies_fall_slower_than_point_masses(galaxy_pair):
    # Overlapping haloes attract less: same pericentre, lower speed, later passage.
    _, vel, t = encounter_orbit(galaxy_pair, 10.0, 1.0, 100.0)
    _, vel_pm, t_pm = encounter_orbit(PointMasses(56.0, 56.0, G), 10.0, 1.0, 100.0)
    assert np.linalg.norm(vel) < np.linalg.norm(vel_pm)
    assert t > t_pm


def test_head_on_orbit(galaxy_pair):
    pos, vel, t = encounter_orbit(galaxy_pair, 0.0, 1.0, 100.0)
    np.testing.assert_allclose(pos, [0.0, -100.0, 0.0])
    # Parabolic: kinetic energy = -W at the start.
    speed = np.sqrt(-2.0 * float(galaxy_pair.W(100.0)) / galaxy_pair.mu)
    np.testing.assert_allclose(vel, [0.0, speed, 0.0])
    # Grazing orbits come in from nearly the same direction: the head-on
    # start is their limit (measured: 2.9 degrees off at r_p = 0.05 kpc,
    # 0.9 at 0.01; slow because the bulges' central cusps still deflect).
    near, _, t_near = encounter_orbit(galaxy_pair, 0.01, 1.0, 100.0)
    closer, _, _ = encounter_orbit(galaxy_pair, 0.05, 1.0, 100.0)
    np.testing.assert_allclose(near, pos, atol=2.0)
    assert np.linalg.norm(near - pos) < 0.5 * np.linalg.norm(closer - pos)
    assert t == pytest.approx(t_near, rel=1e-3)
    with pytest.raises(ValueError):
        encounter_orbit(galaxy_pair, 0.0, 0.8, 100.0)  # head-on orbits are unbound only
    with pytest.raises(ValueError):
        encounter_orbit(PointMasses(1.0, 1.0), 0.0, 1.0, 100.0)
    with pytest.raises(ValueError):
        encounter_orbit(galaxy_pair, 10.0, 0.8, 95.0)  # beyond the apocentre (90)


# ----------------------------------------------------------------- scenes


def disc_spin(m, g):
    sel = (m.galaxy == g) & (m.kind == DISC) & (m.nb.mass > 0)
    p = m.nb.pos[sel] - m.nb.pos[sel].mean(axis=0)
    v = m.nb.vel[sel] - m.nb.vel[sel].mean(axis=0)
    L = np.cross(p, v).sum(axis=0)
    return L / np.linalg.norm(L)


@pytest.fixture(scope="module")
def scenes():
    clear_edits()
    built = {}
    for name in ENCOUNTER_NAMES:
        m = GalaxyModule()
        m.scene_name = name
        m.tracers = "12000"
        m.load()
        built[name] = m
    return built


def test_every_encounter_starts_at_its_distance_at_rest(scenes):
    for name, m in scenes.items():
        nb = m.nb
        assert nb.N == 42000, name
        start = SCENES[name].encounter.start_distance
        assert m._separation_line().strip() == f"Separation: {start:.1f} kpc", name
        np.testing.assert_allclose(nb.mass @ nb.vel, 0.0, atol=1e-10)
        np.testing.assert_allclose(nb.mass @ nb.pos, 0.0, atol=1e-9)


@pytest.mark.parametrize(
    "name, spin0, spin1",
    [
        ("Prograde", (0, 0, 1), (0, 0, 1)),
        ("Inclined", (0, 0, 1), (0, -np.sin(np.radians(60)), np.cos(np.radians(60)))),
        ("Retrograde", (0, 0, 1), (0, 0, -1)),
        ("Head-on", (0, 1, 0), (0, -1, 0)),  # along the collision axis, y
        ("Bound prograde", (0, 0, 1), (0, 0, 1)),
    ],
)
def test_disc_orientations(scenes, name, spin0, spin1):
    m = scenes[name]
    # The measured spin of 4000 disc stars is within a degree or two of the axis.
    assert np.degrees(np.arccos(np.clip(disc_spin(m, 0) @ spin0, -1, 1))) < 3
    assert np.degrees(np.arccos(np.clip(disc_spin(m, 1) @ spin1, -1, 1))) < 3


def test_single_galaxy_scene():
    clear_edits()
    m = GalaxyModule()
    m.scene_name = SINGLE
    m.tracers = "12000"
    m.load()
    assert m.nb.N == 21000
    assert m._separation_line() == ""


@pytest.mark.parametrize("level", ["0", "3000", "24000"])
def test_tracer_setting(level):
    from physengine.modules.galaxymodule import TRACERS

    clear_edits()
    m = GalaxyModule()
    m.tracers = level
    m.load()
    tracers = m.nb.mass == 0
    assert tracers.sum() == 2 * int(level)  # per galaxy, two galaxies
    assert (m.nb.mass > 0).sum() == 18000  # the physics does not change
    if int(level):
        share = np.sum(tracers & (m.kind == DISC)) / tracers.sum()
        assert share == pytest.approx(TRACERS[DISC] / sum(TRACERS.values()), abs=1e-3)


# ------------------------------------------------------------ edited scenes


def test_edited_scene_is_built_and_full_reset_forgets_it():
    scene = scene_for("Prograde")
    assert scene == SCENES["Prograde"] and scene is not SCENES["Prograde"]

    small = GalaxyModel()
    small.halo.mass, small.halo.n = 15.0, 1200
    small.disc.mass, small.disc.n, small.disc.Q = 1.5, 1500, 2.0
    small.bulge.mass = 0.3
    scene.galaxies[1] = small
    scene.encounter.pericentre = 20.0
    scene.encounter.eccentricity = 0.7
    scene.encounter.start_distance = 60.0
    save_edits("Prograde", scene)
    scene.encounter.pericentre = 1.0  # editing the caller's copy changes nothing
    try:
        m = GalaxyModule()
        m.tracers = "0"
        m.load()
        assert m.scene.encounter.pericentre == 20.0
        nb = m.nb
        assert np.sum(m.galaxy == 1) == 1200 + 1000 + 1500
        assert nb.mass[m.galaxy == 1].sum() == pytest.approx(16.8)
        assert nb.mass[m.galaxy == 0].sum() == pytest.approx(56.0)
        np.testing.assert_allclose(nb.mass @ nb.pos, 0.0, atol=1e-9)
        np.testing.assert_allclose(nb.mass @ nb.vel, 0.0, atol=1e-10)
        assert m._separation_line().strip() == "Separation: 60.0 kpc"
        # Another scene does not see the edits.
        assert scene_for("Inclined") == SCENES["Inclined"]

        m.discard_edits()  # what Full reset calls
        assert scene_for("Prograde") == SCENES["Prograde"]
    finally:
        clear_edits()
