"""Tests of the galaxy gravity: direct summation and the Barnes-Hut tree.

Reference values and their sources:

- Plummer-softened Newtonian gravity:
      a_i = G sum_{j != i} m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2).
  module 1's nbodystate.acc evaluates the same formula with NumPy.
- Newton's third law: sum_i m_i a_i = 0 for direct summation (exactly, up to
  rounding). A tree code does not have this symmetry; its momentum error is of
  the order of its force error.
- A Barnes-Hut tree with opening angle theta -> 0 opens every node and so
  reduces to direct summation (Barnes & Hut 1986, Nature 324, 446).
- Typical Barnes-Hut force errors at theta = 0.5 with monopole nodes are a few
  tenths of a percent median, about 1% at the 99th percentile (e.g. Barnes &
  Hut 1989, ApJS 70, 389; Springel 2005, MNRAS 364, 1105, fig. 1 shows the
  same order for a comparable criterion).
- Tree structure: every particle in exactly one leaf, inside that leaf's
  cube; a node's mass is the sum of its particles' masses and its centre of
  mass their mass-weighted mean.
- Leapfrog with a fixed step: energy error bounded and small (Hairer,
  Lubich & Wanner 2006, ch. IX).
"""

import numpy as np
import pytest

from physengine.modules.galaxy_gravity import (
    EMPTY,
    INTERNAL,
    bh_acc,
    bh_acc_grouped,
    build_tree,
    child_center,
    direct_acc,
    find_groups,
    octant,
    particle_ranges,
    root_cell,
)
from physengine.modules.galaxymodule import SINGLE, GalaxyModule
from physengine.modules.nbodystate import Integrator, NBodyState, acc

EPS = 0.1


@pytest.fixture(scope="module")
def galaxy():
    m = GalaxyModule()
    m.scene_name = SINGLE
    m.load()
    return m.nb.pos.copy(), m.nb.mass.copy()


def random_cloud(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 5.0, (n, 3)), rng.uniform(1e-3, 1e-2, n)


# ---------------------------------------------------------------- direct sum


def test_direct_acc_matches_module_one():
    pos, mass = random_cloud(500)
    np.testing.assert_allclose(
        direct_acc(pos, mass, 1.0, EPS), acc(pos, mass, 1.0, EPS), rtol=1e-12, atol=1e-14
    )


def test_direct_acc_third_law():
    pos, mass = random_cloud(500)
    a = direct_acc(pos, mass, 1.0, EPS)
    scale = np.sum(mass[:, None] * np.abs(a))
    np.testing.assert_allclose(mass @ a, 0.0, atol=1e-14 * scale)


def test_direct_acc_two_bodies():
    # Two unit masses 2 apart, eps = 0: each pulled toward the other with G m / r^2 = 1/4.
    pos = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    a = direct_acc(pos, np.ones(2), 1.0, 0.0)
    np.testing.assert_allclose(a, [[0.25, 0, 0], [-0.25, 0, 0]], atol=1e-15)


# ------------------------------------------------------------------ geometry


def test_root_cell_contains_everything(galaxy):
    pos, _ = galaxy
    c, h = root_cell(pos)
    assert np.all(np.abs(pos - c) <= h)


def test_octant_and_child_center_agree():
    rng = np.random.default_rng(1)
    c, h = np.array([1.0, -2.0, 0.5]), 3.0
    seen = set()
    for p in c + rng.uniform(-h, h, (2000, 3)):
        k = octant(c, p)
        seen.add(k)
        assert np.all(np.abs(p - child_center(c, h, k)) <= h / 2)
    assert seen == set(range(8))
    np.testing.assert_allclose(child_center(np.zeros(3), 2.0, 0), [-1, -1, -1])
    np.testing.assert_allclose(child_center(np.zeros(3), 2.0, 7), [1, 1, 1])


# ------------------------------------------------------------------- the tree


def test_tree_structure(galaxy):
    pos, mass = galaxy
    center, half, children, body, node_mass, node_com, n = build_tree(pos, mass)
    body, children = body[:n], children[:n]

    leaves = np.flatnonzero(body >= 0)
    assert sorted(body[leaves]) == list(range(len(pos)))  # every particle exactly once
    for node in leaves:
        assert np.all(np.abs(pos[body[node]] - center[node]) <= half[node] * (1 + 1e-12))
        assert np.all(children[node] == -1)
    internal = body == INTERNAL
    assert np.all((children[internal] >= 0).any(axis=1))
    # The parallel build leaves unused rows between its slices: they must be
    # empty, massless and never referenced by any node.
    empty = np.flatnonzero(body == EMPTY)
    referenced = set(children[children >= 0].tolist())
    assert not referenced.intersection(empty.tolist())
    assert np.all(node_mass[empty] == 0.0)

    assert node_mass[0] == pytest.approx(mass.sum())
    np.testing.assert_allclose(node_com[0], mass @ pos / mass.sum(), atol=1e-10)


def test_node_moments_are_sums_of_children(galaxy):
    pos, mass = galaxy
    center, half, children, body, node_mass, node_com, n = build_tree(pos, mass)
    internal = np.flatnonzero(
        (body[:n] == INTERNAL) & (node_mass[:n] > 0)
    )  # tracer-only cells have no mass
    for node in internal[:: max(1, len(internal) // 500)]:
        kids = children[node][children[node] >= 0]
        kids = kids[node_mass[kids] > 0]
        assert node_mass[node] == pytest.approx(node_mass[kids].sum(), rel=1e-12)
        np.testing.assert_allclose(
            node_com[node],
            node_mass[kids] @ node_com[kids] / node_mass[kids].sum(),
            rtol=1e-10,
            atol=1e-10,
        )


def test_tree_survives_tight_clustering():
    # Many particles packed far closer together than the root cube is wide.
    rng = np.random.default_rng(2)
    pos = np.vstack([rng.normal(0, 1e-6, (500, 3)), rng.normal(0, 100.0, (500, 3))])
    mass = np.ones(1000)
    a = bh_acc(pos, mass, 1.0, EPS, 1e-6)
    np.testing.assert_allclose(a, direct_acc(pos, mass, 1.0, EPS), rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------ the forces


def test_tree_with_tiny_theta_is_direct_summation(galaxy):
    pos, mass = galaxy
    a = bh_acc(pos, mass, 1.0, EPS, 1e-6)
    np.testing.assert_allclose(a, direct_acc(pos, mass, 1.0, EPS), rtol=1e-10, atol=1e-13)


def test_tree_force_error_at_theta_half(galaxy):
    pos, mass = galaxy
    ref = direct_acc(pos, mass, 1.0, EPS)
    a = bh_acc(pos, mass, 1.0, EPS, 0.5)
    err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    # Measured: median 0.30%, 99th percentile 0.93%.
    assert np.median(err) < 0.005
    assert np.quantile(err, 0.99) < 0.02


def test_tree_error_shrinks_with_theta(galaxy):
    pos, mass = galaxy
    ref = direct_acc(pos, mass, 1.0, EPS)
    medians = []
    for theta in (0.7, 0.5, 0.3):
        a = bh_acc(pos, mass, 1.0, EPS, theta)
        medians.append(np.median(np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)))
    assert medians[0] > medians[1] > medians[2]


def test_tree_momentum_error_is_small(galaxy):
    pos, mass = galaxy
    a = bh_acc(pos, mass, 1.0, EPS, 0.5)
    scale = np.sum(mass[:, None] * np.abs(a))
    assert np.max(np.abs(mass @ a)) < 1e-2 * scale


def test_empty_system():
    assert bh_acc(np.zeros((0, 3)), np.zeros(0), 1.0, EPS, 0.5).shape == (0, 3)


# ------------------------------------------------------------- time evolution


def test_short_run_conserves_energy():
    # A 1500-particle galaxy-like cloud, 2 time units (200 leapfrog steps).
    pos, mass = random_cloud(1500, seed=4)
    mass /= mass.sum()
    vel = np.random.default_rng(5).normal(0.0, 0.1, pos.shape)

    s = NBodyState(len(mass))
    s.pos[:], s.vel[:], s.mass[:] = pos, vel, mass

    def energy(p, v):
        kin = 0.5 * np.sum(mass * np.sum(v**2, axis=1))
        pot = 0.0
        for i in range(len(p)):
            d = np.sqrt(np.sum((p[i + 1 :] - p[i]) ** 2, axis=1) + EPS**2)
            pot -= mass[i] * np.sum(mass[i + 1 :] / d)
        return kin + pot

    e0 = energy(s.pos, s.vel)
    integ = Integrator(lambda p, m: bh_acc(p, m, 1.0, EPS, 0.5))
    integ.step(s, 0.01, 200)
    assert abs((energy(s.pos, s.vel) - e0) / e0) < 1e-3


# ----------------------------------------------------------------- group walk


def test_particle_ranges_and_groups(galaxy):
    pos, mass = galaxy
    center, half, children, body, node_mass, node_com, n = build_tree(pos, mass)
    order, start, count = particle_ranges(n, children, body)
    assert sorted(order) == list(range(len(pos)))
    # Each node's range holds exactly the particles inside its cube.
    for node in range(0, n, 97):
        members = order[start[node] : start[node] + count[node]]
        assert np.all(np.abs(pos[members] - center[node]) <= half[node] * (1 + 1e-12))
    groups = find_groups(n, children, count, 32)
    assert np.all(count[groups] <= 32)
    covered = np.concatenate([order[start[g] : start[g] + count[g]] for g in groups])
    assert sorted(covered) == list(range(len(pos)))  # every particle in exactly one group


def test_group_walk_with_tiny_theta_is_direct_summation(galaxy):
    pos, mass = galaxy
    np.testing.assert_allclose(
        bh_acc_grouped(pos, mass, 1.0, EPS, 1e-6),
        direct_acc(pos, mass, 1.0, EPS),
        rtol=1e-9,
        atol=1e-12,
    )


def test_group_walk_is_at_least_as_accurate_as_per_particle_walk(galaxy):
    pos, mass = galaxy
    ref = direct_acc(pos, mass, 1.0, EPS)
    errors = {}
    for name, f in [("particle", bh_acc), ("group", bh_acc_grouped)]:
        a = f(pos, mass, 1.0, EPS, 0.5)
        errors[name] = np.median(np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1))
    assert errors["group"] <= errors["particle"]


def test_group_walk_at_the_module_setting(galaxy):
    # theta = 0.7 with the group walk: measured median 0.27%, 99th percentile 1.2%.
    pos, mass = galaxy
    ref = direct_acc(pos, mass, 1.0, EPS)
    a = bh_acc_grouped(pos, mass, 1.0, EPS, 0.7)
    err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(err) < 0.005
    assert np.quantile(err, 0.99) < 0.02


def test_group_walk_needs_softening():
    pos, mass = random_cloud(10)
    with pytest.raises(ValueError):
        bh_acc_grouped(pos, mass, 1.0, 0.0, 0.5)


def test_tracers_feel_gravity_but_exert_none():
    rng = np.random.default_rng(6)
    heavy_pos, heavy_mass = random_cloud(2000, seed=7)
    tracers = rng.normal(0.0, 5.0, (3000, 3))
    pos = np.vstack([heavy_pos, tracers])
    mass = np.concatenate([heavy_mass, np.zeros(3000)])
    a = bh_acc_grouped(pos, mass, 1.0, EPS, 1e-6, tracer_theta=1e-6)
    # Exact for theta -> 0: massive particles feel only each other, tracers feel the massive ones.
    np.testing.assert_allclose(
        a[:2000], direct_acc(heavy_pos, heavy_mass, 1.0, EPS), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(a, direct_acc(pos, mass, 1.0, EPS), rtol=1e-9, atol=1e-12)
    # With the loose tracer walk the massive forces do not change at all.
    loose = bh_acc_grouped(pos, mass, 1.0, EPS, 0.7, tracer_theta=1.0)
    np.testing.assert_array_equal(
        loose[:2000], bh_acc_grouped(heavy_pos, heavy_mass, 1.0, EPS, 0.7)
    )


def test_only_tracers():
    pos, _ = random_cloud(100)
    np.testing.assert_array_equal(bh_acc_grouped(pos, np.zeros(100), 1.0, EPS, 0.7), 0.0)


def test_parallel_build_matches_serial_build(galaxy):
    from physengine.modules.galaxy_gravity import TreeBuffers, build_tree_serial

    pos, mass = galaxy
    heavy = mass > 0
    hp, hm = np.ascontiguousarray(pos[heavy]), mass[heavy]
    ref = build_tree_serial(hp, hm)
    buffers = TreeBuffers()
    for shift in (0.0, 0.3, 0.6):  # reused buffers must give a clean tree every time
        tree = build_tree(hp + shift, hm, buffers)
        center, half, children, body, node_mass, node_com, n = tree
        leaves = body[:n][body[:n] >= 0]
        assert sorted(leaves) == list(range(len(hp)))
        assert node_mass[0] == pytest.approx(ref[4][0])
        np.testing.assert_allclose(node_com[0], ref[5][0] + shift, atol=1e-9)
        order, start, count = particle_ranges(n, children, body)
        assert count[0] == len(hp)
