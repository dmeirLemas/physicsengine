"""Tests of the GPU tree walk (modules/galaxy_gpu.py).

Skipped when no OpenGL 4.3+ context with compute shaders can be created.

Reference values and their sources:

- With theta -> 0 every node is opened, so the walk is direct summation; the
  GPU works in float32, whose rounding (~6e-8 per operation) accumulated over
  ~10^4 terms gives relative force errors of order 1e-5 against the float64
  CPU result.
- With the same opening test as the CPU per-particle walk
  (galaxy_gravity.tree_acc) the GPU visits the same nodes, so the two agree to
  float32 rounding.
- Threaded (depth-first) tree layout: a node's first child immediately follows
  it, and next[q] is the index just past q's subtree, so next[q] > q and the
  root's next is the node count; walking with every node opened visits every
  node exactly once.
"""

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from physengine.modules.galaxy_gpu import GpuGravity, morton_order, threaded_layout  # noqa: E402
from physengine.modules.galaxy_gravity import (  # noqa: E402
    INTERNAL,
    build_tree,
    direct_acc,
    particle_ranges,
    tree_acc,
)
from physengine.modules.galaxymodule import GalaxyModule  # noqa: E402

EPS = 0.1


@pytest.fixture(scope="module")
def gpu():
    try:
        ctx = moderngl.create_standalone_context(require=430)
        g = GpuGravity(ctx)
    except Exception as error:
        pytest.skip(f"no GPU compute shaders: {error}")
        return
    yield g
    g.release()
    ctx.release()


@pytest.fixture(scope="module")
def small():
    """3000 massive particles and 2000 tracers: enough structure for exact
    (theta -> 0) checks, small enough that opening every node stays well within
    GPU driver time limits."""
    rng = np.random.default_rng(11)
    pos = np.vstack(
        [rng.normal(0, 5, (3000, 3)), rng.normal(0, 2, (500, 3)) + 20, rng.normal(0, 8, (1500, 3))]
    )
    mass = np.concatenate([rng.uniform(1e-3, 1e-2, 3500), np.zeros(1500)])
    return pos, mass


@pytest.fixture(scope="module")
def pair():
    m = GalaxyModule()  # default scene: two galaxies; with tracers
    m.tracers = "12000"
    m.load()
    return m.nb.pos.copy(), m.nb.mass.copy()


def test_threaded_layout(pair):
    pos, mass = pair
    heavy = mass > 0
    center, half, children, body, node_mass, node_com, n = build_tree(pos[heavy], mass[heavy])
    order, start, count = particle_ranges(n, children, body)
    records = threaded_layout(n, children, body, node_mass, node_com, center, half, start, count, 1)
    com_mass, geometry = records[:, :4], records[:, 4:6]
    nxt = records[:, 6].view(np.uint32)
    m = len(nxt)
    assert nxt[0] == m
    assert np.all(nxt > np.arange(m))
    assert np.all(nxt <= m)
    # Leaves (geometry < 0) hold exactly the particles' total mass.
    leaves = geometry[:, 0] < 0
    assert leaves.sum() == heavy.sum()
    assert com_mass[leaves, 3].sum() == pytest.approx(mass.sum(), rel=1e-5)
    assert com_mass[0, 3] == pytest.approx(mass.sum(), rel=1e-6)
    # Internal nodes: the first child is the next node, and the children's
    # subtrees tile the parent's subtree exactly.
    for q in np.flatnonzero(~leaves)[:300]:
        c, total = q + 1, 0.0
        while c < nxt[q]:
            total += com_mass[c, 3]
            c = nxt[c]
        assert c == nxt[q]
        assert total == pytest.approx(com_mass[q, 3], rel=1e-5)
    assert np.sum(body[:n] == INTERNAL) == (~leaves).sum()


def test_morton_order_is_a_permutation(pair):
    pos, _ = pair
    order = morton_order(pos)
    assert sorted(order) == list(range(len(pos)))
    # Neighbours along the curve are close: median step far below the system size.
    steps = np.linalg.norm(np.diff(pos[order], axis=0), axis=1)
    assert np.median(steps) < 0.01 * np.ptp(pos, axis=0).max()


def test_gpu_with_tiny_theta_is_direct_summation(gpu, small):
    pos, mass = small
    a = gpu.acc(pos, mass, 1.0, EPS, 1e-4, 1e-4)
    ref = direct_acc(pos, mass, 1.0, EPS)
    err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(err) < 2e-5
    assert np.max(err) < 1e-3


def test_gpu_matches_cpu_per_particle_walk(gpu, pair):
    pos, mass = pair
    heavy = mass > 0
    hp, hm = np.ascontiguousarray(pos[heavy]), mass[heavy]
    tree = build_tree(hp, hm)
    ref = tree_acc(hp, *tree[:6], 1.0, EPS, 0.5)
    # Buckets change how the near field is summed; with buckets of one
    # particle the GPU visits exactly the nodes the CPU walk visits.
    old = gpu.BUCKET
    gpu.BUCKET, gpu.tree = 1, None
    try:
        a = gpu.acc(pos, mass, 1.0, EPS, 0.5)[heavy]
    finally:
        gpu.BUCKET, gpu.tree = old, None
    err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(err) < 1e-5


def test_gpu_accuracy_at_the_module_setting(gpu, pair):
    # theta 1.0 for massive particles and tracers (the speed setting).
    # Measured against direct summation: massive median 1.3%, 99th
    # percentile 4.2%; tracers median 1.9%. At 0.6: 0.42% / 1.2%.
    from physengine.modules.galaxymodule import GPU_THETA, GPU_TRACER_THETA

    assert (GPU_THETA, GPU_TRACER_THETA) == (1.0, 1.0)
    pos, mass = pair
    heavy = mass > 0
    a = gpu.acc(pos, mass, 1.0, EPS, GPU_THETA, GPU_TRACER_THETA)
    ref = direct_acc(pos, mass, 1.0, EPS)
    err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(err[heavy]) < 0.02
    assert np.quantile(err[heavy], 0.99) < 0.07
    assert np.median(err[~heavy]) < 0.04
    tight = gpu.acc(pos, mass, 1.0, EPS, 0.6, 1.0)
    err = np.linalg.norm(tight - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(err[heavy]) < 0.005
    assert np.quantile(err[heavy], 0.99) < 0.02


def test_gpu_tracers_exert_no_force(gpu):
    rng = np.random.default_rng(3)
    heavy_pos = rng.normal(0, 5, (1500, 3))
    heavy_mass = rng.uniform(1e-3, 1e-2, 1500)
    pos = np.vstack([heavy_pos, rng.normal(0, 5, (2500, 3))])
    mass = np.concatenate([heavy_mass, np.zeros(2500)])
    a = gpu.acc(pos, mass, 1.0, EPS, 1e-4, 1e-4)
    alone = gpu.acc(heavy_pos, heavy_mass, 1.0, EPS, 1e-4)
    np.testing.assert_allclose(a[:1500], alone, rtol=1e-5, atol=1e-7)


def test_gpu_stale_order_is_still_correct(gpu, small):
    # The Morton order is reused between re-sorts; moving the particles must
    # not change the result, only (slightly) the speed.
    pos, mass = small
    gpu.acc(pos, mass, 1.0, EPS, 1e-4, 1e-4)  # sets an order for this system
    gpu.calls = 1  # force reuse of that order
    moved = pos + np.random.default_rng(1).normal(0, 0.5, pos.shape)
    a = gpu.acc(moved, mass, 1.0, EPS, 1e-4, 1e-4)
    ref = direct_acc(moved, mass, 1.0, EPS)
    err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(err) < 2e-5


def test_gpu_edge_cases(gpu):
    assert gpu.acc(np.zeros((0, 3)), np.zeros(0), 1.0, EPS, 0.5).shape == (0, 3)
    np.testing.assert_array_equal(gpu.acc(np.ones((5, 3)), np.zeros(5), 1.0, EPS, 0.5), 0.0)
    with pytest.raises(ValueError):
        gpu.acc(np.ones((5, 3)), np.ones(5), 1.0, 0.0, 0.5)


def test_reused_tree_keeps_its_accuracy(gpu):
    # Rebuild every 4 calls; between rebuilds only the cell moments are
    # refreshed. Move the system with 3 real leapfrog steps of dt 0.04 (the
    # oldest a tree gets) and check the force error of the 4th call against
    # direct summation: it must stay at the fresh-tree level. (Uses the shared
    # GPU fixture: a second OpenGL context released mid-run would leave the
    # fixture's context no longer current.)
    m = GalaxyModule()
    m.load()
    pos, vel, mass = m.nb.pos.copy(), m.nb.vel.copy(), m.nb.mass
    heavy = mass > 0
    old = gpu.rebuild_every
    gpu.rebuild_every, gpu.tree, gpu.order, gpu.calls = 4, None, None, 0
    try:
        a = gpu.acc(pos, mass, 1.0, EPS, 0.6, 1.0)  # builds the tree
        for _ in range(3):
            vel += 0.04 * a
            pos += 0.04 * vel
            a = gpu.acc(pos, mass, 1.0, EPS, 0.6, 1.0)  # reuses it
        assert gpu.calls == 4
    finally:
        gpu.rebuild_every, gpu.tree, gpu.order = old, None, None
    ref = direct_acc(pos, mass, 1.0, EPS)
    err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(err[heavy]) < 0.006
    assert np.quantile(err[heavy], 0.99) < 0.02


def test_buckets_do_not_change_the_answer(gpu, small):
    # Buckets only change how the near field is summed: with theta -> 0 every
    # bucket is opened and summed particle by particle, so the result is still
    # direct summation.
    pos, mass = small
    old = gpu.BUCKET
    try:
        for bucket in (1, 8, 32):
            gpu.BUCKET = bucket
            gpu.tree, gpu.order = None, None
            a = gpu.acc(pos, mass, 1.0, EPS, 1e-4, 1e-4)
            ref = direct_acc(pos, mass, 1.0, EPS)
            err = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
            assert np.median(err) < 2e-5, bucket
    finally:
        gpu.BUCKET, gpu.tree, gpu.order = old, None, None


def test_first_passage_is_near_the_pericentre_asked_for(gpu):
    # End to end: the default prograde encounter, pericentre set to 10 kpc,
    # run with the module's own settings until after the first passage
    # (predicted by galaxy_orbit at 260 Myr = 55 time units). Measured: the
    # bulges pass at 8.6 kpc after 272 Myr. Set up as a Kepler orbit of point
    # masses they passed at 18.7 kpc; as rigid spheres, at 4.5 kpc.
    from physengine.modules.galaxy_components import BULGE
    from physengine.modules.galaxymodule import EPS as MODULE_EPS
    from physengine.modules.galaxymodule import GPU_THETA, G
    from physengine.modules.nbodystate import Integrator

    m = GalaxyModule()
    m.load()
    assert m.scene.encounter.pericentre == 10.0
    integ = Integrator(lambda p, ms: gpu.acc(p, ms, G, MODULE_EPS, GPU_THETA))
    bulge = m.kind == BULGE
    closest, when, t = np.inf, 0.0, 0.0
    while t < 66.0:
        integ.step(m.nb, m.dt, 1)
        t += m.dt
        c0 = np.median(m.nb.pos[bulge & (m.galaxy == 0)], axis=0)
        c1 = np.median(m.nb.pos[bulge & (m.galaxy == 1)], axis=0)
        if np.linalg.norm(c1 - c0) < closest:
            closest, when = np.linalg.norm(c1 - c0), t
    assert 6.5 < closest < 11.5
    assert 52.0 < when < 62.0  # 245 to 292 Myr
