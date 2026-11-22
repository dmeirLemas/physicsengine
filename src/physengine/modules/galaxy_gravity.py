import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def direct_acc(pos, mass, G, eps):
    n = pos.shape[0]
    out = np.zeros((n, 3))
    eps2 = eps * eps
    for i in numba.prange(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            f = mass[j] / (r2 * np.sqrt(r2))
            ax += f * dx
            ay += f * dy
            az += f * dz
        out[i, 0] = G * ax
        out[i, 1] = G * ay
        out[i, 2] = G * az
    return out


def allocate_tree(max_nodes):
    center = np.zeros(shape=(max_nodes, 3), dtype=np.float64)
    half = np.zeros(shape=(max_nodes,), dtype=np.float64)

    children = -1 * np.ones(shape=(max_nodes, 8), dtype=np.int64)
    body = -1 * np.ones(shape=(max_nodes,), dtype=np.int64)

    node_mass = np.zeros(shape=(max_nodes,), dtype=np.float64)
    node_com = np.zeros(shape=(max_nodes, 3), dtype=np.float64)

    return center, half, children, body, node_mass, node_com


def root_cell(pos):
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)

    center = (lo + hi) / 2
    half = np.max(hi - lo) / 2 * 1.0001

    return center, half


@numba.njit(cache=True)
def octant(c, p):
    res = 0
    if p[0] >= c[0]:
        res += 1
    if p[1] >= c[1]:
        res += 2
    if p[2] >= c[2]:
        res += 4
    return res


@numba.njit(cache=True)
def child_center(c, h, k):
    q = h / 2
    out = np.empty(3)
    out[0] = c[0] + q if k & 1 else c[0] - q
    out[1] = c[1] + q if k & 2 else c[1] - q
    out[2] = c[2] + q if k & 4 else c[2] - q
    return out


# Pieces 3-6 of the Barnes-Hut tree: written by Claude under a one-off
# override of CLAUDE.md (2026-11-20). Review the method.
#
# body[node]: -1 empty, >= 0 a leaf holding that particle, -2 split (internal).
EMPTY = -1
INTERNAL = -2


@numba.njit(cache=True)
def _new_child(node, k, n_nodes, center, half, children):
    if n_nodes >= center.shape[0]:
        raise ValueError("Barnes-Hut tree is out of nodes")
    center[n_nodes] = child_center(center[node], half[node], k)
    half[n_nodes] = half[node] / 2
    children[node, k] = n_nodes
    return n_nodes


@numba.njit(cache=True)
def insert_all(pos, center, half, children, body):
    """Insert every particle; row 0 (the root cube) must be set. Returns the number of nodes used."""
    n_nodes = 1
    for i in range(pos.shape[0]):
        node = 0
        while True:
            b = body[node]
            if b == EMPTY:
                body[node] = i
                break
            if b >= 0:
                # occupied leaf: split it, moving its particle one level down
                body[node] = INTERNAL
                c = _new_child(node, octant(center[node], pos[b]), n_nodes, center, half, children)
                n_nodes += 1
                body[c] = b
            k = octant(center[node], pos[i])
            child = children[node, k]
            if child == -1:
                c = _new_child(node, k, n_nodes, center, half, children)
                n_nodes += 1
                body[c] = i
                break
            node = child
    return n_nodes


@numba.njit(cache=True)
def compute_moments(pos, mass, n_nodes, children, body, node_mass, node_com):
    """Mass and centre of mass of every node. Children always come after their parent, so go backwards."""
    for node in range(n_nodes - 1, -1, -1):
        b = body[node]
        if b >= 0:
            node_mass[node] = mass[b]
            node_com[node, 0] = pos[b, 0]
            node_com[node, 1] = pos[b, 1]
            node_com[node, 2] = pos[b, 2]
            continue
        m = 0.0
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for k in range(8):
            c = children[node, k]
            if c >= 0:
                mc = node_mass[c]
                m += mc
                cx += mc * node_com[c, 0]
                cy += mc * node_com[c, 1]
                cz += mc * node_com[c, 2]
        node_mass[node] = m
        if m > 0.0:
            node_com[node, 0] = cx / m
            node_com[node, 1] = cy / m
            node_com[node, 2] = cz / m


@numba.njit(parallel=True, cache=True)
def tree_acc(pos, center, half, children, body, node_mass, node_com, G, eps, theta):
    """Softened accelerations from the tree.

    A node is used as a single point mass when d > s / theta + delta, with d the
    distance to its centre of mass, s its side length and delta the offset of
    its centre of mass from its geometric centre (Barnes 1994; the delta term
    stops a node from being accepted by a particle inside or right next to it).
    Otherwise it is opened and its children are visited. Plummer softening as in
    direct_acc, applied to nodes too.
    """
    n = pos.shape[0]
    out = np.zeros((n, 3))
    eps2 = eps * eps
    inv_theta = 1.0 / theta
    for i in numba.prange(n):
        stack = np.empty(1024, dtype=np.int64)
        sp = 0
        stack[sp] = 0
        sp += 1
        ax = 0.0
        ay = 0.0
        az = 0.0
        px = pos[i, 0]
        py = pos[i, 1]
        pz = pos[i, 2]
        while sp > 0:
            sp -= 1
            node = stack[sp]
            b = body[node]
            if b == i or b == EMPTY:
                continue
            dx = node_com[node, 0] - px
            dy = node_com[node, 1] - py
            dz = node_com[node, 2] - pz
            r2 = dx * dx + dy * dy + dz * dz
            if b == INTERNAL:
                ox = node_com[node, 0] - center[node, 0]
                oy = node_com[node, 1] - center[node, 1]
                oz = node_com[node, 2] - center[node, 2]
                limit = 2.0 * half[node] * inv_theta + np.sqrt(ox * ox + oy * oy + oz * oz)
                if r2 < limit * limit:
                    for k in range(8):
                        c = children[node, k]
                        if c >= 0:
                            stack[sp] = c
                            sp += 1
                    continue
            r2 += eps2
            f = node_mass[node] / (r2 * np.sqrt(r2))
            ax += f * dx
            ay += f * dy
            az += f * dz
        out[i, 0] = G * ax
        out[i, 1] = G * ay
        out[i, 2] = G * az
    return out


@numba.njit(cache=True)
def _split_top(pos, center, half, children, body, idx, cell_limit):
    """Serial top of the parallel build: split cubes until each holds at most
    cell_limit particles. Reorders idx so every final cell's particles are one
    contiguous slice. Returns (cells, starts, ends, n_top): the final cells'
    node indices and slices, and the number of nodes used so far."""
    n = idx.shape[0]
    cells = np.empty(n + 1, dtype=np.int64)
    starts = np.empty(n + 1, dtype=np.int64)
    ends = np.empty(n + 1, dtype=np.int64)
    n_cells = 0
    q_node = np.empty(4096, dtype=np.int64)
    q_start = np.empty(4096, dtype=np.int64)
    q_end = np.empty(4096, dtype=np.int64)
    q_node[0] = 0
    q_start[0] = 0
    q_end[0] = n
    sp = 1
    n_nodes = 1
    scratch = np.empty(n, dtype=np.int64)
    octs = np.empty(n, dtype=np.int64)
    while sp > 0:
        sp -= 1
        node, s, e = q_node[sp], q_start[sp], q_end[sp]
        if (
            e - s <= cell_limit
            or sp + 8 >= q_node.shape[0]
            or n_nodes + 8 > half.shape[0]
            or half[node] < 1e-9 * half[0]
        ):
            cells[n_cells] = node
            starts[n_cells] = s
            ends[n_cells] = e
            n_cells += 1
            continue
        # counting sort of this slice by octant
        cnt = np.zeros(8, dtype=np.int64)
        for t in range(s, e):
            k = octant(center[node], pos[idx[t]])
            octs[t] = k
            cnt[k] += 1
        off = np.zeros(8, dtype=np.int64)
        acc = s
        for k in range(8):
            off[k] = acc
            acc += cnt[k]
        for t in range(s, e):
            k = octs[t]
            scratch[off[k]] = idx[t]
            off[k] += 1
        for t in range(s, e):
            idx[t] = scratch[t]
        body[node] = INTERNAL
        q = h = half[node] / 2
        acc = s
        for k in range(8):
            if cnt[k] > 0:
                c = n_nodes
                n_nodes += 1
                center[c, 0] = center[node, 0] + (q if k & 1 else -q)
                center[c, 1] = center[node, 1] + (q if k & 2 else -q)
                center[c, 2] = center[node, 2] + (q if k & 4 else -q)
                half[c] = h
                children[node, k] = c
                q_node[sp] = c
                q_start[sp] = acc
                q_end[sp] = acc + cnt[k]
                sp += 1
            acc += cnt[k]
    return cells[:n_cells], starts[:n_cells], ends[:n_cells], n_nodes


@numba.njit(cache=True)
def _insert_range(pos, idx, s, e, root, first_free, last_free, center, half, children, body):
    """insert_all for the particles idx[s:e] into the subtree at `root`,
    creating nodes only in [first_free, last_free). Returns the next free node,
    or -1 if the range ran out."""
    n_next = first_free
    for t in range(s, e):
        i = idx[t]
        node = root
        while True:
            b = body[node]
            if b == EMPTY:
                body[node] = i
                break
            if b >= 0:
                if n_next >= last_free:
                    return -1
                body[node] = INTERNAL
                k = octant(center[node], pos[b])
                c = n_next
                n_next += 1
                q = half[node] / 2
                center[c, 0] = center[node, 0] + (q if k & 1 else -q)
                center[c, 1] = center[node, 1] + (q if k & 2 else -q)
                center[c, 2] = center[node, 2] + (q if k & 4 else -q)
                half[c] = q
                children[node, k] = c
                body[c] = b
            k = octant(center[node], pos[i])
            child = children[node, k]
            if child == -1:
                if n_next >= last_free:
                    return -1
                c = n_next
                n_next += 1
                q = half[node] / 2
                center[c, 0] = center[node, 0] + (q if k & 1 else -q)
                center[c, 1] = center[node, 1] + (q if k & 2 else -q)
                center[c, 2] = center[node, 2] + (q if k & 4 else -q)
                half[c] = q
                children[node, k] = c
                body[c] = i
                break
            node = child
    return n_next


@numba.njit(parallel=True, cache=True)
def _fill_cells(pos, idx, cells, starts, ends, bases, caps, center, half, children, body):
    ok = np.ones(cells.shape[0], dtype=np.bool_)
    for j in numba.prange(cells.shape[0]):
        r = _insert_range(
            pos,
            idx,
            starts[j],
            ends[j],
            cells[j],
            bases[j],
            bases[j] + caps[j],
            center,
            half,
            children,
            body,
        )
        if r < 0:
            ok[j] = False
    return ok.all()


class TreeBuffers:
    """Node arrays kept between builds, so a tree built every step does not
    allocate (allocating and filling fresh arrays cost more than the parallel
    build itself). A tree built into these buffers is valid until the next
    build into the same buffers."""

    def __init__(self):
        self.capacity = 0
        self.arrays = None

    def get(self, n_nodes):
        """Arrays with room for n_nodes. Not reset: build_tree resets exactly
        the rows it uses."""
        if n_nodes > self.capacity:
            self.capacity = int(n_nodes * 1.25) + 64
            self.arrays = allocate_tree(self.capacity)
        return self.arrays


NODES_PER_PARTICLE = 2.5  # room per cell slice; measured use is ~1.5, overflow falls back to serial


def build_tree(pos, mass, buffers=None):
    """Build the tree; returns (center, half, children, body, node_mass, node_com, n_nodes).

    Parallel build (written by Claude, 2026-11-20, same override as the tree):
    the top of the tree is split serially until each cell holds at most ~n/128
    particles (adaptive, since nearly all particles sit in a few dense
    regions); then every such cell's subtree is built by insert_all's method
    in parallel, each in its own slice of the node arrays. Node numbers keep
    "children after parents", so compute_moments and all walks work as before;
    unused slots between slices stay empty (body -1, mass 0) and are never
    referenced. Falls back to the serial build if a slice runs out of room.
    With `buffers` (a TreeBuffers) the node arrays are reused instead of
    allocated.
    """
    n = len(pos)
    cell_limit = max(64, n // 128)
    top_max = 16 * (n // cell_limit + 1) + 64
    # Upper bound on rows: the top split uses at most top_max, and the slices
    # add NODES_PER_PARTICLE per particle plus 16 per cell (cells <= top_max).
    max_nodes = 17 * top_max + int(NODES_PER_PARTICLE * n) + 100
    if buffers is None:
        center, half, children, body, node_mass, node_com = allocate_tree(max_nodes)
    else:
        center, half, children, body, node_mass, node_com = buffers.get(max_nodes)
        children[:top_max] = -1  # reused arrays: reset the rows the top split may use
        body[:top_max] = EMPTY
    idx = np.arange(n, dtype=np.int64)
    center[0], half[0] = root_cell(pos)
    # The top split may use at most top_max nodes; the slices go after it.
    cells, starts, ends, n_top = _split_top(
        pos, center, half[:top_max], children, body, idx, cell_limit
    )
    caps = (NODES_PER_PARTICLE * (ends - starts)).astype(np.int64) + 16
    bases = n_top + np.concatenate([[0], np.cumsum(caps)[:-1]]).astype(np.int64)
    n_nodes = int(n_top + caps.sum())
    if buffers is not None and n_nodes <= len(half):
        children[n_top:n_nodes] = -1  # ... and the rows of the slices
        body[n_top:n_nodes] = EMPTY
    if n_nodes <= len(half) and _fill_cells(
        pos, idx, cells, starts, ends, bases, caps, center, half, children, body
    ):
        compute_moments(pos, mass, n_nodes, children, body, node_mass, node_com)
        return center, half, children, body, node_mass, node_com, n_nodes
    return build_tree_serial(pos, mass)


def build_tree_serial(pos, mass):
    """The original one-thread build (kept as the fallback and for reference)."""
    max_nodes = 4 * len(pos) + 100
    while True:
        center, half, children, body, node_mass, node_com = allocate_tree(max_nodes)
        center[0], half[0] = root_cell(pos)
        try:
            n_nodes = insert_all(pos, center, half, children, body)
            break
        except ValueError:
            max_nodes *= 2  # deeply clustered particles; try again with more room
    compute_moments(pos, mass, n_nodes, children, body, node_mass, node_com)
    return center, half, children, body, node_mass, node_com, n_nodes


def bh_acc(pos, mass, G, eps, theta):
    """Barnes-Hut accelerations: build the tree, then walk it for every particle."""
    if len(pos) == 0:
        return np.zeros((0, 3))
    center, half, children, body, node_mass, node_com, _ = build_tree(pos, mass)
    return tree_acc(pos, center, half, children, body, node_mass, node_com, G, eps, theta)


# Group walk (written by Claude, same override as the tree; 2026-11-20).
# The tree is walked once per group of at most GROUP_SIZE neighbouring
# particles (Barnes 1990, J. Comp. Phys. 87, 161). A node is accepted for the
# whole group only if it passes the opening test for the nearest point of the
# group's cube, so it passes for every particle in the group: the result is at
# least as accurate as the per-particle walk. The shared interaction list is
# then summed for each particle of the group in a tight loop.
GROUP_SIZE = 32


@numba.njit(cache=True)
def particle_ranges(n_nodes, children, body):
    """Order the particles so that every node's particles are one contiguous range.

    Returns (order, start, count): order[start[node] : start[node] + count[node]]
    are the particles in node.
    """
    count = np.zeros(n_nodes, dtype=np.int64)
    for node in range(n_nodes - 1, -1, -1):
        if body[node] >= 0:
            count[node] = 1
        else:
            for k in range(8):
                c = children[node, k]
                if c >= 0:
                    count[node] += count[c]
    start = np.zeros(n_nodes, dtype=np.int64)
    order = np.empty(count[0], dtype=np.int64)
    for node in range(n_nodes):
        if body[node] >= 0:
            order[start[node]] = body[node]
            continue
        offset = start[node]
        for k in range(8):
            c = children[node, k]
            if c >= 0:
                start[c] = offset
                offset += count[c]
    return order, start, count


@numba.njit(cache=True)
def find_groups(n_nodes, children, count, group_size):
    """Largest nodes holding at most group_size particles; together they cover every particle."""
    groups = np.empty(n_nodes, dtype=np.int64)
    n_groups = 0
    stack = np.empty(8 * 64, dtype=np.int64)
    sp = 1
    stack[0] = 0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if count[node] <= group_size:
            groups[n_groups] = node
            n_groups += 1
            continue
        for k in range(8):
            c = children[node, k]
            if c >= 0:
                stack[sp] = c
                sp += 1
    return groups[:n_groups]


@numba.njit(parallel=True, fastmath=True, cache=True)
def group_walk_acc(
    t_pos,
    t_center,
    t_half,
    t_order,
    t_start,
    t_count,
    t_groups,
    s_center,
    s_half,
    s_children,
    s_body,
    s_mass,
    s_com,
    n_src,
    G,
    eps,
    theta,
):
    """Accelerations of target particles from a source tree, walking once per target group.

    The targets have their own tree (t_*), used only for its groups: the
    largest cells holding at most GROUP_SIZE targets. The source tree (s_*)
    supplies the masses. For self-gravity both are the same tree. Needs eps > 0
    (see step 2).
    """
    out = np.zeros((t_pos.shape[0], 3))
    eps2 = eps * eps
    inv_theta = 1.0 / theta
    for gi in numba.prange(t_groups.shape[0]):
        g = t_groups[gi]
        gx = t_center[g, 0]
        gy = t_center[g, 1]
        gz = t_center[g, 2]
        gh = t_half[g]

        # 1. Walk the source tree once for the whole group, copying the accepted
        #    nodes into contiguous arrays so the sum in step 2 can use SIMD.
        lx = np.empty(n_src)
        ly = np.empty(n_src)
        lz = np.empty(n_src)
        lm = np.empty(n_src)
        n_int = 0
        stack = np.empty(1024, dtype=np.int64)
        sp = 1
        stack[0] = 0
        while sp > 0:
            sp -= 1
            node = stack[sp]
            b = s_body[node]
            if b == EMPTY or s_mass[node] == 0.0:
                continue
            if b == INTERNAL:
                # distance from the node's centre of mass to the nearest point of the group's cube
                ex = max(abs(s_com[node, 0] - gx) - gh, 0.0)
                ey = max(abs(s_com[node, 1] - gy) - gh, 0.0)
                ez = max(abs(s_com[node, 2] - gz) - gh, 0.0)
                d2 = ex * ex + ey * ey + ez * ez
                ox = s_com[node, 0] - s_center[node, 0]
                oy = s_com[node, 1] - s_center[node, 1]
                oz = s_com[node, 2] - s_center[node, 2]
                limit = 2.0 * s_half[node] * inv_theta + np.sqrt(ox * ox + oy * oy + oz * oz)
                if d2 < limit * limit:
                    for k in range(8):
                        c = s_children[node, k]
                        if c >= 0:
                            stack[sp] = c
                            sp += 1
                    continue
            lx[n_int] = s_com[node, 0]
            ly[n_int] = s_com[node, 1]
            lz[n_int] = s_com[node, 2]
            lm[n_int] = s_mass[node]
            n_int += 1

        # 2. Apply the list to every target of the group. In self-gravity a
        #    particle's own leaf is in the list, but there dx = dy = dz = 0, so
        #    it adds nothing as long as eps > 0 (with eps = 0 it would be 0/0).
        for q in range(t_start[g], t_start[g] + t_count[g]):
            i = t_order[q]
            px = t_pos[i, 0]
            py = t_pos[i, 1]
            pz = t_pos[i, 2]
            ax = 0.0
            ay = 0.0
            az = 0.0
            for t in range(n_int):
                dx = lx[t] - px
                dy = ly[t] - py
                dz = lz[t] - pz
                r2 = dx * dx + dy * dy + dz * dz + eps2
                f = lm[t] / (r2 * np.sqrt(r2))
                ax += f * dx
                ay += f * dy
                az += f * dz
            out[i, 0] = G * ax
            out[i, 1] = G * ay
            out[i, 2] = G * az
    return out


def _grouped_tree(pos, mass, group_size):
    center, half, children, body, node_mass, node_com, n_nodes = build_tree(pos, mass)
    order, start, count = particle_ranges(n_nodes, children, body)
    groups = find_groups(n_nodes, children, count, group_size)
    return center, half, children, body, node_mass, node_com, n_nodes, order, start, count, groups


def bh_acc_grouped(
    pos, mass, G, eps, theta, group_size=GROUP_SIZE, tracer_theta=None, tracer_group_size=64
):
    """Barnes-Hut accelerations with the group walk. eps must be > 0.

    Particles with mass 0 (tracers) are moved by the gravity of the others
    but exert none: the tree is built from the massive particles only, and
    the tracers get a second, separate grouped walk through it, with their
    own opening angle (default: theta) and group size. Tracers only affect
    the picture, so a looser walk is fine for them.
    """
    if eps <= 0.0:
        raise ValueError("the group walk needs softening eps > 0")
    out = np.zeros((len(pos), 3))
    heavy = mass > 0.0
    if not heavy.any():
        return out

    hp, hm = np.ascontiguousarray(pos[heavy]), np.ascontiguousarray(mass[heavy])
    src = _grouped_tree(hp, hm, group_size)
    center, half, children, body, node_mass, node_com, n_nodes, order, start, count, groups = src
    source = (center, half, children, body, node_mass, node_com, n_nodes)
    out[heavy] = group_walk_acc(
        hp, center, half, order, start, count, groups, *source, G, eps, theta
    )

    if not heavy.all():
        tp = np.ascontiguousarray(pos[~heavy])
        t_center, t_half, _, _, _, _, _, t_order, t_start, t_count, t_groups = _grouped_tree(
            tp, np.ones(len(tp)), tracer_group_size
        )
        t_theta = theta if tracer_theta is None else tracer_theta
        out[~heavy] = group_walk_acc(
            tp, t_center, t_half, t_order, t_start, t_count, t_groups, *source, G, eps, t_theta
        )
    return out
