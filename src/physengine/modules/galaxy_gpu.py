"""Barnes-Hut forces with the tree walk on the GPU.

Written by Claude under the same override as the CPU tree (2026-11-20).
Review the method.

Per step:
1. CPU (Numba): build the octree of the massive particles and its node masses
   (galaxy_gravity.build_tree), then lay the nodes out in depth-first order.
   In that order a node's first child is the next node, and each node stores
   `next`, the index just past its subtree. The walk then needs no stack:
   accept a node and jump to `next`, or open it and step to the next index
   (the "threaded tree" used e.g. by GADGET-2, Springel 2005, MNRAS 364, 1105).
2. CPU: sort all targets (massive particles and tracers) along a Morton
   (Z-order) curve, so neighbouring GPU threads walk nearly the same nodes.
3. GPU (compute shader, float32): one thread per target walks the tree with the
   per-particle opening test d > s / theta + delta (as galaxy_gravity.tree_acc)
   and Plummer softening. Each target carries its own 1/theta, so massive
   particles and tracers share one dispatch. Each node is one 32-byte record
   (one memory read per visit), and subtrees of at most BUCKET particles are
   summed directly instead of walked. Measured on the Iris Xe: records 25%
   faster than separate arrays; buckets of 8 another ~6%.

float32 costs ~1e-5 relative force error, far below the tree's own ~1e-3.
"""

import numba
import numpy as np

from physengine.modules.galaxy_gravity import (
    INTERNAL,
    TreeBuffers,
    build_tree,
    compute_moments,
    particle_ranges,
)

SHADER = """
#version 460
layout(local_size_x = 64) in;
// One 32-byte record per node, read in one go: centre of mass and mass, then
// (side length or -1 for a single particle, offset of the centre of mass from
// the cell centre, index past the subtree, bucket = first | count << 24).
struct Node { vec4 cm; vec4 g; };
layout(std430, binding = 0) readonly buffer Targets { vec4 targets[]; };   // xyz, 1/theta
layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };
layout(std430, binding = 2) readonly buffer Bodies { vec4 bodies[]; };     // bucket particles: xyz, mass
layout(std430, binding = 4) writeonly buffer Acc { vec4 acc[]; };
uniform uint n_targets;
uniform uint n_nodes;
uniform float eps2;
uniform float G;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n_targets) return;
    vec4 t = targets[i];
    vec3 p = t.xyz;
    float inv_theta = t.w;
    vec3 a = vec3(0.0);
    uint node = 0u;
    while (node < n_nodes) {
        Node nd = nodes[node];
        vec3 d = nd.cm.xyz - p;
        float r2 = dot(d, d);
        uint next_node = floatBitsToUint(nd.g.z);
        if (nd.g.x >= 0.0) {
            float limit = nd.g.x * inv_theta + nd.g.y;
            if (r2 < limit * limit) {          // too close to use as one point
                uint bucket = floatBitsToUint(nd.g.w);
                uint count = bucket >> 24u;
                if (count == 0u) {             // ordinary cell: open it, first child is next
                    node += 1u;
                    continue;
                }
                uint first = bucket & 0xFFFFFFu;   // bucket: sum its particles, skip the subtree
                for (uint k = 0u; k < count; ++k) {
                    vec4 b = bodies[first + k];
                    vec3 e = b.xyz - p;
                    float inv = inversesqrt(dot(e, e) + eps2);
                    a += b.w * inv * inv * inv * e;   // own particle: e = 0, adds nothing
                }
                node = next_node;
                continue;
            }
        }
        float inv = inversesqrt(r2 + eps2);
        a += nd.cm.w * inv * inv * inv * d;   // own leaf: d = 0, adds nothing (eps > 0)
        node = next_node;
    }
    acc[i] = vec4(G * a, 0.0);
}
"""


@numba.njit(cache=True)
def threaded_layout(
    n_nodes, children, body, node_mass, node_com, center, half, start, count, bucket
):
    """Depth-first node records for the stackless GPU walk, (m, 8) float32.

    Record: centre of mass x, y, z, mass; side length (-1 for a single
    particle); offset of the centre of mass from the cell centre; then, as raw
    uint32 bits, the index just past the node's subtree and its bucket. A
    subtree holding at most `bucket` particles becomes one bucket node: the
    GPU sums its particles (bodies[start : start + count], from
    galaxy_gravity.particle_ranges) directly instead of walking down it.
    Bucket field: first | count << 24 (count 0 = not a bucket).
    """
    order = np.empty(n_nodes, dtype=np.int64)
    parent_pos = np.empty(n_nodes, dtype=np.int64)
    m = 0
    stack = np.empty(1024, dtype=np.int64)
    stack_parent = np.empty(1024, dtype=np.int64)
    sp = 1
    stack[0] = 0
    stack_parent[0] = -1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        order[m] = node
        parent_pos[m] = stack_parent[sp]
        here = m
        m += 1
        if body[node] == INTERNAL and count[node] > bucket:
            for k in range(7, -1, -1):  # pushed in reverse, so child 0 comes out first
                c = children[node, k]
                if c >= 0 and node_mass[c] > 0.0:
                    stack[sp] = c
                    stack_parent[sp] = here
                    sp += 1

    size = np.ones(m, dtype=np.int64)
    for q in range(m - 1, 0, -1):
        size[parent_pos[q]] += size[q]

    records = np.zeros((m, 8), dtype=np.float32)
    bits = records.view(np.uint32)
    for q in range(m):
        node = order[q]
        records[q, 0] = node_com[node, 0]
        records[q, 1] = node_com[node, 1]
        records[q, 2] = node_com[node, 2]
        records[q, 3] = node_mass[node]
        if body[node] == INTERNAL:
            dx = node_com[node, 0] - center[node, 0]
            dy = node_com[node, 1] - center[node, 1]
            dz = node_com[node, 2] - center[node, 2]
            records[q, 4] = 2.0 * half[node]
            records[q, 5] = np.sqrt(dx * dx + dy * dy + dz * dz)
            if count[node] <= bucket:
                bits[q, 7] = np.uint32(start[node]) | (np.uint32(count[node]) << np.uint32(24))
        else:
            records[q, 4] = -1.0
        bits[q, 6] = np.uint32(q + size[q])
    return records


def morton_order(pos):
    """Indices that sort points along a 3D Morton (Z-order) curve, 10 bits per axis."""
    lo = pos.min(axis=0)
    span = max(float(np.max(pos.max(axis=0) - lo)), 1e-30)
    q = np.minimum((pos - lo) / span * 1023.0, 1023.0).astype(np.uint64)

    def spread(x):  # put two zero bits between each bit of a 10-bit number
        x = (x | (x << np.uint64(16))) & np.uint64(0x030000FF)
        x = (x | (x << np.uint64(8))) & np.uint64(0x0300F00F)
        x = (x | (x << np.uint64(4))) & np.uint64(0x030C30C3)
        x = (x | (x << np.uint64(2))) & np.uint64(0x09249249)
        return x

    key = spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)) | (spread(q[:, 2]) << np.uint64(2))
    return np.argsort(key, kind="stable")


class GpuGravity:
    """Barnes-Hut accelerations, tree built on the CPU and walked on the GPU."""

    GROUP = 64  # work-group size of SHADER (32 and 128 measured the same)
    BUCKET = 8  # particles per bucket node (measured best of 1-32)
    # Re-sort the targets along the Morton curve every this many calls. In
    # between, particles barely move, so the old order stays almost as
    # coherent; a stale order only costs speed, never accuracy.
    RESORT_EVERY = 10

    def __init__(self, ctx, rebuild_every=1):
        self.ctx = ctx
        self.program = ctx.compute_shader(SHADER)
        self.buffers = {}
        self.order = None
        self.calls = 0
        # Rebuild the tree's cells every this many calls. In between, the same
        # cells are kept and only their masses and centres of mass are
        # recomputed from the current positions (cheap). Particles drift a
        # little out of their cells, but the opening test's delta term (offset
        # of the centre of mass from the cell centre) grows with that drift and
        # opens such cells sooner, so accuracy degrades gracefully.
        self.rebuild_every = rebuild_every
        self.tree = None
        self.tree_n = -1
        self.ranges = None  # particle_ranges of the current tree
        self.tree_buffers = TreeBuffers()  # node arrays reused between builds

    def _buffer(self, name, data):
        """Write `data` to a persistent storage buffer, growing it when needed."""
        raw = np.ascontiguousarray(data).tobytes()
        buf = self.buffers.get(name)
        if buf is None or buf.size < len(raw):
            if buf is not None:
                buf.release()
            buf = self.ctx.buffer(reserve=max(len(raw), 16) * 2)
            self.buffers[name] = buf
        buf.write(raw)
        return buf

    def acc(self, pos, mass, G, eps, theta, tracer_theta=None):
        """Same contract as galaxy_gravity.bh_acc_grouped: mass-0 particles feel but exert no gravity."""
        if eps <= 0.0:
            raise ValueError("the GPU walk needs softening eps > 0")
        n = len(pos)
        out = np.zeros((n, 3))
        heavy = mass > 0.0
        if n == 0 or not heavy.any():
            return out

        hp = np.ascontiguousarray(pos[heavy])
        hm = np.ascontiguousarray(mass[heavy])
        if self.tree is None or self.tree_n != len(hp) or self.calls % self.rebuild_every == 0:
            self.tree = build_tree(hp, hm, self.tree_buffers)
            self.tree_n = len(hp)
            self.ranges = None  # belongs to the old tree
        else:
            center, half, children, body, node_mass, node_com, n_nodes = self.tree
            compute_moments(hp, hm, n_nodes, children, body, node_mass, node_com)
        center, half, children, body, node_mass, node_com, n_nodes = self.tree
        if self.ranges is None:
            self.ranges = particle_ranges(n_nodes, children, body)  # changes only with the tree
        p_order, start, count = self.ranges
        records = threaded_layout(
            n_nodes, children, body, node_mass, node_com, center, half, start, count, self.BUCKET
        )
        bodies = np.empty((len(hp), 4), dtype=np.float32)
        bodies[:, :3] = hp[p_order]
        bodies[:, 3] = hm[p_order]

        if self.order is None or len(self.order) != n or self.calls % self.RESORT_EVERY == 0:
            # Massive particles first, then tracers, each along its own Morton
            # curve: GPU threads run in lockstep groups that wait for their
            # slowest member, so each group should hold one kind (one theta).
            idx_h, idx_t = np.flatnonzero(heavy), np.flatnonzero(~heavy)
            parts = [idx_h[morton_order(pos[idx_h])]]
            if len(idx_t):
                parts.append(idx_t[morton_order(pos[idx_t])])
            self.order = np.concatenate(parts)
        self.calls += 1
        order = self.order
        n_t = len(order)
        inv_theta = np.where(heavy, 1.0 / theta, 1.0 / (tracer_theta or theta))
        targets = np.empty((n_t, 4), dtype=np.float32)
        targets[:, :3] = pos[order]
        targets[:, 3] = inv_theta[order]

        self._buffer("targets", targets).bind_to_storage_buffer(0)
        self._buffer("nodes", records).bind_to_storage_buffer(1)
        self._buffer("bodies", bodies).bind_to_storage_buffer(2)
        acc_buf = self.buffers.get("acc")
        if acc_buf is None or acc_buf.size < 16 * n:
            if acc_buf is not None:
                acc_buf.release()
            acc_buf = self.ctx.buffer(reserve=32 * n)
            self.buffers["acc"] = acc_buf
        acc_buf.bind_to_storage_buffer(4)

        self.program["n_targets"].value = n_t
        self.program["n_nodes"].value = len(records)
        self.program["eps2"].value = eps * eps
        self.program["G"].value = G
        self.program.run((n_t + self.GROUP - 1) // self.GROUP)
        result = np.frombuffer(acc_buf.read(size=16 * n_t), dtype=np.float32).reshape(n_t, 4)
        out[order] = result[:, :3]
        return out

    def release(self):
        for buf in self.buffers.values():
            buf.release()
        self.buffers = {}
        self.program.release()
