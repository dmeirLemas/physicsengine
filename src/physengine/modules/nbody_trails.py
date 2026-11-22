"""Orbit trails for the N-body module. Display only, no physics.

Keeps a ring buffer of past body positions (one sample per physics step) and
draws the most recent part of it as one fading line per body.
"""

import moderngl
import numpy as np

VERTEX = """
#version 460 core
uniform mat4 view;                // camera-target-relative view matrix
uniform mat4 proj;
in vec3 in_position;              // relative to the camera target
in float in_alpha;                // 0 at the old end of the trail, 1 at the body
out float v_alpha;
void main() {
    gl_Position = proj * view * vec4(in_position, 1.0);
    v_alpha = in_alpha;
}
"""

FRAGMENT = """
#version 460 core
uniform vec3 color;
in float v_alpha;
out vec4 frag_color;
void main() {
    frag_color = vec4(color, 0.8 * v_alpha * v_alpha);
}
"""


class TrailRenderer:
    def __init__(self, n_bodies, capacity):
        self.capacity = max(int(capacity), 1)

        # history[k, i] is the position of body i at sample k (float64).
        self.history = np.zeros((self.capacity, n_bodies, 3))
        self.head = 0  # index the next sample is written to
        self.count = 0  # samples stored so far, at most capacity
        # Samples stored since each body exists. Bodies added later have fewer.
        self.valid = np.zeros(n_bodies, dtype=int)

        self.program = None
        self.buffer = None
        self.vao = None

    def setup(self, ctx):
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)

    def record(self, pos):
        """Store the current positions. Call once per physics step."""
        self.history[self.head] = pos
        self.head = (self.head + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        self.valid = np.minimum(self.valid + 1, self.capacity)

    def add_bodies(self, k):
        """Make room for k more bodies, which start with no trail."""
        self.history = np.concatenate([self.history, np.zeros((self.capacity, k, 3))], axis=1)
        self.valid = np.concatenate([self.valid, np.zeros(k, dtype=int)])

    def remove_body(self, i):
        """Drop body i's trail."""
        self.history = np.delete(self.history, i, axis=1)
        self.valid = np.delete(self.valid, i)

    def forget(self, i):
        """Clear body i's trail, for example after it was moved by hand."""
        self.valid[i] = 0

    def render(self, ctx, camera, pos, mass, colors, n_points, follow=None):
        """Draw the last `n_points` samples of every body's trail, ending at
        its current position `pos`.

        `follow` is None, a body index or "com". When set, trails are drawn
        as seen from that body or from the centre of mass (computed with the
        current masses `mass`): each past position is shifted by how far the
        followed point has moved since.
        """
        n_bodies = len(pos)
        if n_bodies == 0:
            return

        n = min(int(n_points), self.count)
        if follow == "com":
            # The centre of mass needs every body's past position.
            n = min(n, int(self.valid.min()))
        elif follow is not None:
            n = min(n, int(self.valid[follow]))
        if n <= 0:
            return

        # The n most recent samples, oldest first, then the current positions.
        idx = (self.head - n + np.arange(n)) % self.capacity
        points = np.concatenate([self.history[idx], pos[None]], axis=0)  # (n + 1, N, 3)

        if follow == "com":
            total = mass.sum()
            if total > 0.0:
                com = np.einsum("n,knd->kd", mass, points) / total  # (n + 1, 3)
                points = points - com[:, None] + com[-1]
        elif follow is not None:
            points = points - points[:, [follow]] + pos[follow]

        n_verts = n + 1

        # Fade measured against the requested length, so a trail that is still
        # growing does not change brightness along its length as it grows.
        alpha = (np.arange(n_verts) + (n_points - n)) / max(n_points, 1)

        # One contiguous run of vertices per body: (N, n + 1, 4). Positions are
        # made relative to the camera target in float64 before the float32 cast.
        data = np.empty((n_bodies, n_verts, 4), dtype="f4")
        data[:, :, :3] = np.swapaxes(points - camera.target, 0, 1)
        data[:, :, 3] = alpha
        raw = data.tobytes()

        if self.buffer is None or self.buffer.size < len(raw):
            if self.vao is not None:
                self.vao.release()
                self.buffer.release()
            self.buffer = ctx.buffer(reserve=len(raw), dynamic=True)
            self.vao = ctx.vertex_array(
                self.program, [(self.buffer, "3f 1f", "in_position", "in_alpha")]
            )
        self.buffer.write(raw)

        view = camera.view().copy()
        view[:3, 3] += view[:3, :3] @ camera.target
        self.program["view"].write(view.T.astype("f4").tobytes())
        self.program["proj"].write(camera.projection().T.astype("f4").tobytes())

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        for i in range(n_bodies):
            # Skip samples from before this body existed.
            n_body = min(int(self.valid[i]), n) + 1
            if n_body < 2:
                continue
            self.program["color"].value = tuple(colors[i])
            self.vao.render(
                mode=moderngl.LINE_STRIP,
                vertices=n_body,
                first=i * n_verts + (n_verts - n_body),
            )

    def release(self):
        for obj in (self.vao, self.buffer, self.program):
            if obj is not None:
                obj.release()
        self.vao = self.buffer = self.program = None
