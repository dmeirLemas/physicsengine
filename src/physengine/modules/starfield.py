"""Decorative star background for space scenes. Display only, no physics.

Stars are random directions on a sphere around the camera. Only the camera's
rotation is applied to them, so they turn when the camera orbits but do not
move when it pans or zooms, as if infinitely far away.
"""

import moderngl
import numpy as np

VERTEX = """
#version 460 core
uniform mat4 view_proj;           // rotation-only view, times projection
in vec3 in_direction;
in vec4 in_color;                 // rgb tint, a = brightness
in float in_size;
out vec4 v_color;
void main() {
    // Force the depth onto the far plane: the stars are a backdrop, so they
    // must not be clipped whatever the camera's near and far planes are.
    vec4 p = view_proj * vec4(in_direction, 1.0);
    gl_Position = vec4(p.xy, p.w * 0.9999, p.w);
    gl_PointSize = in_size;
    v_color = in_color;
}
"""

FRAGMENT = """
#version 460 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    float r = 2.0 * length(gl_PointCoord - vec2(0.5));
    if (r > 1.0) discard;
    frag_color = vec4(v_color.rgb, v_color.a * (1.0 - r * r));
}
"""


class Starfield:
    def __init__(self, n_stars=2500, seed=7, clear_color=(0.0, 0.0, 0.015)):
        self.n_stars = n_stars
        self.seed = seed
        self.clear_color = clear_color

        self.program = None
        self.buffer = None
        self.vao = None

    def setup(self, ctx):
        rng = np.random.default_rng(self.seed)
        n = self.n_stars

        # Uniform directions on the sphere: normalised Gaussian vectors.
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)

        # Mostly faint stars with a few bright ones.
        brightness = 0.15 + 0.85 * rng.random(n) ** 4
        size = 1.5 + 2.5 * brightness

        # Tint between bluish and yellowish white.
        t = rng.random((n, 1))
        tint = (1 - t) * np.array([0.75, 0.85, 1.0]) + t * np.array([1.0, 0.9, 0.75])

        data = np.hstack([direction, tint, brightness[:, None], size[:, None]]).astype("f4")

        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.buffer = ctx.buffer(data.tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.buffer, "3f 4f 1f", "in_direction", "in_color", "in_size")],
        )

    def render(self, ctx, camera):
        """Clear to the space colour and draw the stars. Call before the scene."""
        ctx.clear(*self.clear_color)

        view = camera.view().copy()
        view[:3, 3] = 0.0  # drop the translation, keep the rotation
        view_proj = camera.projection() @ view
        self.program["view_proj"].write(view_proj.T.astype("f4").tobytes())

        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(mode=moderngl.POINTS)

    def release(self):
        for obj in (self.vao, self.buffer, self.program):
            if obj is not None:
                obj.release()
        self.vao = self.buffer = self.program = None
