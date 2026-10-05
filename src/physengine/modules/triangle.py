import numpy as np

from physengine.core.module import BaseModule

VERTEX = """
#version 460 core
uniform float angle;
in vec2 in_position;              // one value per vertex, from the buffer
void main() {
    mat2 rot = mat2(
        cos(angle), sin(angle),
        -sin(angle), cos(angle)
    );
    gl_Position = vec4(rot * in_position, 0.0, 1.0);   // built-in output: clip-space position
}
"""

FRAGMENT = """
#version 460 core
uniform vec3 color;               // same value for every pixel, set from Python
out vec4 frag_color;              // the pixel colour we write
void main() {
    frag_color = vec4(color, 1.0);
}
"""


class Triangle(BaseModule):
    def __init__(self) -> None:
        self.angle = 0.0
        self.angular_speed = 1.0  # radians per second

    def setup(self, ctx):
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.program["color"].value = (1.0, 0.5, 0.0)

        points = np.array([[0.5, 0], [-0.25, 0.433], [0.25, -0.433]], dtype="f4")
        self.buffer = ctx.buffer(points.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "2f", "in_position")])

    def step(self, dt):
        self.angle += self.angular_speed * dt

    def render(self, ctx, w, h):
        self.program["angle"].value = self.angle
        self.vao.render()

    def draw_ui(self): ...
    def teardown(self): ...
