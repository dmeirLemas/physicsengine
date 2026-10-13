import numpy as np

from physengine.core.module import BaseModule
from physengine.core.spec import SliderFloat

VERTEX = """
#version 460 core
uniform float angle;
uniform mat4 view_proj;
in vec2 in_position;              // one value per vertex, from the buffer
void main() {
    mat2 rot = mat2(
        cos(angle), sin(angle),
        -sin(angle), cos(angle)
    );
    gl_Position = view_proj * vec4(rot * in_position, 0.0, 1.0);
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

        self.specs = (
            SliderFloat(
                attr_name="angular_speed",
                label="Angular Speed",
                default=1.0,
                max=10.0,
                min=0.0,
            ),
        )

    def setup(self, ctx):
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.program["color"].value = (1.0, 0.5, 0.0)

        points = np.array([[0.5, 0], [-0.25, 0.433], [0.25, -0.433]], dtype="f4")
        self.buffer = ctx.buffer(points.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "2f", "in_position")])

    def step(self, dt):
        self.angle += self.angular_speed * dt

    def render(self, ctx, camera):
        self.program["angle"].value = self.angle

        view_proj = camera.view_proj()
        self.program["view_proj"].write(view_proj.T.astype("f4").tobytes())

        self.vao.render()

    def teardown(self): ...
