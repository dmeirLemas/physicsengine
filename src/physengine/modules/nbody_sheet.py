"""The "spacetime sheet" of the N-body module: a grid that dips at the masses and ripples with their gravitational waves.

Display. Two things are drawn on a grid in the plane z = 0:

- Dips at the bodies, deeper for heavier ones. This is the familiar rubber-
  sheet picture: an illustration of "mass curves spacetime", not a
  computed quantity. Its depth and width are chosen for looks and follow the
  zoom. Depth goes as mass^(1/4), so that a planet's dip still shows next to
  its star's (in true proportion it would be thousands of times shallower);
  lighter bodies also get a narrower dip.
- Gravitational waves, from the quadrupole formula (postnewtonian.py). An
  observer in the orbital plane at distance r, in direction n from the
  centre of mass, sees the "plus" polarisation
      h_+ = G / (c^4 r) [e_phi . Qdd . e_phi - Qdd_zz],   e_phi = z x n,
  with Qdd the second time derivative of the trace-free quadrupole at the
  retarded time t - r / c. The sheet is lifted by h_+, so the pattern is the
  real one: it leaves at the speed of light, has twice the orbital
  frequency, and grows and tightens as a binary spirals in. The formula
  holds far from the source (r larger than a wavelength / 2 pi); nearer, r is
  softened and the picture is only indicative.

Height scale of the waves: what is drawn is r h_+, the strain with its 1 / r
fall-off taken out, so that ripples keep their height on the way out and the
growth of the signal itself (the "chirp") is what shows. The largest r h a
binary can reach is of order G M / (12 c^2) (at merger, v^2 ~ c^2 / 6); that
is drawn WAVE_HEIGHT of the sheet's extent high, times the user's gain.
Weak-field systems (planets) therefore show nothing at gain 1: their waves
really are ~1e-8 of a merger's.
"""

import moderngl
import numpy as np

HISTORY = 2048  # steps of quadrupole history kept
MAX_BODIES = 16  # dips drawn for the heaviest ones
LINES = 128  # grid lines each way
SEGMENTS = 384  # per line
# Heights and widths are fractions of the sheet's "extent", a length that
# follows the camera distance smoothly. (The grid's half-width snaps to powers
# of two to keep the lines still; heights tied to it would jump on zooming.)
EXTENT = 1.6 * 2.0**0.5  # extent / camera distance: the middle of the half-width's range
WAVE_HEIGHT = 0.075
WELL_DEPTH = 0.05
WELL_WIDTH = 0.04
WELL_EXPONENT = 0.25  # dip depth ~ (mass / heaviest mass) ^ this
WELL_NARROWEST = 0.5  # width of the lightest bodies' dips, as a fraction of the heaviest's

VERTEX = """
#version 460 core
uniform mat4 view;
uniform mat4 proj;
uniform vec2 centre;         // sheet centre relative to the camera target
uniform float half_width;
uniform float extent;        // length that heights and widths scale with (follows the zoom smoothly)
uniform vec3 body[16];       // xy relative to the camera target, z = relative dip depth (0..1)
uniform int n_bodies;
uniform sampler2D history;   // Qdd_xx, Qdd_xy, Qdd_yy, Qdd_zz per step, G / c^4 included
uniform float head;          // index of the newest sample
uniform float filled;        // number of valid samples
uniform float light_step;    // distance light travels per step
uniform vec2 com;            // centre of mass relative to the camera target
uniform float wave_scale;    // sheet height per unit of r h
uniform float well_depth;
uniform float well_width;
uniform float well_narrowest;

in vec2 in_grid;             // -1..1
out float v_wave;
out float v_edge;

vec4 sample_history(float index) {
    int size = textureSize(history, 0).x;
    float i0 = floor(index);
    vec4 a = texelFetch(history, ivec2(int(mod(i0, float(size))), 0), 0);
    vec4 b = texelFetch(history, ivec2(int(mod(i0 + 1.0, float(size))), 0), 0);
    return mix(a, b, index - i0);
}

void main() {
    vec2 p = centre + in_grid * half_width;
    float z = 0.0;
    for (int i = 0; i < n_bodies; i++) {
        float d = length(p - body[i].xy);
        float w = well_width * mix(well_narrowest, 1.0, body[i].z);
        z -= well_depth * body[i].z * w / sqrt(d * d + w * w);
    }
    vec2 rel = p - com;
    float r = length(rel);
    float delay = r / light_step;
    float wave = 0.0;
    if (delay < filled - 2.0 && r > 0.0) {
        vec4 q = sample_history(head - delay);
        vec2 e = vec2(-rel.y, rel.x) / r;
        // r h_+, faded out close to the source where the formula does not hold.
        float soft = 0.03 * half_width;
        wave = (q.x * e.x * e.x + 2.0 * q.y * e.x * e.y + q.z * e.y * e.y - q.w) * r * r / (r * r + soft * soft);
    }
    v_wave = wave * wave_scale;
    z += v_wave;
    v_edge = length(in_grid);
    gl_Position = proj * view * vec4(p, z - 0.02 * extent, 1.0);
}
"""

FRAGMENT = """
#version 460 core
uniform float extent;
in float v_wave;
in float v_edge;
out vec4 frag_color;
void main() {
    float w = clamp(v_wave / (0.03 * extent), -1.0, 1.0);
    vec3 base = vec3(0.20, 0.32, 0.62);
    vec3 crest = vec3(1.0, 0.72, 0.30);
    vec3 trough = vec3(0.30, 0.85, 1.0);
    vec3 color = mix(base, w > 0.0 ? crest : trough, abs(w));
    float alpha = (0.13 + 0.55 * abs(w)) * (1.0 - smoothstep(0.5, 1.0, v_edge));
    frag_color = vec4(color, alpha);
}
"""


class SpacetimeSheet:
    def __init__(self):
        self.program = None
        self.q = np.zeros((HISTORY, 4), dtype="f4")
        self.head = -1  # index of the newest sample (grows without wrapping)

    def setup(self, ctx):
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        t = np.linspace(-1.0, 1.0, SEGMENTS + 1)
        lines = []
        for c in np.linspace(-1.0, 1.0, LINES + 1):
            a = np.stack([t, np.full_like(t, c)], 1)
            for pts in (a, a[:, ::-1]):
                lines.append(np.stack([pts[:-1], pts[1:]], 1).reshape(-1, 2))
        grid = np.concatenate(lines).astype("f4")
        self.buffer = ctx.buffer(grid.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "2f", "in_grid")])
        self.texture = ctx.texture((HISTORY, 1), 4, dtype="f4")
        self.texture.filter = moderngl.NEAREST, moderngl.NEAREST

    def record(self, qdd, G, c):
        """Store this step's quadrupole second derivative (3x3, see postnewtonian.quadrupole_ddot)."""
        self.head += 1
        k = G / c**4
        self.q[self.head % HISTORY] = (k * qdd[0, 0], k * qdd[0, 1], k * qdd[1, 1], k * qdd[2, 2])

    def clear(self):
        self.q[:] = 0.0
        self.head = -1

    def render(self, ctx, camera, pos, mass, light_step, strain_unit, gain):
        """Draw the sheet. `light_step` is c dt; `strain_unit` is G M / c^2 of the
        whole system (sets the height scale, see the module docstring)."""
        if self.program is None or len(mass) == 0 or mass.max() <= 0.0:
            return
        target = camera.target
        # Width and position snap to a grid, so the lines stay put in space
        # while the camera moves and zooms.
        half_width = 2.0 ** np.ceil(np.log2(1.6 * camera.distance))
        cell = 2.0 * half_width / LINES
        extent = EXTENT * camera.distance
        centre = np.round(target[:2] / cell) * cell - target[:2]

        heavy = np.argsort(mass)[::-1][:MAX_BODIES]
        bodies = np.zeros((MAX_BODIES, 3), dtype="f4")
        bodies[: len(heavy), :2] = pos[heavy, :2] - target[:2]
        bodies[: len(heavy), 2] = (mass[heavy] / mass.max()) ** WELL_EXPONENT
        com = mass @ pos[:, :2] / mass.sum() - target[:2]

        view = camera.view().copy()
        view[:3, 3] += view[:3, :3] @ target
        p = self.program
        p["view"].write(view.T.astype("f4").tobytes())
        p["proj"].write(camera.projection().T.astype("f4").tobytes())
        p["centre"].value = tuple(float(v) for v in centre)
        p["half_width"].value = float(half_width)
        p["extent"].value = float(extent)
        p["body"].write(bodies.tobytes())
        p["n_bodies"].value = len(heavy)
        p["head"].value = float(max(self.head, 0) % HISTORY + HISTORY)
        p["filled"].value = float(min(self.head + 1, HISTORY))
        p["light_step"].value = float(light_step)
        p["com"].value = tuple(float(v) for v in com)
        reference = strain_unit / 12.0  # largest r h
        p["wave_scale"].value = float(gain * WAVE_HEIGHT * extent / max(reference, 1e-300))
        p["well_depth"].value = float(WELL_DEPTH * extent)
        p["well_width"].value = float(WELL_WIDTH * extent)
        p["well_narrowest"].value = WELL_NARROWEST
        self.texture.write(self.q.tobytes())
        self.texture.use(0)
        p["history"].value = 0

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.vao.render(mode=moderngl.LINES)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self):
        for name in ("vao", "buffer", "texture", "program"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.release()
        self.program = None
