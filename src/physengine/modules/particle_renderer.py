"""Draws the particle-physics module: detector outline, tracks that grow in time, deposits.

Display only. Everything is a set of line segments drawn as camera-facing
ribbons. Each end of a segment carries the time at which the particle was
there; the shader shows only the part already travelled, brightest just
behind the particle. Static things (the detector) carry a time far in the
past.
"""

import moderngl
import numpy as np

from physengine.modules.particle_tracks import (
    ECAL_RADIUS,
    ECAL_Z,
    HCAL_OUTER,
    HCAL_RADIUS,
    HCAL_Z,
    MUON_LAYERS,
    OUTER_Z,
    SOLENOID_RADIUS,
    TRACKER_RADIUS,
)

ALWAYS = -1.0e9  # "time" of static geometry

VERTEX = """
#version 460 core
uniform mat4 view_proj;
uniform vec2 viewport;      // pixels
uniform float now;          // ns
uniform float head_glow;    // ns over which the head's extra brightness dies away

in vec4 in_start;           // xyz, time
in vec4 in_end;
in vec4 in_color;           // rgb, width in pixels (negative: shown only briefly, see fragment)

out vec3 v_color;
out float v_time;
out float v_across;
out float v_transient;

void main() {
    // Corner of the quad from the vertex number: 0,1 at the start, 2,3 at the end.
    bool at_end = gl_VertexID >= 2;
    float side = (gl_VertexID % 2 == 0) ? -1.0 : 1.0;
    vec4 a = view_proj * vec4(in_start.xyz, 1.0);
    vec4 b = view_proj * vec4(in_end.xyz, 1.0);
    // Keep both ends in front of the camera so the screen direction is defined.
    float w_min = 1e-4;
    if (a.w < w_min && b.w < w_min) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    if (a.w < w_min) a = mix(a, b, (w_min - a.w) / (b.w - a.w));
    if (b.w < w_min) b = mix(b, a, (w_min - b.w) / (a.w - b.w));
    vec2 pa = a.xy / a.w * viewport;
    vec2 pb = b.xy / b.w * viewport;
    vec2 dir = pb - pa;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 normal = vec2(-dir.y, dir.x);
    float t = at_end ? in_end.w : in_start.w;
    float age = now - t;
    v_transient = in_color.a < 0.0 ? 1.0 : 0.0;
    float width = abs(in_color.a) * (1.0 + 1.5 * exp(-max(age, 0.0) / head_glow));
    vec4 p = at_end ? b : a;
    p.xy += normal * side * (width + 1.0) / viewport * p.w;
    gl_Position = p;
    v_color = in_color.rgb;
    v_time = t;
    v_across = side;
}
"""

FRAGMENT = """
#version 460 core
uniform float now;
uniform float head_glow;
uniform float fade;         // 1: fully shown; goes to 0 as the event fades out
in vec3 v_color;
in float v_time;
in float v_across;
in float v_transient;
out vec4 frag_color;
void main() {
    float age = now - v_time;
    if (age < 0.0) discard;                       // not travelled yet
    float soft = 1.0 - smoothstep(0.35, 1.0, abs(v_across));
    float head = exp(-age / head_glow);
    vec3 color = mix(v_color, vec3(1.0), 0.7 * head) * (0.75 + 1.5 * head);
    float alpha = soft * (v_time < -1.0e8 ? 1.0 : fade);
    // Things that only pass (the incoming protons, the flash): gone soon after.
    if (v_transient > 0.5) alpha *= exp(-age / 0.5);
    frag_color = vec4(color * alpha, alpha);
}
"""


def _circle(radius, z, n=96):
    a = np.linspace(0.0, 2.0 * np.pi, n + 1)
    pts = np.stack([radius * np.cos(a), radius * np.sin(a), np.full_like(a, z)], 1)
    return pts[:-1], pts[1:]


def detector_segments():
    """(starts, ends, colours with width) of the detector outline."""
    starts, ends, colors = [], [], []

    def add(a, b, rgb, width):
        starts.append(a)
        ends.append(b)
        colors.append(np.tile([*rgb, width], (len(a), 1)))

    def cylinder(radius, half_z, rgb, width=1.0, rings=(-1.0, 0.0, 1.0), lines=12):
        for f in rings:
            add(*_circle(radius, f * half_z), rgb, width)
        ang = np.linspace(0.0, 2.0 * np.pi, lines, endpoint=False)
        x, y = radius * np.cos(ang), radius * np.sin(ang)
        add(
            np.stack([x, y, np.full(lines, -half_z)], 1),
            np.stack([x, y, np.full(lines, half_z)], 1),
            rgb,
            width,
        )

    cylinder(TRACKER_RADIUS, ECAL_Z - 0.2, (0.10, 0.13, 0.22))
    cylinder(ECAL_RADIUS, ECAL_Z, (0.05, 0.20, 0.10))
    cylinder(HCAL_RADIUS, HCAL_Z, (0.06, 0.12, 0.26))
    cylinder(HCAL_OUTER, HCAL_Z, (0.06, 0.12, 0.26), rings=(-1.0, 1.0), lines=0 or 1)
    cylinder(SOLENOID_RADIUS, OUTER_Z * 0.9, (0.22, 0.20, 0.12), rings=(-1.0, 1.0), lines=8)
    for layer in MUON_LAYERS:
        cylinder(layer, OUTER_Z, (0.20, 0.07, 0.07), rings=(-1.0, 1.0), lines=8)
    # The beam line.
    add(
        np.array([[0.0, 0.0, -OUTER_Z * 1.4]]),
        np.array([[0.0, 0.0, OUTER_Z * 1.4]]),
        (0.25, 0.25, 0.3),
        1.0,
    )
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(colors)


class ParticleRenderer:
    def __init__(self):
        self.program = None
        self.static = None  # (buffer, vao, count)
        self.event = None

    def setup(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        a, b, c = detector_segments()
        self.static = self._upload(a, np.full(len(a), ALWAYS), b, np.full(len(b), ALWAYS), c)

    def _upload(self, starts, t_start, ends, t_end, colors):
        n = len(starts)
        if n == 0:
            return None
        data = np.hstack(
            [starts, np.asarray(t_start)[:, None], ends, np.asarray(t_end)[:, None], colors]
        )
        buffer = self.ctx.buffer(data.astype("f4").tobytes())
        vao = self.ctx.vertex_array(
            self.program, [(buffer, "4f 4f 4f/i", "in_start", "in_end", "in_color")]
        )
        return buffer, vao, n

    def set_event(self, starts, t_start, ends, t_end, colors):
        """Replace the event's segments (arrays as in detector_segments, plus times)."""
        self._release(self.event)
        self.event = self._upload(starts, t_start, ends, t_end, colors)

    def _release(self, item):
        if item is not None:
            item[1].release()
            item[0].release()

    def render(self, ctx, camera, now, fade, show_detector):
        view_proj = camera.view_proj()
        p = self.program
        p["view_proj"].write(view_proj.T.astype("f4").tobytes())
        p["viewport"].value = (ctx.viewport[2] / 2.0, ctx.viewport[3] / 2.0)
        p["now"].value = float(now)
        p["head_glow"].value = 0.35
        p["fade"].value = float(fade)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE  # additive: light on black
        for item, shown in ((self.static, show_detector), (self.event, True)):
            if item is not None and shown:
                item[1].render(mode=moderngl.TRIANGLE_STRIP, vertices=4, instances=item[2])
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self):
        self._release(self.static)
        self._release(self.event)
        self.static = self.event = None
        if self.program is not None:
            self.program.release()
            self.program = None
