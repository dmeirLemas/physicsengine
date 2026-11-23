"""Draws the atom module: shaded balls (nucleons), soft lights, electron clouds, streaks.

Written by Claude (2026-11-23, atom module). Display only, no physics.

Passes, each frame:
1. Background gradient onto the screen.
2. Everything that glows, added up in an offscreen floating-point (HDR)
   image: the cloud points, the lights, the streaks.
3. A blurred half-size copy of that image (bloom), and the composite onto
   the screen through a tone curve 1 - exp(-x), so that dense parts of a
   cloud saturate smoothly instead of clipping to white.
4. The solid balls (sphere impostors with their own depth, so that packed
   nucleons overlap correctly), drawn last: a nucleon is never washed out
   by the light around it, at the price that a light passing in front of a
   ball is hidden by it.

The cloud is uploaded once (set_cloud) and redrawn; balls and streaks are
small arrays rewritten every frame.
"""

import moderngl
import numpy as np

QUAD_VERTEX = """
#version 460 core
in vec2 in_corner;
out vec2 v_uv;
void main() {
    v_uv = in_corner * 0.5 + 0.5;
    gl_Position = vec4(in_corner, 0.0, 1.0);
}
"""

BACKGROUND_FRAGMENT = """
#version 460 core
in vec2 v_uv;
out vec4 frag_color;
void main() {
    float r = length(v_uv - vec2(0.5)) * 1.5;
    vec3 inner = vec3(0.030, 0.040, 0.075);
    vec3 outer = vec3(0.004, 0.005, 0.014);
    frag_color = vec4(mix(inner, outer, smoothstep(0.0, 1.0, r)), 1.0);
}
"""

BALL_VERTEX = """
#version 460 core
uniform mat4 view;
uniform mat4 proj;
uniform float grow;                 // quad size in ball radii (1 for solids, more for lights)
in vec2 in_corner;
in vec4 in_center_radius;
in vec4 in_color_glow;
out vec2 v_uv;
out vec3 v_center;
out float v_radius;
out vec3 v_color;
out float v_glow;
void main() {
    vec4 c = view * vec4(in_center_radius.xyz, 1.0);
    v_center = c.xyz;
    v_radius = in_center_radius.w;
    v_color = in_color_glow.rgb;
    v_glow = in_color_glow.a;
    v_uv = in_corner;
    gl_Position = proj * vec4(c.xyz + vec3(in_corner * v_radius * grow, 0.0), 1.0);
}
"""

BALL_FRAGMENT = """
#version 460 core
uniform mat4 proj;
in vec2 v_uv;
in vec3 v_center;
in float v_radius;
in vec3 v_color;
in float v_glow;
out vec4 frag_color;
void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;
    float z = sqrt(1.0 - r2);
    vec3 n = vec3(v_uv, z);
    vec4 clip = proj * vec4(v_center + n * v_radius, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    vec3 light = normalize(vec3(0.45, 0.70, 0.55));
    float diffuse = max(dot(n, light), 0.0);
    float highlight = pow(max(dot(n, normalize(light + vec3(0.0, 0.0, 1.0))), 0.0), 48.0);
    float rim = pow(1.0 - z, 3.0);
    vec3 color = v_color * (0.20 + 0.85 * diffuse) + vec3(0.40) * highlight
               + v_color * 0.45 * rim;
    float edge = fwidth(sqrt(r2));
    float alpha = 1.0 - smoothstep(1.0 - 1.5 * edge, 1.0, sqrt(r2));
    frag_color = vec4(color, alpha);
}
"""

LIGHT_FRAGMENT = """
#version 460 core
in vec2 v_uv;
in vec3 v_color;
in float v_glow;
out vec4 frag_color;
void main() {
    float d2 = dot(v_uv, v_uv);
    if (d2 > 1.0) discard;
    float fall = (1.0 - d2) * (1.0 - d2);
    float core = exp(-d2 * 26.0);
    float halo = 0.22 * exp(-d2 * 5.0);
    vec3 color = mix(v_color, vec3(1.0), 0.6 * core) * (core + halo) * fall * abs(v_glow);
    frag_color = vec4(color, 1.0);
}
"""

CLOUD_VERTEX = """
#version 460 core
uniform mat4 view_proj;
uniform float pixel_scale;          // pixels per world unit at w = 1
uniform float max_size;             // pixels
uniform float brightness;
uniform int cut;                    // 0 whole, 1 half removed, 2 thin slice
uniform vec3 cut_normal;            // unit vector toward the camera
uniform float slab;                 // half thickness of the slice, world units
in vec3 in_pos;
in vec3 in_color;
in float in_size;
out vec3 v_color;
void main() {
    gl_Position = view_proj * vec4(in_pos, 1.0);
    float px = in_size * pixel_scale / max(gl_Position.w, 1e-9);
    float fade = brightness;
    // Points that would be larger than max_size (the camera is deep inside the
    // cloud) fade out quickly instead of becoming blobs.
    if (px > max_size) { float f = max_size / px; fade *= f * f * f * f; px = max_size; }
    if (px < 2.0) { float f = px / 2.0; fade *= f * f; px = 2.0; }
    float side = dot(in_pos, cut_normal);
    if (cut == 1 && side > 0.0) fade = 0.0;
    if (cut == 2 && abs(side) > slab) fade = 0.0;
    if (fade <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = px;
    v_color = in_color * fade;
}
"""

CLOUD_FRAGMENT = """
#version 460 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = 4.0 * dot(d, d);
    if (r2 > 1.0) discard;
    frag_color = vec4(v_color * exp(-r2 * 3.0) * (1.0 - r2), 1.0);
}
"""

STREAK_VERTEX = """
#version 460 core
uniform mat4 view_proj;
uniform vec2 viewport;              // half size in pixels
in vec3 in_start;
in vec3 in_end;
in vec4 in_color;                   // rgb, width in pixels
out vec3 v_color;
out float v_across;
void main() {
    bool at_end = gl_VertexID >= 2;
    float side = (gl_VertexID % 2 == 0) ? -1.0 : 1.0;
    vec4 a = view_proj * vec4(in_start, 1.0);
    vec4 b = view_proj * vec4(in_end, 1.0);
    float w_min = 1e-6;
    if (a.w < w_min && b.w < w_min) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    if (a.w < w_min) a = mix(a, b, (w_min - a.w) / (b.w - a.w));
    if (b.w < w_min) b = mix(b, a, (w_min - b.w) / (a.w - b.w));
    vec2 dir = b.xy / b.w * viewport - a.xy / a.w * viewport;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 normal = vec2(-dir.y, dir.x);
    vec4 p = at_end ? b : a;
    p.xy += normal * side * (in_color.a + 1.0) / viewport * p.w;
    gl_Position = p;
    v_color = in_color.rgb;
    v_across = side;
}
"""

STREAK_FRAGMENT = """
#version 460 core
in vec3 v_color;
in float v_across;
out vec4 frag_color;
void main() {
    float soft = 1.0 - smoothstep(0.3, 1.0, abs(v_across));
    frag_color = vec4(v_color * soft, 1.0);
}
"""

BLUR_FRAGMENT = """
#version 460 core
uniform sampler2D source;
uniform vec2 step_uv;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec3 sum = texture(source, v_uv).rgb * 0.227027;
    sum += (texture(source, v_uv + step_uv * 1.3846).rgb
          + texture(source, v_uv - step_uv * 1.3846).rgb) * 0.316216;
    sum += (texture(source, v_uv + step_uv * 3.2308).rgb
          + texture(source, v_uv - step_uv * 3.2308).rgb) * 0.070270;
    frag_color = vec4(sum, 1.0);
}
"""

COMPOSITE_FRAGMENT = """
#version 460 core
uniform sampler2D hdr;
uniform sampler2D bloom;
uniform float exposure;
uniform float bloom_strength;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec3 light = texture(hdr, v_uv).rgb + bloom_strength * texture(bloom, v_uv).rgb;
    frag_color = vec4(vec3(1.0) - exp(-light * exposure), 1.0);
}
"""

CUT_MODES = {"Whole": 0, "Cut in half": 1, "Thin slice": 2}


class _Instances:
    """A buffer of per-instance rows that grows as needed, with its vertex arrays."""

    def __init__(self, ctx, floats, build):
        self.ctx = ctx
        self.stride = 4 * floats  # bytes per row
        self.build = build  # buffer -> [vertex arrays]
        self.capacity = 0
        self.buffer = None
        self.vaos = []
        self.count = 0

    def write(self, rows):
        self.count = len(rows)
        if self.count == 0:
            return
        if self.count > self.capacity:
            self.release()
            self.capacity = max(2 * self.count, 256)
            self.buffer = self.ctx.buffer(reserve=self.capacity * self.stride, dynamic=True)
            self.vaos = self.build(self.buffer)
        self.buffer.write(np.ascontiguousarray(rows, dtype="f4").tobytes())

    def release(self):
        for vao in self.vaos:
            vao.release()
        if self.buffer is not None:
            self.buffer.release()
        self.vaos, self.buffer, self.capacity = [], None, 0


class AtomRenderer:
    def __init__(self):
        self._gl = []
        self._targets = []
        self.size = None
        self.cloud = None  # (buffer, vertex array, count)
        self.cloud_count = 0

    def _keep(self, obj):
        self._gl.append(obj)
        return obj

    def setup(self, ctx):
        self.ctx = ctx
        program = ctx.program
        self.background_prog = self._keep(
            program(vertex_shader=QUAD_VERTEX, fragment_shader=BACKGROUND_FRAGMENT)
        )
        self.ball_prog = self._keep(
            program(vertex_shader=BALL_VERTEX, fragment_shader=BALL_FRAGMENT)
        )
        self.light_prog = self._keep(
            program(vertex_shader=BALL_VERTEX, fragment_shader=LIGHT_FRAGMENT)
        )
        self.cloud_prog = self._keep(
            program(vertex_shader=CLOUD_VERTEX, fragment_shader=CLOUD_FRAGMENT)
        )
        self.streak_prog = self._keep(
            program(vertex_shader=STREAK_VERTEX, fragment_shader=STREAK_FRAGMENT)
        )
        self.blur_prog = self._keep(
            program(vertex_shader=QUAD_VERTEX, fragment_shader=BLUR_FRAGMENT)
        )
        self.comp_prog = self._keep(
            program(vertex_shader=QUAD_VERTEX, fragment_shader=COMPOSITE_FRAGMENT)
        )
        corners = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
        self.corners = self._keep(ctx.buffer(corners.tobytes()))

        def quad(prog):
            return self._keep(ctx.vertex_array(prog, [(self.corners, "2f", "in_corner")]))

        self.background_vao = quad(self.background_prog)
        self.blur_vao = quad(self.blur_prog)
        self.comp_vao = quad(self.comp_prog)

        def ball_arrays(buffer):
            return [
                ctx.vertex_array(
                    prog,
                    [
                        (self.corners, "2f", "in_corner"),
                        (buffer, "4f 4f/i", "in_center_radius", "in_color_glow"),
                    ],
                )
                for prog in (self.ball_prog, self.light_prog)
            ]

        def streak_arrays(buffer):
            return [
                ctx.vertex_array(
                    self.streak_prog, [(buffer, "3f 3f 4f/i", "in_start", "in_end", "in_color")]
                )
            ]

        self.solids = _Instances(ctx, 8, ball_arrays)
        self.lights = _Instances(ctx, 8, ball_arrays)
        self.streaks = _Instances(ctx, 10, streak_arrays)

    def set_cloud(self, data):
        """Replace the cloud: rows x, y, z, r, g, b, size (or None for no cloud)."""
        if self.cloud is not None:
            self.cloud[1].release()
            self.cloud[0].release()
            self.cloud = None
        self.cloud_count = 0
        if data is None or len(data) == 0:
            return
        buffer = self.ctx.buffer(np.ascontiguousarray(data, dtype="f4").tobytes())
        vao = self.ctx.vertex_array(
            self.cloud_prog, [(buffer, "3f 3f 1f", "in_pos", "in_color", "in_size")]
        )
        self.cloud = (buffer, vao)
        self.cloud_count = len(data)

    def _resize(self, ctx, size):
        for obj in self._targets:
            obj.release()
        self._targets = []
        w, h = max(size[0], 1), max(size[1], 1)
        small = (max(w // 2, 1), max(h // 2, 1))

        def texture(sz):
            tex = ctx.texture(sz, 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            fbo = ctx.framebuffer(color_attachments=[tex])
            self._targets += [tex, fbo]
            return tex, fbo

        self.hdr_tex, self.hdr_fbo = texture((w, h))
        self.blur_a_tex, self.blur_a_fbo = texture(small)
        self.blur_b_tex, self.blur_b_fbo = texture(small)
        self.size, self.small = (w, h), small

    def render(
        self,
        ctx,
        camera,
        balls,
        streaks,
        cut="Whole",
        slab=1.0,
        cloud_brightness=0.12,
        cloud_max_size=48.0,
        exposure=1.0,
    ):
        """Draw one frame. balls (n, 8) and streaks (n, 10) as atom_scenes describes them."""
        target = ctx.fbo
        # The viewport, not target.size: see galaxy_renderer.render.
        size = tuple(target.viewport[2:])
        if self.size != size:
            self._resize(ctx, size)
        view = camera.view()
        proj = camera.projection()
        view_bytes = view.T.astype("f4").tobytes()
        proj_bytes = proj.T.astype("f4").tobytes()
        view_proj_bytes = (proj @ view).T.astype("f4").tobytes()

        glow = balls[:, 7] if len(balls) else np.zeros(0)
        self.solids.write(balls[glow <= 0.0])
        lit = balls[glow != 0.0]
        if len(lit):
            lit = lit.copy()
            halo = lit[:, 7] < 0.0  # a solid ball's halo: wider than the ball
            lit[halo, 3] *= 2.4
            lit[~halo, 3] *= 3.5
        self.lights.write(lit)
        self.streaks.write(streaks)

        # 1. Background, on the screen.
        target.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        target.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.background_vao.render(moderngl.TRIANGLE_STRIP)
        # 2. Everything that glows, added up in the HDR image.
        self.hdr_fbo.use()
        self.hdr_fbo.clear(0.0, 0.0, 0.0, 1.0)
        ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        if self.cloud is not None:
            p = self.cloud_prog
            p["view_proj"].write(view_proj_bytes)
            p["pixel_scale"].value = float(proj[1, 1]) * size[1] * 0.5
            p["max_size"].value = float(cloud_max_size)
            cut_mode = CUT_MODES.get(cut, 0)
            p["brightness"].value = float(cloud_brightness) * (1.0, 1.5, 3.0)[cut_mode]
            p["cut"].value = cut_mode
            to_camera = np.array(
                [
                    np.cos(camera.pitch) * np.sin(camera.yaw),
                    np.sin(camera.pitch),
                    np.cos(camera.pitch) * np.cos(camera.yaw),
                ]
            )
            p["cut_normal"].value = tuple(float(v) for v in to_camera)
            p["slab"].value = float(slab)
            self.cloud[1].render(moderngl.POINTS, vertices=self.cloud_count)
        if self.lights.count:
            self.light_prog["view"].write(view_bytes)
            self.light_prog["proj"].write(proj_bytes)
            self.light_prog["grow"].value = 1.0
            self.lights.vaos[1].render(
                moderngl.TRIANGLE_STRIP, vertices=4, instances=self.lights.count
            )
        if self.streaks.count:
            self.streak_prog["view_proj"].write(view_proj_bytes)
            self.streak_prog["viewport"].value = (size[0] / 2.0, size[1] / 2.0)
            self.streaks.vaos[0].render(
                moderngl.TRIANGLE_STRIP, vertices=4, instances=self.streaks.count
            )
        ctx.disable(moderngl.BLEND)

        # 3. Bloom: the HDR image at half size, blurred across and down, twice.
        sw, sh = self.small
        source = self.hdr_tex
        for _ in range(2):
            self.blur_b_fbo.use()
            source.use(0)
            self.blur_prog["source"].value = 0
            self.blur_prog["step_uv"].value = (1.0 / sw, 0.0)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)
            self.blur_a_fbo.use()
            self.blur_b_tex.use(0)
            self.blur_prog["step_uv"].value = (0.0, 1.0 / sh)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)
            source = self.blur_a_tex

        # 4. Composite onto the screen, added to what is there.
        target.use()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.hdr_tex.use(0)
        self.blur_a_tex.use(1)
        self.comp_prog["hdr"].value = 0
        self.comp_prog["bloom"].value = 1
        self.comp_prog["exposure"].value = float(exposure)
        self.comp_prog["bloom_strength"].value = 0.55
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)

        # 5. The solid balls, on top of the light (see the module docstring).
        if self.solids.count:
            ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ball_prog["view"].write(view_bytes)
            self.ball_prog["proj"].write(proj_bytes)
            self.ball_prog["grow"].value = 1.0
            self.solids.vaos[0].render(
                moderngl.TRIANGLE_STRIP, vertices=4, instances=self.solids.count
            )
            ctx.disable(moderngl.DEPTH_TEST)
        # Leave the usual blending for whatever draws next (the interface).
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self):
        if getattr(self, "ctx", None) is not None:
            self.set_cloud(None)
        for group in (getattr(self, name, None) for name in ("solids", "lights", "streaks")):
            if group is not None:
                group.release()
        for obj in self._gl + self._targets:
            obj.release()
        self._gl, self._targets, self.size = [], [], None
