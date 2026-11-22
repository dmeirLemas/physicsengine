"""Star renderer for the galaxy module. Display only, no physics.

Pipeline, every frame:
1. Stars are drawn as soft points into an offscreen floating-point (HDR)
   image with additive blending, so overlapping stars add up without
   clipping: a dense bulge can be a hundred times brighter than a tidal tail.
2. Bloom: the parts of the HDR image above a brightness threshold are shrunk
   to half size and blurred (separable Gaussian, several passes), giving
   bright regions a soft glow. The threshold keeps isolated faint stars from
   blooming into blocks.
3. Composite to the screen: HDR image plus bloom, times the exposure. The
   brightness (luminance) is compressed with the tone curve 1 - exp(-x) while
   the colour is kept, so bright regions do not wash out to white; then a
   gamma of 1/2.2 lifts faint stars.

Each star has a fixed colour by population and a random brightness, so the
picture reads like a galaxy photograph rather than a uniform point cloud.
Halo particles are dark matter and hidden unless asked for.
"""

import moderngl
import numpy as np

from physengine.modules.galaxy_components import BULGE, DISC, HALO

STAR_VERTEX = """
#version 460 core
uniform mat4 view;                // camera-target-relative view matrix
uniform mat4 proj;
uniform float point_size;         // pixels
in vec3 in_position;              // relative to the camera target
in vec3 in_color;                 // colour times brightness
out vec3 v_color;
void main() {
    gl_Position = proj * view * vec4(in_position, 1.0);
    gl_PointSize = point_size;
    v_color = in_color;
}
"""

STAR_FRAGMENT = """
#version 460 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    // Gaussian spot over the point's square; r = 1 at the inscribed circle.
    float r = 2.0 * length(gl_PointCoord - vec2(0.5));
    if (r > 1.0) discard;
    frag_color = vec4(v_color * exp(-4.0 * r * r), 1.0);
}
"""

QUAD_VERTEX = """
#version 460 core
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = 0.5 * in_pos + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Shrink to half size (one bilinear tap averages a 2x2 block), keeping only
# light above the threshold.
DOWNSAMPLE_FRAGMENT = """
#version 460 core
uniform sampler2D source;
uniform float threshold;
in vec2 uv;
out vec4 frag_color;
void main() {
    vec3 c = texture(source, uv).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c *= max(lum - threshold, 0.0) / max(lum, 1e-6);
    frag_color = vec4(c, 1.0);
}
"""

# One direction of a 9-tap Gaussian (sigma ~ 2 px), using linear filtering
# to read two texels per tap.
BLUR_FRAGMENT = """
#version 460 core
uniform sampler2D source;
uniform vec2 step_uv;             // one pixel along the blur direction, in uv
in vec2 uv;
out vec4 frag_color;
void main() {
    const float w0 = 0.2270270270;
    const float w1 = 0.3162162162;
    const float w2 = 0.0702702703;
    const float o1 = 1.3846153846;
    const float o2 = 3.2307692308;
    vec3 c = texture(source, uv).rgb * w0;
    c += texture(source, uv + step_uv * o1).rgb * w1;
    c += texture(source, uv - step_uv * o1).rgb * w1;
    c += texture(source, uv + step_uv * o2).rgb * w2;
    c += texture(source, uv - step_uv * o2).rgb * w2;
    frag_color = vec4(c, 1.0);
}
"""

COMPOSITE_FRAGMENT = """
#version 460 core
uniform sampler2D hdr;
uniform sampler2D bloom;
uniform float exposure;
uniform float bloom_strength;
in vec2 uv;
out vec4 frag_color;
void main() {
    vec3 c = exposure * (texture(hdr, uv).rgb + bloom_strength * texture(bloom, uv).rgb);
    // Tone curve on the brightness only, so colours survive in bright regions.
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c *= (1.0 - exp(-lum)) / max(lum, 1e-6);
    c = min(c, vec3(1.0));
    c = pow(c, vec3(1.0 / 2.2));           // gamma: lift the faint stars
    frag_color = vec4(c, 1.0);
}
"""

# Colours by population (linear RGB, before brightness).
OLD_DISC = (1.0, 0.9, 0.75)  # most disc stars: yellow-white
YOUNG_DISC = (0.55, 0.7, 1.0)  # young, hot, bright: blue
YOUNG_FRACTION = 0.3
BULGE_COLOR = (1.0, 0.72, 0.42)  # old, cool: orange
HALO_COLOR = (0.45, 0.3, 0.65)  # dark matter, shown only on request
GALAXY_TINTS = [(1.0, 1.0, 1.0), (1.0, 0.85, 0.8), (0.85, 1.0, 0.9), (0.9, 0.9, 1.0)]

BLOOM_PASSES = 4
BLOOM_THRESHOLD = (
    1.5  # HDR brightness below which nothing glows: only dense regions, not single stars
)


def star_colors(kind, galaxy, seed=0):
    """Per-star colour times brightness, fixed for a given seed."""
    rng = np.random.default_rng(seed)
    n = len(kind)
    colors = np.empty((n, 3), dtype="f4")
    # Brightness: log-normal, so a few stars are much brighter than most.
    brightness = np.exp(rng.normal(0.0, 0.6, n))

    disc = kind == DISC
    young = disc & (rng.random(n) < YOUNG_FRACTION)
    colors[disc] = OLD_DISC
    colors[young] = YOUNG_DISC
    brightness[young] *= 2.0
    colors[kind == BULGE] = BULGE_COLOR
    colors[kind == HALO] = HALO_COLOR
    brightness[kind == HALO] *= 0.3

    colors *= np.array(GALAXY_TINTS, dtype="f4")[galaxy % len(GALAXY_TINTS)]
    return colors * brightness[:, None].astype("f4")


class GalaxyRenderer:
    def __init__(self):
        self.N = 0
        self.size = None  # size of the offscreen images, follows the screen
        self._gl = []  # every GL object, for release()

    def _keep(self, obj):
        self._gl.append(obj)
        return obj

    def setup(self, ctx, kind, galaxy):
        """`kind` and `galaxy` are per-particle component labels and galaxy indices."""
        kind = np.asarray(kind)
        galaxy = np.asarray(galaxy)
        self.N = len(kind)

        self.star_prog = self._keep(
            ctx.program(vertex_shader=STAR_VERTEX, fragment_shader=STAR_FRAGMENT)
        )
        self.down_prog = self._keep(
            ctx.program(vertex_shader=QUAD_VERTEX, fragment_shader=DOWNSAMPLE_FRAGMENT)
        )
        self.blur_prog = self._keep(
            ctx.program(vertex_shader=QUAD_VERTEX, fragment_shader=BLUR_FRAGMENT)
        )
        self.comp_prog = self._keep(
            ctx.program(vertex_shader=QUAD_VERTEX, fragment_shader=COMPOSITE_FRAGMENT)
        )

        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype="f4")
        quad_buffer = self._keep(ctx.buffer(quad.tobytes()))
        self.down_vao = self._keep(
            ctx.vertex_array(self.down_prog, [(quad_buffer, "2f", "in_pos")])
        )
        self.blur_vao = self._keep(
            ctx.vertex_array(self.blur_prog, [(quad_buffer, "2f", "in_pos")])
        )
        self.comp_vao = self._keep(
            ctx.vertex_array(self.comp_prog, [(quad_buffer, "2f", "in_pos")])
        )

        self.vao_all = self.vao_visible = None
        self.n_visible = 0
        if self.N == 0:
            return  # GL buffers cannot be empty
        self.pos_buffer = self._keep(ctx.buffer(reserve=12 * self.N, dynamic=True))
        color_buffer = self._keep(ctx.buffer(star_colors(kind, galaxy).tobytes()))
        content = [(self.pos_buffer, "3f", "in_position"), (color_buffer, "3f", "in_color")]
        self.vao_all = self._keep(ctx.vertex_array(self.star_prog, content))
        visible = np.flatnonzero(kind != HALO).astype("u4")
        self.n_visible = len(visible)
        if self.n_visible > 0:
            index = self._keep(ctx.buffer(visible.tobytes()))
            self.vao_visible = self._keep(
                ctx.vertex_array(self.star_prog, content, index_buffer=index, index_element_size=4)
            )

    def _resize(self, ctx, size):
        """(Re)create the offscreen images for a screen of `size` pixels."""
        for obj in getattr(self, "_targets", []):
            obj.release()
        self._targets = []
        w, h = max(size[0], 1), max(size[1], 1)
        small = (max(w // 2, 1), max(h // 2, 1))

        def texture(sz):
            tex = ctx.texture(sz, 4, dtype="f2")  # half floats: HDR
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self._targets.append(tex)
            fbo = ctx.framebuffer(color_attachments=[tex])
            self._targets.append(fbo)
            return tex, fbo

        self.hdr_tex, self.hdr_fbo = texture((w, h))
        self.blur_a_tex, self.blur_a_fbo = texture(small)
        self.blur_b_tex, self.blur_b_fbo = texture(small)
        self.size = (w, h)
        self.small = small

    def render(self, ctx, camera, pos, show_halo=False, point_size=2.0, exposure=1.0, bloom=0.6):
        target = ctx.fbo  # the screen (or whatever the caller had bound)
        # Size from the viewport, not target.size: for the window's default
        # framebuffer moderngl keeps .size at the size the window was created
        # with, while moderngl-window updates .viewport on every resize.
        size = tuple(target.viewport[2:])
        if self.size != size:
            self._resize(ctx, size)

        # 1. Stars into the HDR image, added up.
        self.hdr_fbo.use()
        self.hdr_fbo.clear(0.0, 0.0, 0.0, 1.0)
        if self.N > 0:
            # Relative to the camera target, subtracted in float64 before the
            # float32 cast, as in the N-body renderer.
            self.pos_buffer.write(np.ascontiguousarray(pos - camera.target, dtype="f4").tobytes())
            view = camera.view().copy()
            view[:3, 3] += view[:3, :3] @ camera.target
            self.star_prog["view"].write(view.T.astype("f4").tobytes())
            self.star_prog["proj"].write(camera.projection().T.astype("f4").tobytes())
            self.star_prog["point_size"].value = point_size

            ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE  # additive
            if show_halo:
                self.vao_all.render(mode=moderngl.POINTS, vertices=self.N)
            elif self.vao_visible is not None:
                self.vao_visible.render(mode=moderngl.POINTS, vertices=self.n_visible)
        ctx.disable(moderngl.BLEND)

        # 2. Bloom: bright parts at half size, then blur back and forth.
        sw, sh = self.small
        self.blur_a_fbo.use()
        self.hdr_tex.use(0)
        self.down_prog["source"].value = 0
        self.down_prog["threshold"].value = BLOOM_THRESHOLD
        self.down_vao.render(moderngl.TRIANGLE_STRIP)
        for _ in range(BLOOM_PASSES):
            self.blur_b_fbo.use()
            self.blur_a_tex.use(0)
            self.blur_prog["source"].value = 0
            self.blur_prog["step_uv"].value = (1.0 / sw, 0.0)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)
            self.blur_a_fbo.use()
            self.blur_b_tex.use(0)
            self.blur_prog["step_uv"].value = (0.0, 1.0 / sh)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        # 3. Composite onto the screen.
        target.use()
        self.hdr_tex.use(0)
        self.blur_a_tex.use(1)
        self.comp_prog["hdr"].value = 0
        self.comp_prog["bloom"].value = 1
        self.comp_prog["exposure"].value = exposure
        self.comp_prog["bloom_strength"].value = bloom
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)

        # Leave the usual blending for whatever draws next.
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self):
        for obj in self._gl + getattr(self, "_targets", []):
            obj.release()
        self._gl = []
        self._targets = []
        self.size = None
