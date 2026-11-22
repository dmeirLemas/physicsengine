import moderngl
import numpy as np

VERTEX = """
#version 460 core
uniform mat4 view;                // camera-target-relative view matrix
uniform mat4 proj;
uniform float viewport_height;    // pixels
uniform float min_radius_px;

in vec2 in_corner;                // quad corner, -1..1 (per vertex)
in vec3 in_position;              // body position relative to the camera target (per body)
in vec3 in_color;                 // per body
in float in_radius;               // drawn radius in world units (per body)
in float in_hole;                 // 1 for a black hole, 2 for gas (per body)

out vec3 v_color;
out vec2 v_corner;
out float v_hole;

const float GLOW = 3.0;           // the quad reaches this many radii, for the glow

void main() {
    vec4 centre = view * vec4(in_position, 1.0);

    // World radius that covers min_radius_px pixels at this depth. Using the
    // larger of the two makes bodies scale with zoom when close, but never
    // shrink to nothing when far.
    float depth = max(-centre.z, 1e-30);
    float min_radius = min_radius_px * 2.0 * depth / (proj[1][1] * viewport_height);
    float radius = max(in_radius, min_radius);

    // Billboard: offset the corner in view space so the quad faces the camera.
    centre.xy += in_corner * radius * GLOW;
    gl_Position = proj * centre;

    v_color = in_color;
    v_corner = in_corner * GLOW;
    v_hole = in_hole;
}
"""

FRAGMENT = """
#version 460 core
in vec3 v_color;
in vec2 v_corner;
in float v_hole;
out vec4 frag_color;
void main() {
    // r is 0 at the centre and 1 at the edge of the body's disc; the quad
    // goes on to 3 for the glow around it. All of this is cosmetic.
    float r = length(v_corner);
    if (r > 3.0) discard;
    float aa = 1.5 * fwidth(r);
    float disc = 1.0 - smoothstep(1.0 - aa, 1.0, r);
    float fade = 1.0 - smoothstep(2.2, 3.0, r);

    if (v_hole > 0.5 && v_hole < 1.5) {
        // A black disc, a thin bright ring at its edge (the look of light
        // bent round a hole), and a faint glow outside.
        float ring = exp(-pow((r - 1.0) / 0.07, 2.0));
        float glow = 0.35 * exp(-2.5 * max(r - 1.0, 0.0)) * (1.0 - disc) * fade;
        vec3 ring_color = mix(v_color, vec3(1.0, 0.9, 0.75), 0.75);
        vec3 color = ring_color * ring + v_color * glow;
        float alpha = max(disc, max(ring, glow));
        frag_color = vec4(color, alpha);
        return;
    }

    if (v_hole > 1.5) {
        // Gas: a soft blob with no edge, so neighbours merge into a cloud.
        float q = r / 1.3;
        frag_color = vec4(mix(v_color, vec3(1.0), 0.35 * exp(-3.0 * q * q)), 0.55 * exp(-q * q) * fade);
        return;
    }

    // A bright centre fading toward the rim, then a soft glow.
    vec3 body = mix(vec3(1.0), v_color, smoothstep(0.0, 0.7, r)) * (1.0 - 0.35 * r * r);
    float glow = 0.45 * exp(-2.6 * max(r - 1.0, 0.0)) * fade;
    vec3 color = mix(v_color, body, disc);
    frag_color = vec4(color, max(disc, glow));
}
"""


class NBodyRenderer:
    """Draws bodies as shaded discs with a per-body colour and radius.

    `colors` is a sequence of N RGB triples and `radii` a sequence of N drawn
    radii in world units, in the same order as the bodies in the state. Discs
    grow and shrink with zoom like any other geometry, but are never drawn
    smaller than `min_radius_px` pixels.
    """

    def __init__(self, colors, radii, min_radius_px=2.5, holes=None):
        self.colors = np.asarray(colors, dtype="f4").reshape(-1, 3)
        self.radii = np.asarray(radii, dtype="f4").reshape(-1)
        # How each body is drawn: 0 a body, 1 a black hole, 2 gas (default: all 0).
        self.holes = (
            np.zeros(len(self.radii), dtype="f4")
            if holes is None
            else np.asarray(holes, dtype="f4").reshape(-1)
        )
        # Black holes are drawn last, so that a body going into one
        # disappears behind its black disc.
        self.order = np.argsort(self.holes == 1.0, kind="stable")
        self.colors, self.radii, self.holes = (
            self.colors[self.order],
            self.radii[self.order],
            self.holes[self.order],
        )
        self.min_radius_px = min_radius_px

        self.N = 0
        self.program = None
        self.quad_buffer = None
        self.pos_buffer = None
        self.style_buffer = None
        self.vao = None

    def setup(self, ctx, N):
        if len(self.colors) != N or len(self.radii) != N:
            raise ValueError(
                f"need {N} colours and radii, got {len(self.colors)}, {len(self.radii)}"
            )

        self.N = N
        if N == 0:
            return  # nothing to draw; GL buffers cannot be empty
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)

        # One quad (triangle strip), instanced once per body.
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype="f4")
        self.quad_buffer = ctx.buffer(quad.tobytes())

        # Positions change every frame (3 float32 per body); colour and radius
        # are written once.
        self.pos_buffer = ctx.buffer(reserve=12 * N, dynamic=True)
        style = np.hstack([self.colors, self.radii[:, None], self.holes[:, None]]).astype("f4")
        self.style_buffer = ctx.buffer(style.tobytes(), dynamic=True)

        self.vao = ctx.vertex_array(
            self.program,
            [
                (self.quad_buffer, "2f", "in_corner"),
                (self.pos_buffer, "3f/i", "in_position"),
                (self.style_buffer, "3f 1f 1f/i", "in_color", "in_radius", "in_hole"),
            ],
        )

    def set_style(self, colors, radii, holes):
        """Replace the looks of the first len(colors) bodies (for sprites that change every frame)."""
        style = np.hstack(
            [np.asarray(colors), np.asarray(radii)[:, None], np.asarray(holes)[:, None]]
        ).astype("f4")
        self.style_buffer.write(style.tobytes())

    def render(self, ctx, camera, pos, count=None):
        """Draw the bodies at `pos`; with `count`, only the first `count` (pos has that many rows)."""
        if self.N == 0 or count == 0:
            return
        # Positions are sent relative to the camera target, subtracted in
        # float64 before the cast to float32. Otherwise float32 rounding makes
        # bodies jitter when zoomed far in on something away from the origin.
        target = camera.target
        if count is None:
            pos = pos[self.order]
        self.pos_buffer.write(np.ascontiguousarray(pos - target, dtype="f4").tobytes())

        # Matching view matrix: the camera's view with the target moved to the origin.
        view = camera.view().copy()
        view[:3, 3] += view[:3, :3] @ target

        self.program["view"].write(view.T.astype("f4").tobytes())
        self.program["proj"].write(camera.projection().T.astype("f4").tobytes())
        self.program["viewport_height"].value = ctx.viewport[3]
        self.program["min_radius_px"].value = self.min_radius_px

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, instances=self.N if count is None else count)

    def release(self):
        for obj in (self.vao, self.quad_buffer, self.pos_buffer, self.style_buffer, self.program):
            if obj is not None:
                obj.release()
        self.vao = self.quad_buffer = self.pos_buffer = self.style_buffer = self.program = None
