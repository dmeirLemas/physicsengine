import moderngl
import numpy as np

VERTEX = """
#version 460 core
uniform mat4 view;                // camera-target-relative view matrix
uniform mat4 proj;
uniform float viewport_height;    // pixels
uniform float min_radius_px;
uniform float hole_min_radius_px; // black holes are never drawn smaller than this

in vec2 in_corner;                // quad corner, -1..1 (per vertex)
in vec3 in_position;              // body position relative to the camera target (per body)
in vec3 in_color;                 // per body
in float in_radius;               // drawn radius in world units (per body)
in float in_hole;                 // 1 for a black hole, 2 for gas (per body)
in vec3 in_axis;                  // gas: world vector the blob is stretched along (per body)
in float in_look;                 // 1 normal; towards 0 as its light is redshifted away (per body)
in float in_alpha;                // opacity factor (per body)

out vec3 v_color;
out vec2 v_corner;
out float v_hole;
out float v_look;
out float v_alpha;

const float GLOW = 3.0;           // the quad reaches this many radii, for the glow
const float GAS_MIN_PX = 1.0;

void main() {
    vec4 centre = view * vec4(in_position, 1.0);

    // World radius that covers min_radius_px pixels at this depth. Using the
    // larger of the two makes bodies scale with zoom when close, but never
    // shrink to nothing when far.
    float depth = max(-centre.z, 1e-30);
    // (Gas is many overlapping blobs: its minimum is small, or a distant
    // cloud would be drawn far larger than the star it came from.)
    float min_px = in_hole > 1.5 ? GAS_MIN_PX
                 : in_hole > 0.5 ? hole_min_radius_px : min_radius_px;
    float min_radius = min_px * 2.0 * depth / (proj[1][1] * viewport_height);
    float radius = max(in_radius, min_radius);

    // Billboard: offset the corner in view space so the quad faces the camera.
    // A stretched blob is longer by |axis| along the axis as seen on screen.
    vec2 axis = (view * vec4(in_axis, 0.0)).xy;
    float stretch = length(axis);
    vec2 along = stretch > 0.0 ? axis / stretch : vec2(1.0, 0.0);
    vec2 across = vec2(-along.y, along.x);
    centre.xy += (in_corner.x * (radius + stretch) * along + in_corner.y * radius * across) * GLOW;
    gl_Position = proj * centre;

    v_color = in_color;
    v_corner = in_corner * GLOW;
    v_hole = in_hole;
    v_look = in_look;
    v_alpha = in_alpha;
}
"""

FRAGMENT = """
#version 460 core
in vec3 v_color;
in vec2 v_corner;
in float v_hole;
in float v_look;
in float v_alpha;
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
        // (Stretched along a stream, see in_axis, so that a row of pieces
        // reads as one strand.)
        float q = r / 1.3;
        float white = 0.35 * exp(-3.0 * q * q) * v_look * v_look;
        frag_color = vec4(mix(v_color, vec3(1.0), white), 0.55 * exp(-q * q) * fade * v_alpha);
        return;
    }

    // A bright centre fading toward the rim, then a soft glow.
    // The white centre goes with the light: a body whose light is
    // redshifted away (v_look < 1) keeps only its (reddened) colour.
    vec3 centre = mix(v_color, vec3(1.0), v_look * v_look);
    vec3 body = mix(centre, v_color, smoothstep(0.0, 0.7, r)) * (1.0 - 0.35 * r * r);
    float glow = 0.45 * exp(-2.6 * max(r - 1.0, 0.0)) * fade;
    vec3 color = mix(v_color, body, disc);
    frag_color = vec4(color, max(disc, glow) * v_alpha);
}
"""


# Gas (display only). A piece is drawn as a blob stretched along the local
# direction of the gas around it, so that a row of pieces reads as one strand
# and not as dots.
GAS_NEIGHBOURS = 6  # pieces that set a blob's direction and length
GAS_WIDTH = 4.0  # a blob is at most this many times wider than its own drawn radius
GAS_STRETCH = 0.5  # half-length added along the strand / rms distance of the neighbours along it
GAS_LINK = 80.0  # no stretching where the neighbours are further than this many drawn radii
GAS_FAINT = 0.3  # least opacity factor of a stretched blob


def gas_strands(pos, base):
    """How to draw pieces of gas at `pos` (K, 3) with drawn radii `base` (K).

    Returns (width, axis, opacity): the blob's radius across the strand,
    the world vector it is stretched along (zero for a round blob) and an
    opacity factor. The axis is the direction along which the piece's
    GAS_NEIGHBOURS nearest neighbours are spread out most (the principal
    axis of their offsets), with a length in proportion to that spread: in
    a ball of gas the blobs stay nearly round, in a stream they join end to
    end. The blob is wider where the gas is spread out sideways too (up to
    GAS_WIDTH) and fainter the more area it covers, so that thin gas reads
    as a faint strand.
    """
    k = len(pos)
    base = np.asarray(base, dtype=float)
    n = min(GAS_NEIGHBOURS, k - 1)
    if n < 2:
        return base.copy(), np.zeros((k, 3)), np.ones(k)
    p = pos - pos.mean(axis=0)
    sq = np.sum(p * p, axis=1)
    d2 = sq[:, None] + sq[None] - 2.0 * (p @ p.T)
    np.fill_diagonal(d2, np.inf)
    nearest = np.argpartition(d2, n - 1, axis=1)[:, :n]
    offsets = p[nearest] - p[:, None]  # (K, n, 3)
    spread, vectors = np.linalg.eigh(np.einsum("kni,knj->kij", offsets, offsets) / n)
    along = np.sqrt(np.maximum(spread[:, 2], 0.0))  # rms offset along the principal axis
    across = np.sqrt(np.maximum(spread[:, 1], 0.0))
    length = np.where(along < GAS_LINK * base, GAS_STRETCH * along, 0.0)
    width = np.clip(across, base, GAS_WIDTH * base)
    opacity = np.clip(base / np.sqrt(width * (width + length)), GAS_FAINT, 1.0)
    return width, vectors[:, :, 2] * length[:, None], opacity


STYLE_FLOATS = 10  # colour (3), radius, kind, axis (3), look, alpha


class NBodyRenderer:
    """Draws bodies as shaded discs with a per-body colour and radius.

    `colors` is a sequence of N RGB triples and `radii` a sequence of N drawn
    radii in world units, in the same order as the bodies in the state. Discs
    grow and shrink with zoom like any other geometry, but are never drawn
    smaller than `min_radius_px` pixels.
    """

    def __init__(self, colors, radii, min_radius_px=2.5, holes=None, hole_min_radius_px=None):
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
        # Black holes get their own minimum (a display minimum: the horizon
        # of a small hole is far below a pixel).
        self.hole_min_radius_px = (
            min_radius_px if hole_min_radius_px is None else hole_min_radius_px
        )

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
        self.style_buffer = ctx.buffer(reserve=4 * STYLE_FLOATS * N, dynamic=True)
        self.set_style(self.colors, self.radii, self.holes)

        self.vao = ctx.vertex_array(
            self.program,
            [
                (self.quad_buffer, "2f", "in_corner"),
                (self.pos_buffer, "3f/i", "in_position"),
                (
                    self.style_buffer,
                    "3f 1f 1f 3f 1f 1f/i",
                    "in_color",
                    "in_radius",
                    "in_hole",
                    "in_axis",
                    "in_look",
                    "in_alpha",
                ),
            ],
        )

    def set_style(self, colors, radii, holes, axes=None, look=None, alpha=None):
        """Replace the looks of the first len(colors) bodies (for sprites that change every frame).

        `axes`: per body, the world vector a gas blob is stretched along
        (default none). `look`: 1 for a normal body, towards 0 for one whose
        light is redshifted away. `alpha`: opacity factor (default 1).
        """
        n = len(radii)
        style = np.zeros((n, STYLE_FLOATS), dtype="f4")
        style[:, 0:3] = np.asarray(colors).reshape(n, 3)
        style[:, 3] = radii
        style[:, 4] = holes
        if axes is not None:
            style[:, 5:8] = axes
        style[:, 8] = 1.0 if look is None else look
        style[:, 9] = 1.0 if alpha is None else alpha
        self.style_buffer.write(style.tobytes())

    def render(self, ctx, camera, pos, count=None, without_holes=False):
        """Draw the bodies at `pos`; with `count`, only the first `count` (pos
        has that many rows). With without_holes the black holes are left
        out (they are last in the drawing order), for a caller that draws
        them separately on top of something else."""
        instances = self.N if count is None else count
        if without_holes:
            instances -= int(np.count_nonzero(self.holes == 1.0))
        if self.N == 0 or instances <= 0:
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
        self.program["hole_min_radius_px"].value = self.hole_min_radius_px

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, instances=instances)

    def release(self):
        for obj in (self.vao, self.quad_buffer, self.pos_buffer, self.style_buffer, self.program):
            if obj is not None:
                obj.release()
        self.vao = self.quad_buffer = self.pos_buffer = self.style_buffer = self.program = None
