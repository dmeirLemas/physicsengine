"""Ray-traced image of a Schwarzschild black hole with a thin accretion disc.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). The shader computes physics (the light paths and the disc's redshift);
review it with schwarzschild.py, which has the formulas and references.

Every pixel follows its light ray backwards from the camera, in units
G = c = M = 1 with the hole at the origin:

1. The camera is a static observer at r0 = |camera position|. A pixel's ray
   leaves at angle alpha from the outward radial direction; it moves in the
   plane spanned by the radial direction e_r and the sideways part e_t of the
   ray direction, with impact parameter b = r0 sin(alpha) / sqrt(1 - 2/r0).
2. In that plane, u = 1/r obeys u'' = 3 u^2 - u (' = d/dphi), integrated with
   fourth-order Runge-Kutta in steps of phi. The start slope follows from
   (u')^2 = 1/b^2 - u^2 (1 - 2u) = u0^2 (1 - 2 u0) cot^2(alpha), negative for
   an outgoing ray.
3. The ray ends:
   - in the hole when u reaches 1/2 (r = 2): black;
   - at infinity when u reaches 0: the sky in direction
     cos(phi) e_r + sin(phi) e_t;
   - on the disc when it crosses the disc plane (y = 0) between the inner and
     outer radius. The crossing angle is exact; u there is a cubic Hermite
     interpolation within the step.
   Rays still going after max_steps orbit close to the photon sphere: black.
4. Disc colour: flux shape F(r) (schwarzschild.disc_flux_shape), emitted
   temperature T = T_peak (F / F_max)^(1/4), redshift factor g
   (schwarzschild.disc_redshift; L_z of the photon is -b (e_r x e_t)_y, minus
   because the real photon travels the traced path backwards). Observed:
   bolometric intensity g^4 F / F_max, colour of a blackbody at g T.

The trace is written to a floating-point image and only redone when the view
or a setting changes; exposure and tone mapping are applied afterwards.

Display choices (no physics): the blackbody colour is shown with luminance 1
and scaled by the bolometric intensity; the sky is a texture of stars and a
faint Milky-Way band, or a checkerboard to show the lensing; light from the
sky is not redshifted.
"""

from dataclasses import dataclass
from functools import lru_cache

import moderngl
import numpy as np

from physengine.modules.kerr import disc_flux_table
from physengine.modules.schwarzschild import R_ISCO

FLUX_LUT_RMAX = 60.0  # [M] the disc flux table reaches this far
FLUX_LUT_SIZE = 2048

QUAD_VERTEX = """
#version 460 core
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = 0.5 * in_pos + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

TRACE_FRAGMENT = """
#version 460 core
uniform vec3 cam_pos;           // relative to the hole, units of M
uniform mat3 cam_to_world;      // camera axes (right, up, back) as columns
uniform float tan_half_fov;
uniform float aspect;
uniform int show_disc;
uniform float r_in;
uniform float r_out;
uniform float t_peak;           // K, at the flux maximum
uniform sampler2D flux_lut;     // F / F_max against sqrt((r - r_in) / (lut_rmax - r_in))
uniform float lut_rmax;
uniform int redshift_mode;      // 0 full, 1 gravity only (static gas), 2 none
uniform int sky_mode;           // 0 stars, 1 checkerboard
uniform float sky_brightness;
uniform samplerCube sky;
uniform sampler2D blackbody;    // rgb of luminance 1 against log10(T)
uniform vec2 log_t_range;
uniform float h;                // step in phi
uniform int max_steps;
uniform int output_mode;        // 0 colour, 1 diagnostics (see below)
in vec2 uv;
out vec4 frag;

const float M = 1.0;
const float PI = 3.14159265358979;

vec2 deriv(vec2 s) {            // s = (u, du/dphi)
    return vec2(s.y, 3.0 * M * s.x * s.x - s.x);
}

vec3 checkerboard(vec3 d) {
    float lon = atan(d.z, d.x);
    float lat = asin(clamp(d.y, -1.0, 1.0));
    float cell = PI / 18.0;     // 10 degrees
    float parity = mod(floor(lon / cell) + floor(lat / cell), 2.0);
    vec3 a = d.y > 0.0 ? vec3(0.9, 0.55, 0.2) : vec3(0.25, 0.5, 0.9);
    return mix(0.08 * a, a, parity) * 0.6;
}

vec3 blackbody_rgb(float t) {
    float x = (log(max(t, 1.0)) / log(10.0) - log_t_range.x) / (log_t_range.y - log_t_range.x);
    return texture(blackbody, vec2(clamp(x, 0.0, 1.0), 0.5)).rgb;
}

void main() {
    vec2 ndc = 2.0 * uv - 1.0;
    vec3 dir = normalize(cam_to_world * vec3(ndc.x * aspect * tan_half_fov, ndc.y * tan_half_fov, -1.0));
    float r0 = length(cam_pos);

    // Outcome: 0 fell in, 1 escaped (sky_dir), 2 hit the disc, 3 still orbiting.
    int outcome = 0;
    vec3 sky_dir = dir;
    float r_hit = 0.0;
    float b = 0.0;
    vec3 eh = vec3(0.0), th = vec3(0.0);

    if (r0 > 2.0 * M) {
        eh = cam_pos / r0;
        float cos_a = dot(dir, eh);
        vec3 tp = dir - cos_a * eh;
        float sin_a = length(tp);
        if (sin_a < 1e-7) {
            outcome = cos_a > 0.0 ? 1 : 0;   // radial ray
        } else {
            th = tp / sin_a;
            float u0 = 1.0 / r0;
            b = r0 * sin_a / sqrt(1.0 - 2.0 * M * u0);
            vec2 s = vec2(u0, -u0 * sqrt(1.0 - 2.0 * M * u0) * cos_a / sin_a);
            float phi = 0.0;
            float y0 = eh.y;               // height sign of the ray point, at phi
            float cross_base = atan(-eh.y, th.y);  // disc plane at cross_base + k pi
            outcome = 3;
            for (int i = 0; i < max_steps; ++i) {
                vec2 k1 = deriv(s);
                vec2 k2 = deriv(s + 0.5 * h * k1);
                vec2 k3 = deriv(s + 0.5 * h * k2);
                vec2 k4 = deriv(s + h * k3);
                vec2 sn = s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                float phin = phi + h;
                float y1 = cos(phin) * eh.y + sin(phin) * th.y;

                if (show_disc == 1 && (y0 > 0.0) != (y1 > 0.0)) {
                    // The crossing nearest the middle of the step (robust when it
                    // falls on a step boundary, where ceil could pick the next one).
                    float pc = cross_base + PI * round((phi + 0.5 * h - cross_base) / PI);
                    float t = clamp((pc - phi) / h, 0.0, 1.0);
                    float t2 = t * t, t3 = t2 * t;
                    float uc = (2.0 * t3 - 3.0 * t2 + 1.0) * s.x + (t3 - 2.0 * t2 + t) * h * s.y
                             + (-2.0 * t3 + 3.0 * t2) * sn.x + (t3 - t2) * h * sn.y;
                    if (uc > 0.0) {
                        float rc = 1.0 / uc;
                        if (rc >= r_in && rc <= r_out) {
                            outcome = 2;
                            r_hit = rc;
                            break;
                        }
                    }
                }
                if (sn.x <= 0.0) {
                    float t = s.x / (s.x - sn.x);
                    float pinf = phi + t * h;
                    sky_dir = cos(pinf) * eh + sin(pinf) * th;
                    outcome = 1;
                    break;
                }
                if (sn.x >= 0.5 / M) {
                    outcome = 0;
                    break;
                }
                s = sn;
                phi = phin;
                y0 = y1;
            }
        }
    }

    // Disc: redshift factor and observed colour.
    float g = 0.0;
    vec3 disc_rgb = vec3(0.0);
    if (outcome == 2) {
        float r = r_hit;
        float obs = sqrt(1.0 - 2.0 * M / r0);
        float lz = -b * cross(eh, th).y;
        float omega = sqrt(M / (r * r * r));
        if (redshift_mode == 0) {
            g = sqrt(1.0 - 3.0 * M / r) / obs / (1.0 - omega * lz);
        } else if (redshift_mode == 1) {
            g = sqrt(1.0 - 2.0 * M / r) / obs;
        } else {
            g = 1.0;
        }
        float shape = texture(flux_lut, vec2(sqrt(clamp((r - r_in) / (lut_rmax - r_in), 0.0, 1.0)), 0.5)).r;
        float t_obs = t_peak * pow(max(shape, 0.0), 0.25) * g;
        disc_rgb = blackbody_rgb(t_obs) * shape * pow(g, 4.0);
    }

    // Sampled in uniform control flow, so the mipmap level is well defined.
    vec3 star_rgb = sky_brightness * texture(sky, sky_dir).rgb;

    if (output_mode == 1) {
        // Diagnostics for tests: escaped (sky_dir, 1); disc (r, g, b, 2);
        // fell in (0, 0, b, 0); still orbiting (0, 0, b, 3).
        if (outcome == 1) frag = vec4(sky_dir, 1.0);
        else if (outcome == 2) frag = vec4(r_hit, g, b, 2.0);
        else frag = vec4(0.0, 0.0, b, float(outcome));
        return;
    }
    vec3 c = vec3(0.0);
    if (outcome == 1) c = sky_mode == 1 ? checkerboard(sky_dir) : star_rgb;
    else if (outcome == 2) c = disc_rgb;
    frag = vec4(c, 1.0);
}
"""

# Images of bodies: soft-edged discs at a direction on the sky (display only;
# where, how large and how redshifted comes from blackhole_images.py).
SPRITE_VERTEX = """
#version 460 core
uniform mat4 view_proj;           // rotation-only view, times projection
in vec3 in_dir;
in vec3 in_color;                 // surface brightness x colour
in float in_radius;               // pixels; negative: a soft blob of that radius (gas)
out vec3 v_color;
out float v_radius;
out float v_size;
void main() {
    vec4 p = view_proj * vec4(in_dir, 1.0);
    gl_Position = vec4(p.xy, p.w * 0.9999, p.w);   // never clipped: it is on the sky
    // A disc gets one pixel of soft edge round it; a blob reaches two radii.
    v_size = in_radius < 0.0 ? -4.0 * in_radius + 2.0 : 2.0 * in_radius + 2.0;
    gl_PointSize = v_size;
    v_color = in_color;
    v_radius = in_radius;
}
"""

SPRITE_FRAGMENT = """
#version 460 core
in vec3 v_color;
in float v_radius;
in float v_size;
out vec4 frag;
void main() {
    float r = length(gl_PointCoord - vec2(0.5)) * v_size;                    // pixels from the centre
    if (v_radius < 0.0) {
        // Gas: a Gaussian blob, so that neighbours merge into a smooth cloud.
        float q = r / -v_radius;
        if (q > 2.0) discard;
        frag = vec4(v_color * exp(-2.0 * q * q), 1.0);
        return;
    }
    float cover = clamp(v_radius + 0.5 - r, 0.0, 1.0);                       // anti-aliased edge
    if (cover <= 0.0) discard;
    frag = vec4(v_color * cover, 1.0);
}
"""

KERR_TRACE_FRAGMENT = """
#version 460 core
// Kerr: see kerr.py for the equations. Photon with E = 1; Mino time lambda.
uniform vec3 cam_pos;
uniform mat3 cam_to_world;
uniform float tan_half_fov;
uniform float aspect;
uniform float spin;
uniform int show_disc;
uniform float r_in;
uniform float r_out;
uniform float t_peak;
uniform sampler2D flux_lut;     // F / F_max against sqrt((r - r_in) / (lut_rmax - r_in))
uniform float lut_rmax;
uniform int redshift_mode;      // 0 full, 1 gravity only (gas at rest in the ZAMO frame), 2 none
uniform int sky_mode;
uniform float sky_brightness;
uniform samplerCube sky;
uniform sampler2D blackbody;
uniform vec2 log_t_range;
uniform float h;                // largest change of r (relative), theta, phi per step
uniform int max_steps;
uniform int output_mode;
in vec2 uv;
out vec4 frag;

const float PI = 3.14159265358979;

vec3 checkerboard(vec3 d) {
    float lon = atan(d.z, d.x);
    float lat = asin(clamp(d.y, -1.0, 1.0));
    float cell = PI / 18.0;
    float parity = mod(floor(lon / cell) + floor(lat / cell), 2.0);
    vec3 a = d.y > 0.0 ? vec3(0.9, 0.55, 0.2) : vec3(0.25, 0.5, 0.9);
    return mix(0.08 * a, a, parity) * 0.6;
}

vec3 blackbody_rgb(float t) {
    float x = (log(max(t, 1.0)) / log(10.0) - log_t_range.x) / (log_t_range.y - log_t_range.x);
    return texture(blackbody, vec2(clamp(x, 0.0, 1.0), 0.5)).rgb;
}

// World unit vectors along d/dr, d/dtheta, d/dphi (oblate-spheroidal map, spin along +y).
void axes(float r, float th, float ph, out vec3 er, out vec3 et, out vec3 ep) {
    float a = spin;
    float R = sqrt(r * r + a * a);
    float s = sin(th), c = cos(th), sp = sin(ph), cp = cos(ph);
    er = normalize(vec3(r / R * s * cp, c, -r / R * s * sp));
    et = normalize(vec3(R * c * cp, -r * s, -R * c * sp));
    ep = vec3(-sp, 0.0, -cp);
}

// d/dlambda of (r, theta, phi, rdot, thetadot).
void rhs(float r, float th, float rd, float td, float L, float Q,
         out float dr, out float dth, out float dph, out float drd, out float dtd) {
    float a = spin;
    float s = sin(th);
    s = (abs(s) < 1e-6) ? (s >= 0.0 ? 1e-6 : -1e-6) : s;
    float c = cos(th);
    float dl = r * r - 2.0 * r + a * a;
    float P = r * r + a * a - a * L;
    dr = rd;
    dth = td;
    dph = -(a - L / (s * s)) + a * P / dl;
    drd = 0.5 * (4.0 * r * P - (2.0 * r - 2.0) * ((L - a) * (L - a) + Q));
    dtd = -a * a * c * s + L * L * c / (s * s * s);
}

// Radial and polar potentials R(r), Theta(theta) for a photon with E = 1.
float Rpot(float r, float L, float Q) {
    float a = spin;
    float P = r * r + a * a - a * L;
    return P * P - (r * r - 2.0 * r + a * a) * ((L - a) * (L - a) + Q);
}
float Thpot(float th, float L, float Q) {
    float s = sin(th), c = cos(th);
    return Q + spin * spin * c * c - L * L * c * c / max(s * s, 1e-12);
}

void main() {
    float a = spin;
    vec2 ndc = 2.0 * uv - 1.0;
    vec3 dir = normalize(cam_to_world * vec3(ndc.x * aspect * tan_half_fov, ndc.y * tan_half_fov, -1.0));

    vec3 p = cam_pos;
    float rho2 = dot(p, p);
    float r = sqrt(0.5 * (rho2 - a * a + sqrt((rho2 - a * a) * (rho2 - a * a) + 4.0 * a * a * p.y * p.y)));
    float th = acos(clamp(p.y / r, -1.0, 1.0));
    float ph = atan(-p.z, p.x);
    float rh = 1.0 + sqrt(1.0 - a * a);
    float r0 = r;

    int outcome = 0;      // 0 fell in, 1 escaped, 2 disc, 3 still going
    vec3 sky_dir = dir;
    float r_hit = 0.0, g = 0.0, L = 0.0, E0 = 1.0;

    if (r > rh * 1.0001) {
        vec3 er, et, ep;
        axes(r, th, ph, er, et, ep);
        // The photon arrives moving along -dir. Kerr is not symmetric under time
        // reversal, so the constants are the real photon's and the path is followed
        // backwards in Mino time from the camera.
        vec3 n = -vec3(dot(dir, er), dot(dir, et), dot(dir, ep));
        float s = sin(th), c = cos(th);
        float sig = r * r + a * a * c * c;
        float dl = r * r - 2.0 * r + a * a;
        float A = (r * r + a * a) * (r * r + a * a) - a * a * dl * s * s;
        float om = 2.0 * a * r / A;
        float enu = sqrt(sig * dl / A);
        float epsi = sqrt(A / sig) * s;
        float L0 = epsi * n.z;
        E0 = enu + om * L0;
        L = L0 / E0;
        float pr = sqrt(sig / dl) * n.x / E0;
        float pt = sqrt(sig) * n.y / E0;
        float Q = pt * pt + c * c * (L * L / max(s * s, 1e-12) - a * a);
        float rd = dl * pr, td = pt;
        float r_far = max(2.0 * r0, 300.0);
        outcome = 3;
        for (int i = 0; i < max_steps; ++i) {
            float k1r, k1t, k1p, k1rd, k1td;
            rhs(r, th, rd, td, L, Q, k1r, k1t, k1p, k1rd, k1td);
            float step = -h / max(max(abs(k1r) / r, abs(k1t)), max(abs(k1p), 1e-9));
            float k2r, k2t, k2p, k2rd, k2td;
            rhs(r + 0.5 * step * k1r, th + 0.5 * step * k1t, rd + 0.5 * step * k1rd, td + 0.5 * step * k1td, L, Q, k2r, k2t, k2p, k2rd, k2td);
            float k3r, k3t, k3p, k3rd, k3td;
            rhs(r + 0.5 * step * k2r, th + 0.5 * step * k2t, rd + 0.5 * step * k2rd, td + 0.5 * step * k2td, L, Q, k3r, k3t, k3p, k3rd, k3td);
            float k4r, k4t, k4p, k4rd, k4td;
            rhs(r + step * k3r, th + step * k3t, rd + step * k3rd, td + step * k3td, L, Q, k4r, k4t, k4p, k4rd, k4td);
            float rn = r + step / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
            float tn = th + step / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
            float pn = ph + step / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
            float rdn = rd + step / 6.0 * (k1rd + 2.0 * k2rd + 2.0 * k3rd + k4rd);
            float tdn = td + step / 6.0 * (k1td + 2.0 * k2td + 2.0 * k3td + k4td);
            float c0 = cos(th), c1 = cos(tn);
            if (show_disc == 1 && (c0 > 0.0) != (c1 > 0.0)) {
                float f = c0 / (c0 - c1);
                float rc = r + f * (rn - r);
                if (rc >= r_in && rc <= r_out) {
                    outcome = 2;
                    r_hit = rc;
                    break;
                }
            }
            // Back onto the exact first integrals (dr/dlambda)^2 = R, (dtheta/dlambda)^2 = Theta:
            // in float32 the second-order equations alone drift badly far out, where R ~ r^4.
            rdn = (rdn >= 0.0 ? 1.0 : -1.0) * sqrt(max(Rpot(rn, L, Q), 0.0));
            // Not near the poles: there Theta has L^2 cot^2(theta), and in float32 the
            // projection can stall a ray (L ~ 0) at the pole; the second-order equation
            // alone carries it across.
            if (abs(sin(tn)) > 0.1) tdn = (tdn >= 0.0 ? 1.0 : -1.0) * sqrt(max(Thpot(tn, L, Q), 0.0));
            r = rn; th = tn; ph = pn; rd = rdn; td = tdn;
            if (r <= rh * 1.0001) { outcome = 0; break; }
            if (r >= r_far && rd < 0.0) {
                // It came in from there: minus its direction of travel in the local frame.
                float s2 = sin(th), c2 = cos(th);
                float sig2 = r * r + a * a * c2 * c2;
                float dl2 = r * r - 2.0 * r + a * a;
                float A2 = (r * r + a * a) * (r * r + a * a) - a * a * dl2 * s2 * s2;
                vec3 m = vec3(rd / dl2 / sqrt(sig2 / dl2), td / sqrt(sig2), L / (sqrt(A2 / sig2) * s2));
                vec3 er2, et2, ep2;
                axes(r, th, ph, er2, et2, ep2);
                sky_dir = -normalize(m.x * er2 + m.y * et2 + m.z * ep2);
                outcome = 1;
                break;
            }
        }
    }

    vec3 disc_rgb = vec3(0.0);
    if (outcome == 2) {
        float rr = r_hit;
        if (redshift_mode == 0) {
            float ut = (pow(rr, 1.5) + a) / (pow(rr, 0.75) * sqrt(pow(rr, 1.5) - 3.0 * sqrt(rr) + 2.0 * a));
            float Om = 1.0 / (pow(rr, 1.5) + a);
            g = 1.0 / (E0 * ut * (1.0 - Om * L));
        } else if (redshift_mode == 1) {
            float dle = rr * rr - 2.0 * rr + a * a;
            float Ae = (rr * rr + a * a) * (rr * rr + a * a) - a * a * dle;
            float ome = 2.0 * a * rr / Ae;
            float enue = sqrt(rr * rr * dle / Ae);
            g = enue / (E0 * (1.0 - ome * L));
        } else {
            g = 1.0;
        }
        float x = sqrt(clamp((rr - r_in) / (lut_rmax - r_in), 0.0, 1.0));
        float shape = texture(flux_lut, vec2(x, 0.5)).r;
        float t_obs = t_peak * pow(max(shape, 0.0), 0.25) * g;
        disc_rgb = blackbody_rgb(t_obs) * shape * pow(g, 4.0);
    }

    vec3 star_rgb = sky_brightness * texture(sky, sky_dir).rgb;
    if (output_mode == 1) {
        if (outcome == 1) frag = vec4(sky_dir, 1.0);
        else if (outcome == 2) frag = vec4(r_hit, g, L, 2.0);
        else frag = vec4(0.0, 0.0, L, float(outcome));
        return;
    }
    vec3 col = vec3(0.0);
    if (outcome == 1) col = sky_mode == 1 ? checkerboard(sky_dir) : star_rgb;
    else if (outcome == 2) col = disc_rgb;
    frag = vec4(col, 1.0);
}
"""

COMPOSITE_FRAGMENT = """
#version 460 core
uniform sampler2D hdr;
uniform sampler2D bodies;
uniform float exposure;
in vec2 uv;
out vec4 frag_color;
void main() {
    vec3 c = exposure * (texture(hdr, uv).rgb + texture(bodies, uv).rgb);
    // Tone curve on the brightness only, so colours survive in bright regions.
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c *= (1.0 - exp(-lum)) / max(lum, 1e-6);
    c = min(c, vec3(1.0));
    frag_color = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
}
"""

# ------------------------------------------------------------ colour of heat

LOG_T_RANGE = (2.8, 5.0)  # 630 K to 100 000 K
_HC_OVER_K_NM = 1.438777e7  # h c / k_B in nm K


def _cie_cmf(lam):
    """CIE 1931 colour matching functions, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013)."""

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    x = (
        1.056 * g(lam, 599.8, 37.9, 31.0)
        + 0.362 * g(lam, 442.0, 16.0, 26.7)
        - 0.065 * g(lam, 501.1, 20.4, 26.2)
    )
    y = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    return x, y, z


XYZ_TO_LINEAR_SRGB = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
)
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def blackbody_rgb(temperature):
    """Linear sRGB colour of a blackbody, scaled to luminance 1 (negative parts clipped)."""
    lam = np.linspace(380.0, 780.0, 401)
    t = np.atleast_1d(np.asarray(temperature, dtype=float))[:, None]
    planck = lam**-5 / np.expm1(_HC_OVER_K_NM / (lam * t))
    xyz = np.stack([planck @ c for c in _cie_cmf(lam)], axis=1)
    rgb = np.maximum(xyz @ XYZ_TO_LINEAR_SRGB.T, 0.0)
    return rgb / (rgb @ LUMINANCE)[:, None]


def blackbody_table(n=512):
    temps = 10.0 ** np.linspace(*LOG_T_RANGE, n)
    return blackbody_rgb(temps).astype("f4")


# ---------------------------------------------------------------- the sky

# Face order of a GL cube map and, for face coordinates (sc, tc) in [-1, 1],
# the direction each texel looks along (OpenGL 4.6 spec, table 8.19, inverted).
_FACES = [
    lambda s, t: (np.ones_like(s), -t, -s),  # +X
    lambda s, t: (-np.ones_like(s), -t, s),  # -X
    lambda s, t: (s, np.ones_like(s), t),  # +Y
    lambda s, t: (s, -np.ones_like(s), -t),  # -Y
    lambda s, t: (s, -t, np.ones_like(s)),  # +Z
    lambda s, t: (-s, -t, -np.ones_like(s)),  # -Z
]


def cube_texel(direction, size):
    """(face, row, column) of the cube-map texel that a unit direction hits."""
    d = np.atleast_2d(direction)
    ax = np.argmax(np.abs(d), axis=1)
    sign = d[np.arange(len(d)), ax] > 0
    face = 2 * ax + (~sign)
    x, y, z = d.T
    ma = np.abs(d[np.arange(len(d)), ax])
    sc = np.select([face == 0, face == 1, face == 2, face == 3, face == 4], [-z, z, x, x, x], -x)
    tc = np.select([face == 0, face == 1, face == 2, face == 3, face == 4], [-y, -y, z, -z, -y], -y)
    col = np.clip(((sc / ma + 1.0) * 0.5 * size).astype(int), 0, size - 1)
    row = np.clip(((tc / ma + 1.0) * 0.5 * size).astype(int), 0, size - 1)
    return face, row, col


@lru_cache(maxsize=1)
def sky_faces(size=768, n_stars=24000, seed=3):
    """Six cube-map faces (6, size, size, 3) float32: stars and a faint band of light."""
    rng = np.random.default_rng(seed)
    c = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    tc, sc = np.meshgrid(c, c, indexing="ij")  # rows are t, columns are s
    faces = np.empty((6, size, size, 3), dtype="f4")
    band_normal = np.array([0.35, 0.85, 0.4])
    band_normal /= np.linalg.norm(band_normal)
    for f, to_dir in enumerate(_FACES):
        d = np.stack(to_dir(sc, tc), axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        height = d @ band_normal
        wiggle = 0.6 + 0.4 * np.sin(9.0 * d[..., 0] + 4.0 * d[..., 2]) * np.sin(
            7.0 * d[..., 1] - 5.0 * d[..., 0]
        )
        band = 0.05 * np.exp(-0.5 * (height / 0.13) ** 2) * wiggle
        faces[f] = band[..., None] * np.array([1.0, 0.85, 0.7], dtype="f4")

    direction = rng.normal(size=(n_stars, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    brightness = 0.3 + 6.0 * rng.random(n_stars) ** 12
    colour = blackbody_rgb(10.0 ** rng.uniform(3.45, 4.1, n_stars))
    face, row, col = cube_texel(direction, size)
    np.add.at(faces, (face, row, col), (colour * brightness[:, None]).astype("f4"))
    return faces


# ----------------------------------------------------------------- renderer


class BlackHoleRenderer:
    def __init__(self):
        self.size = None
        self.key = None  # what the current trace was made with
        self.trace_ms = 0.0  # GPU time of the last trace
        self._gl = []

    def _keep(self, obj):
        self._gl.append(obj)
        return obj

    def setup(self, ctx):
        self.ctx = ctx
        self.schw_prog = self._keep(
            ctx.program(vertex_shader=QUAD_VERTEX, fragment_shader=TRACE_FRAGMENT)
        )
        self.kerr_prog = self._keep(
            ctx.program(vertex_shader=QUAD_VERTEX, fragment_shader=KERR_TRACE_FRAGMENT)
        )
        self.flux_tex = None
        self.flux_spin = None
        self.comp_prog = self._keep(
            ctx.program(vertex_shader=QUAD_VERTEX, fragment_shader=COMPOSITE_FRAGMENT)
        )
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype="f4")
        buf = self._keep(ctx.buffer(quad.tobytes()))
        self.schw_vao = self._keep(ctx.vertex_array(self.schw_prog, [(buf, "2f", "in_pos")]))
        self.kerr_vao = self._keep(ctx.vertex_array(self.kerr_prog, [(buf, "2f", "in_pos")]))
        self.comp_vao = self._keep(ctx.vertex_array(self.comp_prog, [(buf, "2f", "in_pos")]))
        self.sprite_prog = self._keep(
            ctx.program(vertex_shader=SPRITE_VERTEX, fragment_shader=SPRITE_FRAGMENT)
        )
        self.sprite_buf = None
        self.sprite_vao = None

        faces = sky_faces()
        size = faces.shape[1]
        self.sky = self._keep(ctx.texture_cube((size, size), 3, faces.tobytes(), dtype="f4"))
        self.sky.build_mipmaps()
        self.sky.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        table = blackbody_table()
        self.bb = self._keep(ctx.texture((len(table), 1), 3, table.tobytes(), dtype="f4"))
        self.bb.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bb.repeat_x = self.bb.repeat_y = False

    def _resize(self, size):
        for obj in getattr(self, "_targets", []):
            obj.release()
        self.hdr = self.ctx.texture(size, 4, dtype="f4")
        self.hdr.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr.repeat_x = self.hdr.repeat_y = False
        self.fbo = self.ctx.framebuffer(color_attachments=[self.hdr])
        self.body_tex = self.ctx.texture(size, 4, dtype="f2")
        self.body_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.body_tex.repeat_x = self.body_tex.repeat_y = False
        self.body_fbo = self.ctx.framebuffer(color_attachments=[self.body_tex])
        self._targets = [self.fbo, self.hdr, self.body_fbo, self.body_tex]
        self.size = size

    def trace(self, size, cam_pos, cam_to_world, fov, aspect, settings, output_mode=0):
        """Trace every pixel of a size = (w, h) image into self.hdr (RGBA float32).

        cam_pos: camera position relative to the hole [M]; cam_to_world: 3x3
        with the camera's right, up and backward axes as columns; settings: a
        TraceSettings.
        """
        size = (max(int(size[0]), 1), max(int(size[1]), 1))
        if self.size != size:
            self._resize(size)
        kerr = settings.spin > 0.0
        p = self.kerr_prog if kerr else self.schw_prog
        self._flux_lut(settings.spin)
        p["cam_pos"].value = tuple(float(x) for x in cam_pos)
        p["cam_to_world"].write(np.asarray(cam_to_world, dtype="f4").T.tobytes())
        p["tan_half_fov"].value = float(np.tan(fov / 2.0))
        p["aspect"].value = float(aspect)
        p["show_disc"].value = int(settings.show_disc)
        p["r_in"].value = settings.r_in
        p["r_out"].value = settings.r_out
        p["t_peak"].value = settings.t_peak
        p["lut_rmax"].value = FLUX_LUT_RMAX
        if kerr:
            p["spin"].value = settings.spin
        p["redshift_mode"].value = settings.redshift_mode
        p["sky_mode"].value = settings.sky_mode
        p["sky_brightness"].value = settings.sky_brightness
        p["log_t_range"].value = LOG_T_RANGE
        p["h"].value = settings.step
        p["max_steps"].value = settings.max_steps
        p["output_mode"].value = output_mode
        p["sky"].value = 0
        p["blackbody"].value = 1
        p["flux_lut"].value = 2
        self.sky.use(0)
        self.bb.use(1)
        self.flux_tex.use(2)

        self.fbo.use()  # left bound: render() rebinds its target afterwards
        query = self.ctx.query(time=True)
        with query:
            (self.kerr_vao if kerr else self.schw_vao).render(moderngl.TRIANGLE_STRIP)
        self.trace_ms = query.elapsed * 1e-6

    def _flux_lut(self, spin):
        """Disc flux table for this spin (kerr.disc_flux_table), as a texture, rebuilt on change."""
        if self.flux_spin == spin and self.flux_tex is not None:
            return
        if self.flux_tex is not None:
            self.flux_tex.release()
        r, F = disc_flux_table(max(spin, 1e-9), r_out=FLUX_LUT_RMAX * 1.01, n=6000)
        r_in = r[0]
        x = np.linspace(0.0, 1.0, FLUX_LUT_SIZE)
        table = np.interp(r_in + (FLUX_LUT_RMAX - r_in) * x**2, r, F).astype("f4")
        self.flux_tex = self.ctx.texture((FLUX_LUT_SIZE, 1), 1, table.tobytes(), dtype="f4")
        self.flux_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.flux_tex.repeat_x = self.flux_tex.repeat_y = False
        self.flux_spin = spin

    def read(self):
        """The last trace as an (h, w, 4) float32 array, row 0 at the bottom."""
        w, h = self.size
        return np.frombuffer(self.hdr.read(), dtype="f4").reshape(h, w, 4)

    def _draw_sprites(self, camera, sprites, scale):
        """Body images into body_tex: sprites = (directions (n, 3), rgb (n, 3), radius in screen px (n,))."""
        self.body_fbo.use()
        self.body_fbo.clear(0.0, 0.0, 0.0, 1.0)
        if sprites is None or len(sprites[0]) == 0:
            return
        dirs, rgb, radius = sprites
        data = np.hstack(
            [
                np.asarray(dirs, "f4"),
                np.asarray(rgb, "f4"),
                (np.asarray(radius) * scale)[:, None].astype("f4"),
            ]
        )
        raw = np.ascontiguousarray(data).tobytes()
        if self.sprite_buf is None or self.sprite_buf.size < len(raw):
            if self.sprite_buf is not None:
                self.sprite_vao.release()
                self.sprite_buf.release()
            self.sprite_buf = self.ctx.buffer(reserve=max(len(raw), 28) * 2)
            self.sprite_vao = self.ctx.vertex_array(
                self.sprite_prog, [(self.sprite_buf, "3f 3f 1f", "in_dir", "in_color", "in_radius")]
            )
        self.sprite_buf.write(raw)
        view = camera.view().copy()
        view[:3, 3] = 0.0
        self.sprite_prog["view_proj"].write((camera.projection() @ view).T.astype("f4").tobytes())
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE  # light adds up
        self.sprite_vao.render(moderngl.POINTS, vertices=len(data))

    def render(self, ctx, camera, settings, exposure=1.0, scale=1.0, sprites=None):
        """Draw the view of `camera` (world units of M, hole at the origin) to the current target.

        sprites: images of bodies to add, see _draw_sprites.
        """
        target = ctx.fbo
        vw, vh = target.viewport[2:]
        size = (max(int(vw * scale), 1), max(int(vh * scale), 1))
        view = camera.view()
        rot = view[:3, :3]
        cam_pos = -rot.T @ view[:3, 3]
        key = (size, tuple(np.round(view.ravel(), 12)), camera.fov, camera.aspect_ratio, settings)
        if key != self.key:
            self.trace(size, cam_pos, rot.T, camera.fov, camera.aspect_ratio, settings)
            self.key = key

        self._draw_sprites(camera, sprites, scale)

        target.use()
        ctx.disable(moderngl.BLEND)
        self.hdr.use(0)
        self.body_tex.use(1)
        self.comp_prog["hdr"].value = 0
        self.comp_prog["bodies"].value = 1
        self.comp_prog["exposure"].value = exposure
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self):
        if self.flux_tex is not None:
            self.flux_tex.release()
            self.flux_tex = None
        for obj in (self.sprite_vao, self.sprite_buf):
            if obj is not None:
                obj.release()
        self.sprite_vao = self.sprite_buf = None
        for obj in self._gl + getattr(self, "_targets", []):
            obj.release()
        self._gl = []
        self._targets = []
        self.size = None
        self.key = None


@dataclass(frozen=True)
class TraceSettings:
    """Everything the trace depends on besides the camera (hashable, for the cache)."""

    show_disc: bool = True
    r_in: float = R_ISCO  # disc inner edge [M]
    r_out: float = 20.0  # disc outer edge [M]
    t_peak: float = 7000.0  # emitted temperature at the flux maximum [K]
    redshift_mode: int = 0  # 0 full, 1 gravity only, 2 none
    sky_mode: int = 0  # 0 stars, 1 checkerboard
    sky_brightness: float = 1.0
    step: float = 0.02  # RK4 step in phi [rad]
    max_steps: int = 3000
    spin: float = 0.0  # a / M; 0 uses the exact Schwarzschild tracer
