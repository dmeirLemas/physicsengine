"""Where bodies near the hole appear on the camera: lensed images, size, redshift.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). Review the physics; formulas in schwarzschild.py and geodesics.py.

Units G = c = M = 1. The camera is a static observer at C (r0 = |C|), a body
is at P. Light between them moves in the plane through the hole, C and P.
Seen from the hole, P is the angle D = angle(C, P) away from C. A ray from
the camera sweeping that plane reaches P's direction after phi = D (the
direct image) or, going round the other side, phi = 2 pi - D (the secondary
image, just outside the shadow on the opposite side). Higher orders hug the
photon ring and are left out.

1. Ray table, built once per camera distance: for launch angles alpha (from
   the outward radial direction), u = 1/r of the ray at phi = k dphi, from
   u'' = 3u^2 - u (as in the shader). Rays that escaped are marked -1, rays
   that fell in +1.
2. For each body and image, bisection over alpha for u(alpha, phi_target) =
   1/|P|. u grows with alpha at fixed phi (a ray launched more steeply
   inward is closer to the hole at every angle), so the root is unique.
   Direct images within NEAR_AXIS of the line of sight come from nearly
   radial rays that cross the whole table within one phi step; for those b
   is solved exactly from phi = integral b du / sqrt(1 - b^2 u^2 (1 - 2u)).
3. The image direction is cos(alpha) e_C + sin(alpha) e, with e the unit
   vector in the plane, perpendicular to e_C, towards P (direct) or away
   from it (secondary).
4. Size: rays launched into the solid angle dOmega reach the sphere
   |x| = |P| on a patch of area
       dA = dOmega |P|^2 sin(phi) / (sin(alpha) |dalpha/dphi|)
   (sideways factor from turning the plane about e_C, radial one from
   dalpha/dphi, numerical). Their cross-section across the ray is
   dA |cos(theta)|, theta the angle between the ray and the radial direction
   in the body's static frame: cos(theta) = |p^r| / E below. A body of
   radius R is therefore seen under pi R^2 times
       dOmega / dA_perp = sin(alpha) |dalpha/dphi| / (|P|^2 sin(phi) |cos(theta)|),
   which in flat space is 1 / distance^2.
5. Redshift g = nu_obs / nu_emit for a photon of E = 1 and angular momentum
   vector L = -b n (n the normal of the traced sweep; minus: the real photon
   runs the traced path backwards), b = r0 sin(alpha) / sqrt(1 - 2/r0):
       -p.u = u^t - p_r u^r - L . Omega,
   u^t = E_body / (1 - 2/r), u^r = dr/dtau, Omega = x x w / r^2 the body's
   angular velocity per proper time, p_r = p^r / (1 - 2/r) with
   p^r = +-sqrt(1 - (1 - 2/r) b^2 / r^2) (sign: the real photon moves out
   where the traced ray moves in);  g = 1 / (sqrt(1 - 2/r0) (-p.u)).
   For gas on a circular orbit this is schwarzschild.disc_redshift.
6. Hidden: if the ray crosses the disc plane between the disc's radii before
   reaching the body.
7. Light-travel time: the coordinate time the light needs from the body to
   the camera, dt/du = 1 / (u^2 (1 - 2u) sqrt(1/b^2 - u^2 + 2u^3)) (from
   dt/dlambda = E / (1 - 2u) and the radial equation; Hartle 2003, ch. 9),
   integrated by Gauss-Legendre over u, through the turning point if the ray
   passed it. For a radial ray it is the difference of the tortoise
   coordinate r + 2 ln(r/2 - 1). The module shows each image where the body
   was at t - delay.
"""

from typing import NamedTuple

import numba
import numpy as np

from physengine.modules.schwarzschild import shadow_angular_radius

N_ALPHA = 4096
DPHI = 0.01
PHI_MAX = 2.0 * np.pi + 0.05
ESCAPED = -1.0
CAPTURED = 1.0


def alpha_grid(r0, n=N_ALPHA):
    """Launch angles in [0, pi], dense near the shadow edge alpha_c = pi - shadow radius."""
    a_c = np.pi - float(shadow_angular_radius(r0))
    half = n // 2
    s = np.linspace(0.0, 1.0, half, endpoint=False)
    below = a_c * (1.0 - (1.0 - s) ** 3)  # 0 ... just below a_c
    s2 = np.linspace(0.0, 1.0, n - half)
    above = a_c + (np.pi - a_c) * s2**3  # a_c ... pi
    return np.concatenate([below, above])


@numba.njit(parallel=True, cache=True)
def _build_table(r0, alphas, dphi, n_phi):
    n = len(alphas)
    table = np.empty((n, n_phi), dtype=np.float64)
    u0 = 1.0 / r0
    for i in numba.prange(n):
        a = alphas[i]
        sa = np.sin(a)
        if sa < 1e-12:
            # Radial ray: outward escapes at once, inward falls straight in.
            fill = ESCAPED if np.cos(a) > 0.0 else CAPTURED
            table[i, 0] = u0
            for k in range(1, n_phi):
                table[i, k] = fill
            continue
        u = u0
        v = -u0 * np.sqrt(1.0 - 2.0 * u0) * np.cos(a) / sa
        table[i, 0] = u
        done = 0.0
        for k in range(1, n_phi):
            if done != 0.0:
                table[i, k] = done
                continue
            h = dphi
            k1u, k1v = v, 3.0 * u * u - u
            u2, v2 = u + 0.5 * h * k1u, v + 0.5 * h * k1v
            k2u, k2v = v2, 3.0 * u2 * u2 - u2
            u3, v3 = u + 0.5 * h * k2u, v + 0.5 * h * k2v
            k3u, k3v = v3, 3.0 * u3 * u3 - u3
            u4, v4 = u + h * k3u, v + h * k3v
            k4u, k4v = v4, 3.0 * u4 * u4 - u4
            u += h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            if u <= 0.0:
                done = ESCAPED
                table[i, k] = u  # the crossing itself, for interpolation
            elif u >= 0.5:
                done = CAPTURED
                table[i, k] = CAPTURED
            else:
                table[i, k] = u
    return table


class RayTable:
    """u(alpha, phi) of camera rays for one camera distance r0."""

    def __init__(self, r0):
        self.r0 = float(r0)
        self.alphas = alpha_grid(self.r0)
        self.n_phi = int(np.ceil(PHI_MAX / DPHI)) + 1
        self.table = _build_table(self.r0, self.alphas, DPHI, self.n_phi)


@numba.njit(cache=True)
def _u_at(table, fi, phi, dphi):
    """u at continuous alpha index fi and angle phi (bilinear)."""
    n, m = table.shape
    i0 = min(max(int(fi), 0), n - 2)
    fa = fi - i0
    x = phi / dphi
    k0 = min(max(int(x), 0), m - 2)
    fk = x - k0
    a = table[i0, k0] + fk * (table[i0, k0 + 1] - table[i0, k0])
    b = table[i0 + 1, k0] + fk * (table[i0 + 1, k0 + 1] - table[i0 + 1, k0])
    # Sentinels must not be blended into real values: take the nearer row.
    if (a == CAPTURED) != (b == CAPTURED) or (a == ESCAPED) != (b == ESCAPED):
        return a if fa < 0.5 else b
    return a + fa * (b - a)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


@numba.njit(cache=True)
def _turning_u(b):
    """u at the closest approach of a ray with b > sqrt(27) (schwarzschild.turning_point)."""
    return 1.0 / (2.0 * b / np.sqrt(3.0) * np.cos(np.arccos(-3.0 * np.sqrt(3.0) / b) / 3.0))


@numba.njit(cache=True)
def _dt_du(u, b):
    """|dt/du| along a ray: 1 / (b u^2 (1 - 2u) sqrt(1/b^2 - u^2 + 2u^3))."""
    q = 1.0 - b * b * u * u * (1.0 - 2.0 * u)  # = b^2 (1/b^2 - u^2 + 2u^3)
    return 1.0 / (u * u * (1.0 - 2.0 * u) * np.sqrt(max(q, 1e-300)))


@numba.njit(cache=True)
def _time_between(ua, ub, b, ub_is_turning, gx, gw):
    """Coordinate time for a ray to go between u = ua and u = ub (Gauss-Legendre).

    If ub is the turning point, u = ub - (ub - ua) s^2 removes the inverse
    square root there.
    """
    total = 0.0
    if ub_is_turning:
        for j in range(len(gx)):
            sj = 0.5 * (gx[j] + 1.0)
            u = ub - (ub - ua) * sj * sj
            total += 0.5 * gw[j] * _dt_du(u, b) * 2.0 * (ub - ua) * sj
    else:
        lo, hi = min(ua, ub), max(ua, ub)
        for j in range(len(gx)):
            u = lo + 0.5 * (gx[j] + 1.0) * (hi - lo)
            total += 0.5 * gw[j] * (hi - lo) * _dt_du(u, b)
    return total


NEAR_AXIS = 0.05  # [rad] direct images closer than this to the line of sight are solved exactly


@numba.njit(cache=True)
def _phi_no_turn(ua, ub, b, gx, gw):
    """Angle swept between u = ua and ub by a ray without a turning point in between.

    dphi/du = b / sqrt(1 - b^2 u^2 (1 - 2u)), regular as b -> 0 (radial rays).
    """
    lo, hi = min(ua, ub), max(ua, ub)
    total = 0.0
    for j in range(len(gx)):
        u = lo + 0.5 * (gx[j] + 1.0) * (hi - lo)
        total += (
            0.5
            * gw[j]
            * (hi - lo)
            * b
            / np.sqrt(max(1.0 - b * b * u * u * (1.0 - 2.0 * u), 1e-300))
        )
    return total


@numba.njit(cache=True)
def _solve_b_near_axis(u0, ub, phi, gx, gw):
    """Impact parameter of the nearly radial ray from u0 that reaches ub after sweeping phi, or -1.

    The swept angle grows with b; b stays below the value at which a turning
    point would appear between u0 and ub (1 / max(u sqrt(1 - 2u)) there).
    """
    lo_u, hi_u = min(u0, ub), max(u0, ub)
    peak = hi_u * np.sqrt(1.0 - 2.0 * hi_u)
    if lo_u < 1.0 / 3.0 < hi_u:
        peak = max(peak, 1.0 / np.sqrt(27.0))
    peak = max(peak, lo_u * np.sqrt(1.0 - 2.0 * lo_u))
    b_hi = 0.999 / peak
    if _phi_no_turn(u0, ub, b_hi, gx, gw) < phi:
        return -1.0
    lo, hi = 0.0, b_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _phi_no_turn(u0, ub, mid, gx, gw) < phi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@numba.njit(cache=True)
def _u_after_phi_no_turn(u0, ub, b, phi_c, gx, gw):
    """u on that ray after sweeping phi_c (< the angle to ub)."""
    lo, hi = min(u0, ub), max(u0, ub)
    inward = ub > u0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        swept = _phi_no_turn(u0, mid, b, gx, gw)
        if (swept < phi_c) == inward:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@numba.njit(cache=True)
def _light_delay(u0, u_body, b, inward_start, inward_at_body, gx, gw):
    """Coordinate time the light takes between camera (u0) and body, for a ray of impact parameter b.

    A ray that starts inward and is moving outward at the body passed its
    turning point u_t in between: u0 -> u_t -> u_body.
    """
    if inward_start and not inward_at_body and b > np.sqrt(27.0):
        ut = _turning_u(b)
        return _time_between(u0, ut, b, True, gx, gw) + _time_between(u_body, ut, b, True, gx, gw)
    return _time_between(u0, u_body, b, False, gx, gw)


@numba.njit(cache=True)
def _solve_alpha(table, phi, u_target, dphi):
    """Continuous alpha index of the ray at u_target after sweeping phi, or -1."""
    n = table.shape[0]
    lo, hi = 0.0, float(n - 1)
    if _u_at(table, lo, phi, dphi) > u_target or _u_at(table, hi, phi, dphi) < u_target:
        return -1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _u_at(table, mid, phi, dphi) < u_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@numba.njit(cache=True)
def _alpha_of(alphas, fi):
    n = len(alphas)
    i0 = min(max(int(fi), 0), n - 2)
    f = fi - i0
    return alphas[i0] + f * (alphas[i0 + 1] - alphas[i0])


@numba.njit(parallel=True, cache=True)
def _images(
    table, alphas, dphi, r0, cam, pos, w, e_body, r_in, r_out, disc_on, gx, gw, out_dir, out_info
):
    """Both images of every body.

    out_info[k, order] = (visible, solid angle per area across the ray, g, alpha, delay).
    """
    n = len(pos)
    ch = cam / r0
    lapse0 = np.sqrt(1.0 - 2.0 / r0)
    two_pi = 2.0 * np.pi
    for k in numba.prange(n):
        p = pos[k]
        r = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
        for order in range(2):
            out_info[k, order, 0] = 0.0
        if r <= 2.0:
            continue
        ph = p / r
        cosd = min(max(ph[0] * ch[0] + ph[1] * ch[1] + ph[2] * ch[2], -1.0), 1.0)
        d = np.arccos(cosd)
        et = ph - cosd * ch
        etn = np.sqrt(et[0] ** 2 + et[1] ** 2 + et[2] ** 2)
        if etn < 1e-12:  # on the camera's axis: any plane will do
            et = np.array([0.0, 1.0, 0.0]) - ch[1] * ch
            if np.sqrt(et[0] ** 2 + et[1] ** 2 + et[2] ** 2) < 1e-6:
                et = np.array([1.0, 0.0, 0.0]) - ch[0] * ch
            etn = np.sqrt(et[0] ** 2 + et[1] ** 2 + et[2] ** 2)
        et = et / etn
        u_t = 1.0 / r
        for order in range(2):
            phi = d if order == 0 else two_pi - d
            e = et if order == 0 else -et
            u0 = 1.0 / r0
            near = order == 0 and phi < NEAR_AXIS
            fi = -1.0
            b_near = -1.0
            if near:
                # Nearly radial ray: the table cannot resolve it; solve exactly.
                b_near = _solve_b_near_axis(u0, u_t, phi, gx, gw)
                if b_near < 0.0:
                    continue
                inward = u_t > u0
                a = np.arcsin(min(b_near * lapse0 / r0, 1.0))
                if inward:
                    a = np.pi - a
                delta = 0.05 * phi + 1e-7
                bp = _solve_b_near_axis(u0, u_t, phi + delta, gx, gw)
                bm = _solve_b_near_axis(u0, u_t, max(phi - delta, 1e-9), gx, gw)
                if bp < 0.0 or bm < 0.0:
                    continue
                ap = np.arcsin(min(bp * lapse0 / r0, 1.0))
                am = np.arcsin(min(bm * lapse0 / r0, 1.0))
                dadphi = (ap - am) / (phi + delta - max(phi - delta, 1e-9))
            else:
                fi = _solve_alpha(table, phi, u_t, dphi)
                if fi < 0.0:
                    continue
                a = _alpha_of(alphas, fi)
                # Magnification: alpha at phi +- delta.
                delta = 2e-3
                fa = _solve_alpha(table, phi + delta, u_t, dphi)
                fb = _solve_alpha(table, max(phi - delta, 1e-6), u_t, dphi)
                if fa < 0.0 or fb < 0.0:
                    continue
                dadphi = (_alpha_of(alphas, fa) - _alpha_of(alphas, fb)) / (
                    phi + delta - max(phi - delta, 1e-6)
                )
            sphi = abs(np.sin(phi))
            solid = np.sin(a) * abs(dadphi) / (r * r * max(sphi, 1e-4))  # per area on the sphere
            # Hidden behind the disc?
            hidden = False
            if disc_on:
                base = np.arctan2(-ch[1], e[1])
                for j in range(-1, 5):
                    pc = base + j * np.pi
                    if pc <= 1e-9 or pc >= phi:
                        continue
                    if near:
                        uc = _u_after_phi_no_turn(u0, u_t, b_near, pc, gx, gw)
                    else:
                        uc = _u_at(table, fi, pc, dphi)
                    if uc > 0.0 and uc < 1.0:
                        rc = 1.0 / uc
                        if rc >= r_in and rc <= r_out:
                            hidden = True
            if hidden:
                continue
            # Redshift.
            sa = np.sin(a)
            b = r0 * sa / lapse0
            # Traced sweep normal n = ch x e; the real photon's L = -b n.
            nx = ch[1] * e[2] - ch[2] * e[1]
            ny = ch[2] * e[0] - ch[0] * e[2]
            nz = ch[0] * e[1] - ch[1] * e[0]
            # Sign of du/dphi at the body: from the table, across the hit.
            if near:
                sign = 1.0 if u_t > u0 else -1.0
            else:
                u_after = _u_at(table, fi, phi + 0.5 * dphi, dphi)
                u_before = _u_at(table, fi, max(phi - 0.5 * dphi, 0.0), dphi)
                sign = 1.0 if u_after > u_before else -1.0
            f = 1.0 - 2.0 / r
            pr_up = sign * np.sqrt(max(1.0 - f * b * b / (r * r), 0.0))
            p_r = pr_up / f
            wk = w[k]
            ut = e_body[k] / f
            ur = (p[0] * wk[0] + p[1] * wk[1] + p[2] * wk[2]) / r
            ox = (p[1] * wk[2] - p[2] * wk[1]) / (r * r)
            oy = (p[2] * wk[0] - p[0] * wk[2]) / (r * r)
            oz = (p[0] * wk[1] - p[1] * wk[0]) / (r * r)
            l_dot_omega = -b * (nx * ox + ny * oy + nz * oz)
            minus_pu = ut - p_r * ur - l_dot_omega
            solid /= max(abs(pr_up), 1e-3)  # per area across the ray
            g = 1.0 / (lapse0 * minus_pu) if minus_pu > 0.0 else 0.0
            ca = np.cos(a)
            for c in range(3):
                out_dir[k, order, c] = ca * ch[c] + sa * e[c]
            out_info[k, order, 0] = 1.0
            out_info[k, order, 1] = solid
            out_info[k, order, 2] = g
            out_info[k, order, 3] = a
            out_info[k, order, 4] = _light_delay(u0, u_t, b, ca < 0.0, sign > 0.0, gx, gw)


class Images(NamedTuple):
    """Images of n bodies; axis 1 is 0 for the direct image, 1 for the secondary one."""

    dirs: np.ndarray  # (n, 2, 3) unit directions on the camera's sky
    visible: np.ndarray  # (n, 2) bool
    solid: (
        np.ndarray
    )  # (n, 2) solid angle per unit cross-section: a body of radius R appears pi R^2 x
    g: np.ndarray  # (n, 2) nu_observed / nu_emitted
    delay: np.ndarray  # (n, 2) light-travel time to the camera [M]


def body_images(rays, cam, pos, w, e_body, r_in=6.0, r_out=20.0, disc_on=True):
    """Images of bodies at `pos` (velocities w = dx/dtau, energies e_body), seen from `cam`."""
    pos = np.ascontiguousarray(pos, dtype=float).reshape(-1, 3)
    n = len(pos)
    out_dir = np.zeros((n, 2, 3))
    out_info = np.zeros((n, 2, 5))
    if n:
        _images(
            rays.table,
            rays.alphas,
            DPHI,
            rays.r0,
            np.asarray(cam, dtype=float),
            pos,
            np.ascontiguousarray(w, dtype=float).reshape(-1, 3),
            np.ascontiguousarray(e_body, dtype=float),
            float(r_in),
            float(r_out),
            bool(disc_on),
            _GL_X,
            _GL_W,
            out_dir,
            out_info,
        )
    return Images(
        out_dir, out_info[..., 0] > 0.5, out_info[..., 1], out_info[..., 2], out_info[..., 4]
    )
