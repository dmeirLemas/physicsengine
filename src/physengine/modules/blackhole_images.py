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
   that fell in +1. The equation is integrated by its Taylor series: with
   c_k = u^(k) / k!, differentiating u'' = 3u^2 - u k times gives
       (k + 2)(k + 1) c_(k+2) = 3 sum_(j=0..k) c_j c_(k-j) - c_k,
   so all c_k follow from u and u' at a point, and u(phi + t) = sum c_k t^k.
   One series (to t^TAYLOR_ORDER) gives TABLE_STRIDE columns; the next
   starts from its u and u' at the last of them (t <= 0.08; the series
   converges out to t ~ 1, the distance to the solution's nearest pole).
   Measured against RK4 with a hundredth of the table's step, the table is
   right to 4e-10 for all rays, those that wind round the photon sphere
   included (the RK4 with step dphi used before: 6e-10, at twice the cost).
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

Speed (2026-11-23): thousands of images are solved every frame. Step 2 is
now a bisection over the table's rows (12 steps) followed by the exact root
of the interpolation between the two rows found, instead of 48 bisection
steps on the interpolated function: the same root (to 1e-11 of a row), a
quarter of the table look-ups. The two neighbouring solves of step 4 start
from the rows of the first. The kernel solves one image per row (the module
needs the direct image of a body at one time and the secondary one at
another), and a table can be rebuilt in place for a new camera distance
(see step 1 for how it is filled).

Fixed with it: a body within the table's interpolation error of its ray's
turning point could come out just inside that point, where the travel-time
integrand is not real; the delay was then ~-1e145 (the module's clamp to
"now" hid it). _light_delay now takes such a body to be at the turning point.
"""

from typing import NamedTuple

import numba
import numpy as np

from physengine.modules.schwarzschild import shadow_angular_radius

N_ALPHA = 4096
DPHI = 0.01
TABLE_STRIDE = 8  # table columns per Taylor series
TAYLOR_ORDER = 8
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
def _build_table(r0, alphas, dphi, table):
    n = len(alphas)
    n_phi = table.shape[1]
    u0 = 1.0 / r0
    h = TABLE_STRIDE * dphi
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
        c = np.empty(TAYLOR_ORDER + 1)
        u = u0
        v = -u0 * np.sqrt(1.0 - 2.0 * u0) * np.cos(a) / sa
        table[i, 0] = u
        done = 0.0
        for k0 in range(0, n_phi - 1, TABLE_STRIDE):
            if done != 0.0:
                for k in range(k0 + 1, min(k0 + TABLE_STRIDE, n_phi - 1) + 1):
                    table[i, k] = done
                continue
            # Taylor coefficients of u about this column.
            c[0] = u
            c[1] = v
            for k in range(TAYLOR_ORDER - 1):
                total = 0.0
                for j in range(k + 1):
                    total += c[j] * c[k - j]
                c[k + 2] = (3.0 * total - c[k]) / ((k + 2) * (k + 1))
            for step in range(1, TABLE_STRIDE + 1):
                k = k0 + step
                if k > n_phi - 1:
                    break
                if done != 0.0:
                    table[i, k] = done
                    continue
                t = step * dphi
                uk = c[TAYLOR_ORDER]
                for j in range(TAYLOR_ORDER - 1, -1, -1):
                    uk = uk * t + c[j]
                if uk <= 0.0:
                    done = ESCAPED
                    table[i, k] = uk  # the crossing itself, for interpolation
                elif uk >= 0.5:
                    done = CAPTURED
                    table[i, k] = CAPTURED
                else:
                    table[i, k] = uk
            # u and u' at the end of the stride start the next series.
            u = c[TAYLOR_ORDER]
            v = TAYLOR_ORDER * c[TAYLOR_ORDER]
            for j in range(TAYLOR_ORDER - 1, 0, -1):
                u = u * h + c[j]
                v = v * h + j * c[j]
            u = u * h + c[0]


class RayTable:
    """u(alpha, phi) of camera rays for one camera distance r0.

    `reuse`: a RayTable that is no longer needed; its array is overwritten
    with the new table (a camera that moves needs a new table every frame,
    and filling 20 MB is cheaper than allocating it afresh).
    """

    def __init__(self, r0, reuse=None):
        self.r0 = float(r0)
        self.alphas = alpha_grid(self.r0)
        self.n_phi = int(np.ceil(PHI_MAX / DPHI)) + 1
        if reuse is not None and reuse.table.shape == (len(self.alphas), self.n_phi):
            self.table = reuse.table
            reuse.table = None
        else:
            self.table = np.empty((len(self.alphas), self.n_phi), dtype=np.float64)
        _build_table(self.r0, self.alphas, DPHI, self.table)


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
    turning point u_t in between: u0 -> u_t -> u_body. No point of such a
    ray has u > u_t; a body that comes out there (it is within the table's
    interpolation error of the turning point) is taken to be at it.
    """
    if inward_start and b > np.sqrt(27.0):
        ut = _turning_u(b)
        if not inward_at_body or u_body >= ut:
            ub = min(u_body, ut)
            return _time_between(u0, ut, b, True, gx, gw) + _time_between(ub, ut, b, True, gx, gw)
    return _time_between(u0, u_body, b, False, gx, gw)


@numba.njit(cache=True)
def _row_u(table, i, phi, dphi):
    """u of the table's ray i at angle phi (linear in phi): _u_at at a whole alpha index."""
    m = table.shape[1]
    x = phi / dphi
    k0 = min(max(int(x), 0), m - 2)
    fk = x - k0
    return table[i, k0] + fk * (table[i, k0 + 1] - table[i, k0])


@numba.njit(cache=True)
def _solve_alpha(table, phi, u_target, dphi, near):
    """Continuous alpha index of the ray at u_target after sweeping phi, or -1.

    Bisection over the rows for the two neighbouring rays with u below and
    not below u_target (u grows with alpha), then the root of _u_at between
    them: it is linear there, or, where one of the two rays has ended, a
    step at the middle. `near`: a row close to the answer (the search starts
    from the three intervals round it), or -1.
    """
    n = table.shape[0]
    lo, hi = 0, n - 1
    u_lo = 0.0
    u_hi = 0.0
    bracketed = False
    if near >= 0:
        lo, hi = max(near - 1, 0), min(near + 2, n - 1)
        u_lo = _row_u(table, lo, phi, dphi)
        u_hi = _row_u(table, hi, phi, dphi)
        bracketed = u_lo < u_target and u_hi >= u_target
    if not bracketed:
        lo, hi = 0, n - 1
        u_lo = _row_u(table, lo, phi, dphi)
        u_hi = _row_u(table, hi, phi, dphi)
        if u_lo > u_target or u_hi < u_target:
            return -1.0
        if u_lo == u_target:
            return 0.0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        u_mid = _row_u(table, mid, phi, dphi)
        if u_mid < u_target:
            lo, u_lo = mid, u_mid
        else:
            hi, u_hi = mid, u_mid
    if (u_lo == CAPTURED) != (u_hi == CAPTURED) or (u_lo == ESCAPED) != (u_hi == ESCAPED):
        return lo + 0.5
    return lo + (u_target - u_lo) / (u_hi - u_lo)


@numba.njit(cache=True)
def _alpha_of(alphas, fi):
    n = len(alphas)
    i0 = min(max(int(fi), 0), n - 2)
    f = fi - i0
    return alphas[i0] + f * (alphas[i0 + 1] - alphas[i0])


@numba.njit(parallel=True, cache=True)
def _images(
    table, alphas, dphi, r0, cam, pos, w, e_body, order, r_in, r_out, disc_on, gx, gw,
    out_dir, out_info,
):  # fmt: skip
    """One image of every row: of the body at pos[k], the direct one (order[k] = 0) or the secondary.

    out_info[k] = (visible, solid angle per area across the ray, g, alpha, delay).
    """
    n = len(pos)
    ch0, ch1, ch2 = cam[0] / r0, cam[1] / r0, cam[2] / r0
    lapse0 = np.sqrt(1.0 - 2.0 / r0)
    two_pi = 2.0 * np.pi
    u0 = 1.0 / r0
    for k in numba.prange(n):
        out_info[k, 0] = 0.0
        p0, p1, p2 = pos[k, 0], pos[k, 1], pos[k, 2]
        r = np.sqrt(p0**2 + p1**2 + p2**2)
        if not r > 2.0:
            continue
        cosd = min(max((p0 * ch0 + p1 * ch1 + p2 * ch2) / r, -1.0), 1.0)
        d = np.arccos(cosd)
        e0, e1, e2 = p0 / r - cosd * ch0, p1 / r - cosd * ch1, p2 / r - cosd * ch2
        etn = np.sqrt(e0**2 + e1**2 + e2**2)
        if etn < 1e-12:  # on the camera's axis: any plane will do
            e0, e1, e2 = -ch1 * ch0, 1.0 - ch1 * ch1, -ch1 * ch2
            if np.sqrt(e0**2 + e1**2 + e2**2) < 1e-6:
                e0, e1, e2 = 1.0 - ch0 * ch0, -ch0 * ch1, -ch0 * ch2
            etn = np.sqrt(e0**2 + e1**2 + e2**2)
        e0, e1, e2 = e0 / etn, e1 / etn, e2 / etn
        u_t = 1.0 / r
        phi = d
        if order[k] != 0:
            phi = two_pi - d
            e0, e1, e2 = -e0, -e1, -e2
        near = order[k] == 0 and phi < NEAR_AXIS
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
            fi = _solve_alpha(table, phi, u_t, dphi, -1)
            if fi < 0.0:
                continue
            a = _alpha_of(alphas, fi)
            # Magnification: alpha at phi +- delta.
            delta = 2e-3
            fa = _solve_alpha(table, phi + delta, u_t, dphi, int(fi))
            fb = _solve_alpha(table, max(phi - delta, 1e-6), u_t, dphi, int(fi))
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
            base = np.arctan2(-ch1, e1)
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
        nx = ch1 * e2 - ch2 * e1
        ny = ch2 * e0 - ch0 * e2
        nz = ch0 * e1 - ch1 * e0
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
        w0, w1, w2 = w[k, 0], w[k, 1], w[k, 2]
        ut = e_body[k] / f
        ur = (p0 * w0 + p1 * w1 + p2 * w2) / r
        ox = (p1 * w2 - p2 * w1) / (r * r)
        oy = (p2 * w0 - p0 * w2) / (r * r)
        oz = (p0 * w1 - p1 * w0) / (r * r)
        l_dot_omega = -b * (nx * ox + ny * oy + nz * oz)
        minus_pu = ut - p_r * ur - l_dot_omega
        solid /= max(abs(pr_up), 1e-3)  # per area across the ray
        g = 1.0 / (lapse0 * minus_pu) if minus_pu > 0.0 else 0.0
        ca = np.cos(a)
        out_dir[k, 0] = ca * ch0 + sa * e0
        out_dir[k, 1] = ca * ch1 + sa * e1
        out_dir[k, 2] = ca * ch2 + sa * e2
        out_info[k, 0] = 1.0
        out_info[k, 1] = solid
        out_info[k, 2] = g
        out_info[k, 3] = a
        out_info[k, 4] = _light_delay(u0, u_t, b, ca < 0.0, sign > 0.0, gx, gw)


class Images(NamedTuple):
    """Images of n bodies; axis 1 is 0 for the direct image, 1 for the secondary one."""

    dirs: np.ndarray  # (n, 2, 3) unit directions on the camera's sky
    visible: np.ndarray  # (n, 2) bool
    solid: (
        np.ndarray
    )  # (n, 2) solid angle per unit cross-section: a body of radius R appears pi R^2 x
    g: np.ndarray  # (n, 2) nu_observed / nu_emitted
    delay: np.ndarray  # (n, 2) light-travel time to the camera [M]


def image_rows(rays, cam, pos, w, e_body, order, r_in=6.0, r_out=20.0, disc_on=True):
    """One image per row: of a body at pos[k] (velocity w[k] = dx/dtau, energy e_body[k]) seen
    from `cam`, the direct image where order[k] is 0, the secondary one where it is 1.

    Returns Images with one entry per row (no axis for the order).
    """
    pos = np.ascontiguousarray(pos, dtype=float).reshape(-1, 3)
    n = len(pos)
    out_dir = np.zeros((n, 3))
    out_info = np.zeros((n, 5))
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
            np.ascontiguousarray(np.broadcast_to(order, n), dtype=np.int64),
            float(r_in),
            float(r_out),
            bool(disc_on),
            _GL_X,
            _GL_W,
            out_dir,
            out_info,
        )
    return Images(out_dir, out_info[:, 0] > 0.5, out_info[:, 1], out_info[:, 2], out_info[:, 4])


def body_images(rays, cam, pos, w, e_body, r_in=6.0, r_out=20.0, disc_on=True):
    """Both images of bodies at `pos` (velocities w = dx/dtau, energies e_body), seen from `cam`."""
    pos = np.ascontiguousarray(pos, dtype=float).reshape(-1, 3)
    w = np.ascontiguousarray(w, dtype=float).reshape(-1, 3)
    e_body = np.ascontiguousarray(e_body, dtype=float)
    n = len(pos)
    both = image_rows(
        rays,
        cam,
        np.concatenate([pos, pos]),
        np.concatenate([w, w]),
        np.concatenate([e_body, e_body]),
        np.repeat([0, 1], n),
        r_in,
        r_out,
        disc_on,
    )
    # Rows 0..n-1 are the direct images, n..2n-1 the secondary ones.
    return Images(*(np.stack([q[:n], q[n:]], axis=1) for q in both))
