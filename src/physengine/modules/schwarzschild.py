"""Schwarzschild black hole: reference formulas for the black-hole module.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). Review the physics.

Units: G = c = 1 and lengths in units of the mass M (geometric units), so the
horizon is at r = 2, the photon sphere at r = 3 and the innermost stable
circular orbit at r = 6. r is the Schwarzschild (areal) radius.

Light rays move in a plane through the hole. With u = 1/r and phi the angle
in that plane, a ray obeys (e.g. Hartle 2003, Gravity, eq. 9.48; Chandrasekhar
1983, ch. 3)
    (du/dphi)^2 = 1/b^2 - u^2 (1 - 2 M u),     d^2u/dphi^2 = 3 M u^2 - u,
with b = L / E the impact parameter. The shader (blackhole_renderer.py)
integrates the second form; the functions here give exact results to check
it against.

A static observer at radius r who sees a ray at angle alpha from the outward
radial direction measures b = r sin(alpha) / sqrt(1 - 2M/r) (Synge 1966,
MNRAS 131, 463).
"""

import numpy as np

M = 1.0
R_HORIZON = 2.0 * M
R_PHOTON = 3.0 * M  # circular light orbits
R_ISCO = 6.0 * M  # innermost stable circular orbit of massive bodies
B_CRITICAL = np.sqrt(27.0) * M  # rays with b < this fall in


def impact_parameter(r_obs, alpha):
    """b of a ray at angle alpha from the outward radial direction, for a static observer at r_obs."""
    return r_obs * np.sin(alpha) / np.sqrt(1.0 - 2.0 * M / r_obs)


def shadow_angular_radius(r_obs):
    """Angular radius of the shadow seen by a static observer at r_obs (Synge 1966).

    sin(alpha) = sqrt(27) M sqrt(1 - 2M/r) / r. The shadow is centred on the
    direction to the hole; inside the photon sphere it covers more than half
    the sky, so alpha is measured from the outward direction there.
    """
    s = np.clip(B_CRITICAL * np.sqrt(1.0 - 2.0 * M / r_obs) / r_obs, 0.0, 1.0)
    a = np.arcsin(s)
    return np.where(r_obs >= R_PHOTON, a, np.pi - a)


def turning_point(b):
    """u = 1/r at the closest approach of a ray with impact parameter b > sqrt(27) M.

    The smallest positive root of 1/b^2 - u^2 + 2 M u^3 = 0, which for r is
    r = (2 b / sqrt 3) cos(arccos(-3 sqrt(3) M / b) / 3) (e.g. Darwin 1959).
    """
    r = 2.0 * b / np.sqrt(3.0) * np.cos(np.arccos(-3.0 * np.sqrt(3.0) * M / b) / 3.0)
    return 1.0 / r


def _phi_between(b, u_a, u_b, n=200):
    """Angle swept while u goes from u_a to u_b (u_a < u_b <= turning point), by quadrature.

    The integrand 1 / sqrt(P(u)), P = 1/b^2 - u^2 + 2 M u^3, has an inverse
    square-root singularity at the turning point; the substitution
    u = u_t - (u_t - u_a) s^2 removes it when u_b is the turning point.
    """
    u_t = turning_point(b) if b > B_CRITICAL else np.inf
    x, w = np.polynomial.legendre.leggauss(n)
    if np.isfinite(u_t) and np.isclose(u_b, u_t, rtol=0, atol=1e-14):
        s_lo = 0.0
        s_hi = 1.0
        s = 0.5 * (x + 1.0) * (s_hi - s_lo) + s_lo
        u = u_t - (u_t - u_a) * s**2
        du_ds = 2.0 * (u_t - u_a) * s
        p = 1.0 / b**2 - u**2 + 2.0 * M * u**3
        return 0.5 * (s_hi - s_lo) * np.sum(w * du_ds / np.sqrt(np.maximum(p, 1e-300)))
    u = 0.5 * (x + 1.0) * (u_b - u_a) + u_a
    p = 1.0 / b**2 - u**2 + 2.0 * M * u**3
    return 0.5 * (u_b - u_a) * np.sum(w / np.sqrt(p))


def escape_angle(r_obs, alpha):
    """Total angle phi a ray swept from the observer to infinity, or None if it falls in.

    The ray leaves a static observer at r_obs at angle alpha from the outward
    radial direction. Measured in the ray's plane from the observer's
    position, in the direction the ray turns; its direction at infinity is
    cos(phi) e_r + sin(phi) e_t, e_t the sideways part of the starting
    direction.
    """
    b = impact_parameter(r_obs, alpha)
    u0 = 1.0 / r_obs
    if np.cos(alpha) >= 0.0:  # outward: straight on to infinity
        return _phi_between(b, 0.0, u0) if b > 0 else 0.0
    if b <= B_CRITICAL:
        return None
    u_t = turning_point(b)
    if u_t < u0:  # (observer inside the photon sphere, turning point beyond it)
        return None
    return _phi_between(b, u0, u_t) + _phi_between(b, 0.0, u_t)


def deflection_angle(b):
    """Bending of a ray from infinity to infinity: 2 phi(0 -> u_t) - pi."""
    u_t = turning_point(b)
    return 2.0 * _phi_between(b, 0.0, u_t) - np.pi


# ------------------------------------------------------------------ the disc


def disc_flux_shape(r, r_in=R_ISCO):
    """Radial profile of the flux from a thin accretion disc, up to a constant.

    F(r) proportional to (1 - sqrt(r_in / r)) / r^3: the Shakura & Sunyaev
    (1973, A&A 24, 337) thin disc with no torque at its inner edge. Newtonian;
    the fully relativistic profile (Page & Thorne 1974, ApJ 191, 499) has the
    same shape but differs by tens of percent within a few r_in. Peaks at
    r = 49/36 r_in.
    """
    r = np.asarray(r, dtype=float)
    return np.where(r > r_in, (1.0 - np.sqrt(r_in / np.maximum(r, r_in))) / r**3, 0.0)


def disc_flux_max(r_in=R_ISCO):
    return float(disc_flux_shape(49.0 / 36.0 * r_in, r_in))


def disc_redshift(r, lz_over_e, r_obs):
    """g = nu_observed / nu_emitted for light from disc gas on a circular orbit.

    Gas at radius r on a circular geodesic, angular velocity
    Omega = sqrt(M / r^3), four-velocity u^t (1, 0, 0, Omega) with
    u^t = 1 / sqrt(1 - 3M/r). A photon with energy E and angular momentum L_z
    about the disc axis has nu_emitted proportional to u^t (E - Omega L_z);
    a static observer at r_obs measures nu proportional to
    E / sqrt(1 - 2M/r_obs). So
        g = sqrt(1 - 3M/r) / sqrt(1 - 2M/r_obs) / (1 - Omega L_z / E)
    (e.g. Luminet 1979, A&A 75, 228).
    """
    omega = np.sqrt(M / r**3)
    return np.sqrt(1.0 - 3.0 * M / r) / np.sqrt(1.0 - 2.0 * M / r_obs) / (1.0 - omega * lz_over_e)
