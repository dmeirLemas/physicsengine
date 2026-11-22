"""Rotating (Kerr) black hole: formulas and a reference photon integrator.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). Review the physics.

Units G = c = M = 1, spin a = J / M in [0, 1). Boyer-Lindquist coordinates
(t, r, theta, phi) with the spin along the world's +y axis; on screen the
oblate-spheroidal map
    x = sqrt(r^2 + a^2) sin(theta) cos(phi),  y = r cos(theta),
    z = -sqrt(r^2 + a^2) sin(theta) sin(phi),
so phi grows in the sense of the rotation (counter-clockwise seen from +y),
like the disc. For a = 0 this is the Schwarzschild picture of
schwarzschild.py.

Metric functions (Bardeen, Press & Teukolsky 1972, ApJ 178, 347):
    Sigma = r^2 + a^2 cos^2(theta),  Delta = r^2 - 2r + a^2,
    A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta),  omega = 2 a r / A,
    e^nu = sqrt(Sigma Delta / A),  e^psi = sqrt(A / Sigma) sin(theta),
    e^mu1 = sqrt(Sigma / Delta),  e^mu2 = sqrt(Sigma).
The camera is a zero-angular-momentum observer (ZAMO, frame dragged with
omega). A photon seen in the ZAMO frame moving along unit vector
n = (n_r, n_theta, n_phi) with local energy 1 has
    p_r = e^mu1 n_r, p_theta = e^mu2 n_theta, L = p_phi = e^psi n_phi,
    E = -p_t = e^nu + omega L,
and Carter's constant Q = p_theta^2 + cos^2(theta) (L^2 / sin^2(theta) - a^2 E^2).

Geodesics in Mino time lambda (d/dlambda = Sigma d/dsigma, sigma affine;
Carter 1968; e.g. Gralla & Lupsasca 2020, PRD 101, 044031):
    (dr/dlambda)^2 = R(r) = (E (r^2 + a^2) - a L)^2 - Delta (mu^2 r^2 + (L - a E)^2 + Q),
    (dtheta/dlambda)^2 = Theta = Q - cos^2(theta) (a^2 (mu^2 - E^2) + L^2 / sin^2(theta)),
    dphi/dlambda = -(a E - L / sin^2(theta)) + a P / Delta,
    dt/dlambda = -a (a E sin^2(theta) - L) + (r^2 + a^2) P / Delta,
with P = E (r^2 + a^2) - a L and mu = 0 for light, 1 for a body. Integrated
in the second-order form r'' = R'(r) / 2, theta'' = Theta'(theta) / 2, which
passes the turning points without sign bookkeeping.

The disc: circular prograde equatorial orbits (Bardeen et al. 1972)
    Omega = 1 / (r^1.5 + a),
    E = (r^1.5 - 2 r^0.5 + a) / (r^0.75 sqrt(r^1.5 - 3 r^0.5 + 2a)),
    L = (r^2 - 2 a r^0.5 + a^2) / (r^0.75 sqrt(r^1.5 - 3 r^0.5 + 2a)),
from the innermost stable one (isco). Flux of a thin disc with no torque at
its inner edge (Page & Thorne 1974, ApJ 191, 499, eq. 15n in integral form):
    F(r) proportional to -Omega' / ((E - Omega L)^2 r) * integral_{r_isco}^r (E - Omega L) L' dr.
"""

import numba
import numpy as np


def horizon(a):
    """Outer horizon r_+ = 1 + sqrt(1 - a^2)."""
    return 1.0 + np.sqrt(1.0 - a * a)


def isco(a):
    """Radius of the innermost stable prograde circular orbit (Bardeen et al. 1972, eq. 2.21)."""
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def circular_orbit(r, a):
    """(E, L, Omega) of the prograde circular equatorial orbit at r (Bardeen et al. 1972)."""
    r = np.asarray(r, dtype=float)
    sq = np.sqrt(r)
    den = r**0.75 * np.sqrt(r**1.5 - 3.0 * sq + 2.0 * a)
    E = (r**1.5 - 2.0 * sq + a) / den
    L = (r * r - 2.0 * a * sq + a * a) / den
    return E, L, 1.0 / (r**1.5 + a)


def disc_flux_table(a, r_out=200.0, n=4000):
    """(r, F / F_max) of the Page-Thorne flux from the isco to r_out (integral form above)."""
    r_in = isco(a)
    r = r_in + (r_out - r_in) * np.linspace(0.0, 1.0, n) ** 2  # dense near the isco
    E, L, Om = circular_orbit(r, a)
    dL = np.gradient(L, r)
    dOm = np.gradient(Om, r)
    integrand = (E - Om * L) * dL
    integral = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))]
    )
    F = -dOm / ((E - Om * L) ** 2 * r) * integral
    F = np.maximum(F, 0.0)
    return r, F / F.max()


def metric(r, th, a):
    """(Sigma, Delta, A, omega, e^nu, e^psi, e^mu1, e^mu2) at (r, theta)."""
    s, c = np.sin(th), np.cos(th)
    sig = r * r + a * a * c * c
    dl = r * r - 2.0 * r + a * a
    A = (r * r + a * a) ** 2 - a * a * dl * s * s
    return (
        sig,
        dl,
        A,
        2.0 * a * r / A,
        np.sqrt(sig * dl / A),
        np.sqrt(A / sig) * s,
        np.sqrt(sig / dl),
        np.sqrt(sig),
    )


def to_world(r, th, ph, a):
    """World position of Boyer-Lindquist (r, theta, phi)."""
    R = np.sqrt(r * r + a * a)
    return np.array([R * np.sin(th) * np.cos(ph), r * np.cos(th), -R * np.sin(th) * np.sin(ph)])


def from_world(p, a):
    """Boyer-Lindquist (r, theta, phi) of a world position."""
    x, y, z = p
    rho2 = x * x + y * y + z * z
    r2 = 0.5 * (rho2 - a * a + np.sqrt((rho2 - a * a) ** 2 + 4.0 * a * a * y * y))
    r = np.sqrt(r2)
    th = np.arccos(np.clip(y / r, -1.0, 1.0)) if r > 0 else 0.0
    return r, th, np.arctan2(-z, x)


def local_axes(r, th, ph, a):
    """World unit vectors along d/dr, d/dtheta, d/dphi at a point (mutually orthogonal)."""
    R = np.sqrt(r * r + a * a)
    s, c, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    er = np.array([r / R * s * cp, c, -r / R * s * sp])
    et = np.array([R * c * cp, -r * s, -R * c * sp])
    ep = np.array([-sp, 0.0, -cp])
    return er / np.linalg.norm(er), et / np.linalg.norm(et), ep


def photon_from_camera(cam_world, direction, a):
    """(r, theta, phi, dr/dlambda, dtheta/dlambda, L, Q, E0) of the photon a ZAMO camera sees in `direction`.

    The photon arrives moving along -direction; these are its constants and
    its velocities at the camera (follow it back with trace_photon). E0 is
    the photon's E for local energy 1: the camera measures nu = 1 / E0 for a
    photon normalised to E = 1, which is what L and Q are.
    """
    r, th, ph = from_world(cam_world, a)
    er, et, ep = local_axes(r, th, ph, a)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    n_r, n_t, n_p = -(d @ er), -(d @ et), -(d @ ep)
    sig, dl, A, om, enu, epsi, emu1, emu2 = metric(r, th, a)
    L0 = epsi * n_p
    E0 = enu + om * L0
    L = L0 / E0
    p_r = emu1 * n_r / E0
    p_t = emu2 * n_t / E0
    Q = p_t**2 + np.cos(th) ** 2 * (L * L / np.sin(th) ** 2 - a * a)
    return r, th, ph, dl * p_r, p_t, L, Q, E0


@numba.njit(cache=True)
def _photon_rhs(y, a, L, Q):
    """d/dlambda of (r, theta, phi, t, rdot, thetadot) for a photon with E = 1."""
    r, th, rd, td = y[0], y[1], y[4], y[5]
    s, c = np.sin(th), np.cos(th)
    s = max(abs(s), 1e-12) * (1.0 if s >= 0 else -1.0)
    dl = r * r - 2.0 * r + a * a
    P = r * r + a * a - a * L
    dR = 4.0 * r * P - (2.0 * r - 2.0) * ((L - a) ** 2 + Q)
    dTh = -2.0 * a * a * c * s + 2.0 * L * L * c / (s * s * s)
    out = np.empty(6)
    out[0] = rd
    out[1] = td
    out[2] = -(a - L / (s * s)) + a * P / dl
    out[3] = -a * (a * s * s - L) + (r * r + a * a) * P / dl
    out[4] = 0.5 * dR
    out[5] = 0.5 * dTh
    return out


@numba.njit(cache=True)
def trace_photon(r0, th0, ph0, rd0, td0, L, Q, a, r_in, r_out, r_far, eta, max_steps):
    """Follow a photon back from the camera (lambda decreasing). Returns (outcome, r, theta, phi, t, rdot, thetadot).

    outcome: 0 came from the horizon (black), 1 came from far away (state at
    r_far), 2 came from the disc (state at the crossing), 3 still going after
    max_steps. Steps keep the changes of r (relative), theta and phi below
    eta. t is the coordinate time back from the camera (negative).
    """
    y = np.array([r0, th0, ph0, 0.0, rd0, td0])
    r_h = 1.0 + np.sqrt(1.0 - a * a)
    for _ in range(max_steps):
        k1 = _photon_rhs(y, a, L, Q)
        scale = max(abs(k1[0]) / y[0], abs(k1[1]), abs(k1[2]), 1e-9)
        h = -eta / scale
        k2 = _photon_rhs(y + 0.5 * h * k1, a, L, Q)
        k3 = _photon_rhs(y + 0.5 * h * k2, a, L, Q)
        k4 = _photon_rhs(y + h * k3, a, L, Q)
        yn = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        c0, c1 = np.cos(y[1]), np.cos(yn[1])
        if r_out > 0.0 and (c0 > 0.0) != (c1 > 0.0):
            f = c0 / (c0 - c1)
            rc = y[0] + f * (yn[0] - y[0])
            if rc >= r_in and rc <= r_out:
                ym = y + f * (yn - y)
                return 2, ym[0], ym[1], ym[2], ym[3], ym[4], ym[5]
        y = yn
        if y[0] <= r_h * 1.0001:
            return 0, y[0], y[1], y[2], y[3], y[4], y[5]
        if y[0] >= r_far and y[4] < 0.0:
            return 1, y[0], y[1], y[2], y[3], y[4], y[5]
    return 3, y[0], y[1], y[2], y[3], y[4], y[5]


def escape_direction(r, th, ph, rd, td, L, a):
    """World direction on the sky a photon came from: minus its local (ZAMO-frame) direction of travel far out."""
    sig, dl, A, om, enu, epsi, emu1, emu2 = metric(r, th, a)
    p_r = rd / dl
    n = np.array([p_r / emu1, td / emu2, L / epsi])
    er, et, ep = local_axes(r, th, ph, a)
    d = n[0] * er + n[1] * et + n[2] * ep
    return -d / np.linalg.norm(d)


def disc_redshift(r, L, E0, a, mode=0):
    """g = nu_obs / nu_emit for disc light (photon E normalised to 1, camera's E0).

    mode 0: gas on prograde circular orbits, u^t (1, 0, 0, Omega),
    u^t = (r^1.5 + a) / (r^0.75 sqrt(r^1.5 - 3 r^0.5 + 2a)).
    mode 1: gas at rest in the ZAMO frame (no orbital Doppler):
    nu_emit = (1 - omega L) / e^nu.
    """
    if mode == 0:
        ut = (r**1.5 + a) / (r**0.75 * np.sqrt(r**1.5 - 3.0 * np.sqrt(r) + 2.0 * a))
        Om = 1.0 / (r**1.5 + a)
        return 1.0 / (E0 * ut * (1.0 - Om * L))
    _, _, _, om, enu, _, _, _ = metric(r, np.pi / 2, a)
    return enu / (E0 * (1.0 - om * L))


def shadow_edge(a, inclination, n=400):
    """The shadow's outline for a distant observer (Bardeen 1973): screen (alpha, beta) in M.

    Spherical photon orbits at r between the prograde and retrograde circular
    photon orbits have xi = L / E and eta = Q / E^2 of
        xi = -(r^3 - 3 r^2 + a^2 r + a^2) / (a (r - 1)),
        eta = r^3 (4 a^2 - r (r - 3)^2) / (a^2 (r - 1)^2);
    seen at inclination i (from the spin axis) at alpha = -xi / sin(i),
    beta = +-sqrt(eta + a^2 cos^2(i) - xi^2 cot^2(i)).
    """
    r1 = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
    r2 = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(a)))
    r = np.linspace(r1, r2, n)
    xi = -(r**3 - 3 * r**2 + a * a * r + a * a) / (a * (r - 1.0))
    et = r**3 * (4 * a * a - r * (r - 3) ** 2) / (a * a * (r - 1.0) ** 2)
    b2 = et + a * a * np.cos(inclination) ** 2 - xi**2 / np.tan(inclination) ** 2
    ok = b2 >= 0
    alpha = -xi[ok] / np.sin(inclination)
    beta = np.sqrt(b2[ok])
    return alpha, beta
