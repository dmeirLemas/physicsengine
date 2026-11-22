"""First post-Newtonian (1PN) gravity for the N-body module: the EIH equations.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20). Review
the physics.

The Einstein-Infeld-Hoffmann equations (Einstein, Infeld & Hoffmann 1938;
in this form e.g. Poisson & Will 2014, Gravity, eq. 9.127, harmonic
coordinates) give each body's acceleration to first order in (v/c)^2 and
G m / (r c^2). With n_ab = (x_a - x_b) / r_ab and a_b^N the Newtonian
acceleration of b:

    a_a = sum_b G m_b (x_b - x_a) / r_ab^3
        + 1/c^2 sum_b G m_b (x_b - x_a) / r_ab^3 [ v_a^2 + 2 v_b^2 - 4 v_a.v_b
              - 3/2 (n_ab.v_b)^2 - 4 sum_{c!=a} G m_c / r_ac
              - sum_{c!=b} G m_c / r_bc + 1/2 (x_b - x_a).a_b^N ]
        + 1/c^2 sum_b G m_b / r_ab^2 [ n_ab.(4 v_a - 3 v_b) ] (v_a - v_b)
        + 7 / (2 c^2) sum_b G m_b a_b^N / r_ab.

For a test body around one mass M at rest this is
    a = -G M / r^2 [ (1 - 4 G M / (r c^2) + v^2 / c^2) n - 4 (n.v) v / c^2 ],
the 1PN Schwarzschild orbit in harmonic coordinates, whose pericentre
advances by 6 pi G M / (c^2 a (1 - e^2)) per orbit: 43 arcseconds per
century for Mercury.

Conserved energy, E = sum v.dL/dv - L of the EIH Lagrangian (Landau &
Lifshitz, The Classical Theory of Fields, sec. 106; in this pairwise form
e.g. Blanchet 2014, Living Rev. Relativ. 17, 2):
    E = sum m v^2 / 2 + 3/8 sum m v^4 / c^2
      + sum_{a<b} G m_a m_b / r_ab [-1 + (3 (v_a^2 + v_b^2) - 7 v_a.v_b
                                         - (v_a.n_ab)(v_b.n_ab)) / (2 c^2)]
      + sum_a sum_{b!=a} sum_{c!=a} G^2 m_a m_b m_c / (2 c^2 r_ab r_ac).
For a test body: v^2/2 - GM/r + (3/8 v^4 + 3/2 GM v^2 / r + 1/2 G^2 M^2 / r^2) / c^2.

Gravitational waves. Orbiting masses radiate; the loss first shows up at
order (v/c)^5 ("2.5PN") as a force on each pair (Damour & Deruelle 1981,
Phys. Lett. A 87, 81; e.g. Blanchet 2014, Living Rev. Relativ. 17, 2,
eq. 203, harmonic coordinates). For the separation x = x_1 - x_2, with
M = m_1 + m_2, nu = m_1 m_2 / M^2, n = x / r, v = v_1 - v_2:

    d^2 x / dt^2 = 8 G^2 M^2 nu / (5 c^5 r^3)
                   [ (n.v) (3 v^2 + 17 G M / (3 r)) n - (v^2 + 3 G M / r) v ],

shared between the two bodies as m_2 / M and -m_1 / M (momentum is
conserved). On a circular orbit it shrinks the separation at Peters' rate
(Peters 1964, Phys. Rev. 136, B1224),
    da/dt = -64 G^3 m_1 m_2 M / (5 c^5 a^3),
so two bodies merge after T = 5 c^5 a^4 / (256 G^3 m_1 m_2 M). It is summed
over pairs (rr_acc); it only matters for compact, massive pairs (Earth
around the Sun: T ~ 1e23 years).

The waves themselves, far from the source, follow from the second time
derivative of the trace-free mass quadrupole Q_ij = sum m (x_i x_j -
delta_ij x^2 / 3) at the retarded time (quadrupole formula, Einstein 1918;
Misner, Thorne & Wheeler 1973, sec. 36.10):
    h_ij^TT (t, x) = 2 G / (c^4 r) [d^2 Q_ij / dt^2 (t - r / c)]^TT.
quadrupole_ddot returns that second derivative from positions, velocities
and accelerations: sum m (2 v_i v_j + x_i a_j + a_i x_j), trace removed.

1PN is accurate where v << c and r >> G M / c^2. Near a black hole (within
~10 G M / c^2) it is not; there the black-hole module's exact geodesics
apply. Units: the N-body module's (AU, years, solar masses).
"""

import numba
import numpy as np

C_AU_PER_YEAR = 299792.458 * 365.25 * 86400.0 / 1.495978707e8  # 63 241.08


def eih_acc(pos, vel, mass, G, c, debris=None):
    """1PN (EIH) accelerations of all bodies. Pairs of debris (nbodystate.DEBRIS) exert nothing on each other."""
    if debris is None:
        debris = np.zeros(len(mass), dtype=np.bool_)
    return _eih_acc(pos, vel, mass, G, c, debris)


@numba.njit(cache=True)
def _eih_acc(pos, vel, mass, G, c, debris):
    n = len(mass)
    acc = np.zeros((n, 3))
    if n < 2:
        return acc
    newton = np.zeros((n, 3))
    phi = np.zeros(n)  # sum_{c != a} G m_c / r_ac
    for a in range(n):
        for b in range(n):
            if b == a or (debris[a] and debris[b]):
                continue
            d = pos[b] - pos[a]
            r = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r == 0.0:
                continue  # two bodies at the same point exert nothing on each other
            newton[a] += G * mass[b] * d / r**3
            phi[a] += G * mass[b] / r
    inv_c2 = 1.0 / (c * c)
    for a in range(n):
        va = vel[a]
        va2 = va[0] ** 2 + va[1] ** 2 + va[2] ** 2
        corr = np.zeros(3)
        for b in range(n):
            if b == a or mass[b] == 0.0 or (debris[a] and debris[b]):
                continue
            d = pos[b] - pos[a]  # x_b - x_a
            r = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r == 0.0:
                continue
            nab = -d / r  # (x_a - x_b) / r_ab
            vb = vel[b]
            vb2 = vb[0] ** 2 + vb[1] ** 2 + vb[2] ** 2
            vavb = va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]
            nvb = nab[0] * vb[0] + nab[1] * vb[1] + nab[2] * vb[2]
            d_ab = d[0] * newton[b, 0] + d[1] * newton[b, 1] + d[2] * newton[b, 2]
            gm = G * mass[b]
            bracket = (
                va2 + 2.0 * vb2 - 4.0 * vavb - 1.5 * nvb * nvb - 4.0 * phi[a] - phi[b] + 0.5 * d_ab
            )
            corr += gm * d / r**3 * bracket
            w = nab[0] * (4.0 * va[0] - 3.0 * vb[0])
            w += nab[1] * (4.0 * va[1] - 3.0 * vb[1])
            w += nab[2] * (4.0 * va[2] - 3.0 * vb[2])
            corr += gm / r**2 * w * (va - vb)
            corr += 3.5 * gm * newton[b] / r
        acc[a] = newton[a] + inv_c2 * corr
    return acc


@numba.njit(cache=True)
def eih_energy(pos, vel, mass, G, c):
    """The conserved 1PN energy (Landau & Lifshitz sec. 106)."""
    n = len(mass)
    inv_c2 = 1.0 / (c * c)
    e = 0.0
    for a in range(n):
        v2 = vel[a, 0] ** 2 + vel[a, 1] ** 2 + vel[a, 2] ** 2
        e += 0.5 * mass[a] * v2 + 0.375 * mass[a] * v2 * v2 * inv_c2
    for a in range(n):
        va = vel[a]
        va2 = va[0] ** 2 + va[1] ** 2 + va[2] ** 2
        phi_a = 0.0
        for b in range(n):
            if b == a:
                continue
            d = pos[a] - pos[b]
            r = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r == 0.0:
                continue
            nab = d / r
            vb = vel[b]
            gmm = G * mass[a] * mass[b]
            e -= 0.5 * gmm / r
            vavb = va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]
            van = va[0] * nab[0] + va[1] * nab[1] + va[2] * nab[2]
            vbn = vb[0] * nab[0] + vb[1] * nab[1] + vb[2] * nab[2]
            vb2 = vb[0] ** 2 + vb[1] ** 2 + vb[2] ** 2
            e += gmm / (4.0 * r) * (3.0 * (va2 + vb2) - 7.0 * vavb - van * vbn) * inv_c2
            phi_a += G * mass[b] / r
        # sum_{b != a} sum_{c != a} G^2 m_a m_b m_c / (r_ab r_ac) = m_a phi_a^2
        e += 0.5 * mass[a] * phi_a * phi_a * inv_c2
    return e


def schwarzschild_radius(mass, G, c=C_AU_PER_YEAR):
    """2 G M / c^2: a black hole's horizon (radius used for absorbing bodies)."""
    return 2.0 * G * mass / c**2


@numba.njit(cache=True)
def rr_acc(pos, vel, mass, G, c):
    """Radiation-reaction (2.5PN) accelerations of all bodies, summed over pairs."""
    n = len(mass)
    acc = np.zeros((n, 3))
    c5 = c**5
    for a in range(n):
        for b in range(a + 1, n):
            total = mass[a] + mass[b]
            if mass[a] <= 0.0 or mass[b] <= 0.0:
                continue
            x = pos[a] - pos[b]
            v = vel[a] - vel[b]
            r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            if r == 0.0:
                continue
            v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
            nv = (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]) / r
            gm = G * total
            pref = 8.0 * G * G * mass[a] * mass[b] / (5.0 * c5 * r**3)  # G^2 M^2 nu = G^2 m_a m_b
            rel = pref * (nv * (3.0 * v2 + 17.0 * gm / (3.0 * r)) * x / r - (v2 + 3.0 * gm / r) * v)
            acc[a] += mass[b] / total * rel
            acc[b] -= mass[a] / total * rel
    return acc


def inspiral_time(m1, m2, a, G, c=C_AU_PER_YEAR):
    """Time for a circular binary of separation a to merge (Peters 1964): 5 c^5 a^4 / (256 G^3 m1 m2 M)."""
    return 5.0 * c**5 * a**4 / (256.0 * G**3 * m1 * m2 * (m1 + m2))


def circular_speed_1pn(m1, m2, r, G, c=C_AU_PER_YEAR):
    """Relative speed of a circular orbit of separation r at 1PN order (harmonic coordinates).

    omega^2 = G M / r^3 [1 - (3 - nu) G M / (r c^2)] (e.g. Blanchet 2014,
    eq. 228 to first order), nu = m1 m2 / M^2.
    """
    total = m1 + m2
    nu = m1 * m2 / total**2
    return np.sqrt(G * total / r * (1.0 - (3.0 - nu) * G * total / (r * c * c)))


def quadrupole_ddot(pos, vel, acc, mass):
    """Second time derivative of the trace-free mass quadrupole about the centre of mass (3x3)."""
    total = mass.sum()
    if total <= 0.0:
        return np.zeros((3, 3))
    x = pos - mass @ pos / total
    v = vel - mass @ vel / total
    a = acc - mass @ acc / total
    q = np.einsum("n,ni,nj->ij", mass, v, v) * 2.0
    xa = np.einsum("n,ni,nj->ij", mass, x, a)
    q += xa + xa.T
    return q - np.eye(3) * np.trace(q) / 3.0


# Fraction of the total mass radiated in the merger of two non-spinning
# black holes: 0.048 for equal masses, scaling as (4 nu)^2 (fit to numerical
# relativity, e.g. Berti et al. 2007, Phys. Rev. D 76, 064034).
MERGER_RADIATED = 0.048


def merger_mass(m1, m2):
    """Mass of the black hole left by the merger of two (see MERGER_RADIATED)."""
    total = m1 + m2
    nu = m1 * m2 / total**2
    return total * (1.0 - MERGER_RADIATED * (4.0 * nu) ** 2)
