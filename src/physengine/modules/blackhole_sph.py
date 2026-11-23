"""Gas dynamics of a disrupted star's debris: smoothed-particle hydrodynamics (SPH).

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
3). Review the physics.

A star is gas. Once it is torn, its debris has pressure, and where streams
of it collide, shocks turn their motion into heat. SPH (Gingold & Monaghan
1977, MNRAS 181, 375; Lucy 1977, AJ 82, 1013; the formulation here is the
standard one of Monaghan 1992, ARA&A 30, 543) follows the gas as particles of
fixed mass m, each smoothed over a length h with the cubic-spline kernel W
(support 2h):

    density        rho_i = sum_j m_j W(r_ij, h_i),
                   h_i = ETA (m_i / rho_i)^(1/3)   (about 60 neighbours)
    pressure       P = (gamma - 1) rho u,  gamma = 5/3  (ideal monatomic gas,
                   u = thermal energy per unit mass; no cooling)
    acceleration   a_i = - sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2 + Pi_ij) grad_i W(r_ij, h_ij)
    heating        du_i/dt = 1/2 sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2 + Pi_ij) v_ij . grad_i W
    with h_ij = (h_i + h_j) / 2.

Pi_ij is the artificial viscosity that makes shocks possible (Monaghan &
Gingold 1983, J. Comput. Phys. 52, 374): for approaching pairs (v_ij . r_ij < 0)
    mu = h_ij v_ij . r_ij / (r_ij^2 + 0.01 h_ij^2),
    Pi_ij = (-ALPHA c_ij mu + BETA mu^2) / rho_ij    (means of sound speed, density),
zero otherwise. Pi_ij also acts in a shear flow (gas orbiting at different
radii is "approaching" along one diagonal), where there is no shock; it is
therefore multiplied by the mean of the two particles' Balsara factors
(Balsara 1995, J. Comput. Phys. 121, 357),
    f = |div v| / (|div v| + |curl v| + 1e-4 c / h),
close to 1 in a compression and to 0 in pure shear. The pair forces are equal and opposite and the heating
matches the work done, so momentum and total energy are conserved.

The particles' mutual gravity stays where it was (geodesics.newtonian_mutual);
these accelerations are added to it. All of this is Newtonian physics applied
in each particle's local frame, as the mutual gravity already is (velocities are
w = dx/dtau; the gas moves on the hole's exact geodesics in between). That
is how relativistic disruptions were first simulated (e.g. Laguna et al.
1993, ApJ 410, L83, used SPH on a fixed Schwarzschild background); it is
accurate while neighbouring gas moves slowly relative to itself.

What this does not have:
- Resolution. A few thousand particles is 100 to 1000 times fewer than
  research runs; the debris stream is thinner than h, so its shocks are
  smeared and their dissipation overestimated.
- Magnetic fields. In a real disc the turbulence they drive carries angular
  momentum outward and lets the gas sink (Balbus & Hawley 1991, ApJ 376,
  214). Here only the artificial viscosity does that; it acts like an
  alpha ~ 0.1 ALPHA h / H disc (Lodato & Price 2010, MNRAS 405, 1212), which at
  this resolution is of the same order as the real one, by accident of the
  particle number.
- Radiation transport. Shocked debris is hot and radiates (that light is
  how disruptions are found); how fast it cools is an open question. Two
  limits are offered: no cooling (the heat of the shocks and of the accretion
  stays in the gas, which swells into a hot cloud), or cooling on a time
  BETA_COOL / Omega, Omega = r^-1.5 the local orbital frequency (the "beta
  cooling" used for discs, Gammie 2001, ApJ 553, 174), with which the gas
  settles into a disc.

A ring of gas (the "Gas ring" scene). Gas on circular orbits shears: inner
gas orbits faster than outer. Friction between neighbouring rings moves
angular momentum outward, so the inner gas sinks towards the hole and the
outer gas spreads out (Lynden-Bell & Pringle 1974, MNRAS 168, 603). In real
discs that friction is magnetic turbulence, summarised as a viscosity
nu = alpha c_s H (Shakura & Sunyaev 1973, A&A 24, 337). In SPH the same job
is done by the artificial viscosity when it is allowed to act on shear (no
Balsara factor, `shear_viscosity`), which is how SPH discs are routinely run
(Lodato & Price 2010): its strength is set by ALPHA times `viscosity`, the
user's setting. The draining time is the viscous time r^2 / nu, far longer
than an orbit. The friction also heats the gas; a thin disc radiates that
heat away as fast as it is made, which is modelled by holding the gas at the
temperature of a disc of fixed aspect ratio H / r ("locally isothermal", the
usual choice for SPH discs): c_s = (H / r) r^-1/2, u = c_s^2 / (gamma (gamma - 1))
(`disc_aspect`). Without it the heat stays and the ring swells into a cloud.

A star before disruption: the Plummer sphere the particles are drawn from is
the n = 5 polytrope, in hydrostatic balance with P / rho = m / (6 sqrt(r^2 + a^2)),
so its particles start at rest with u = P / ((gamma - 1) rho) (plummer_energy).

Finding the neighbours (2026-11-23). The sums above only run over particles
closer than 2h. `density` and `forces` test every pair (N^2 distances, three
times per evaluation); `density_tree` and `forces_tree`, which Gas uses,
find the same pairs through a tree of boxes (build_tree): the gas is halved
along its longest side until a box holds LEAF_SIZE particles at most, and a
box is skipped whole when it is out of reach: for the density, when its
nearest point is 2 h_i away; for the forces (a pair interacts within
2 h_ij = h_i + h_j), when the box round its particles' spheres of radius
h_j (sphere_boxes) is h_i away. No pair is left out or approximated: the
two ways give the same sums up to rounding (a test compares them). A tree and not a grid because h spans orders of magnitude between
the core of a star and its thin debris.
"""

import numba
import numpy as np

GAMMA = 5.0 / 3.0
ETA = 1.2
ALPHA = 1.0
BETA = 2.0
COURANT = 0.3
H_MAX = 8.0  # [M] smoothing length of an isolated particle (Gas.h_max: per scene)
BETA_COOL = 3.0


def plummer_energy(dx, mass, scale):
    """Thermal energy per unit mass of gas in hydrostatic balance in a Plummer sphere.

    `dx` are positions relative to its centre, `mass` its total mass, `scale` a.
    """
    r2 = np.sum(np.asarray(dx) ** 2, axis=1)
    return mass / (6.0 * (GAMMA - 1.0) * np.sqrt(r2 + scale * scale))


@numba.njit(cache=True)
def _w(q, h):
    s = 1.0 / (np.pi * h * h * h)
    if q < 1.0:
        return s * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    if q < 2.0:
        return s * 0.25 * (2.0 - q) ** 3
    return 0.0


@numba.njit(cache=True)
def _dw(q, h):
    """dW/dr."""
    s = 1.0 / (np.pi * h * h * h * h)
    if q < 1.0:
        return s * (-3.0 * q + 2.25 * q * q)
    if q < 2.0:
        return -s * 0.75 * (2.0 - q) ** 2
    return 0.0


@numba.njit(parallel=True, cache=True)
def density(x, v, m, h, u, gas, rho, shock):
    """rho_i = sum_j m_j W(r_ij, h_i) over gas particles (including i itself), and
    the Balsara factor of each (from div v and curl v, same sums with grad W)."""
    n = len(x)
    for i in numba.prange(n):
        shock[i] = 1.0
        if not gas[i]:
            rho[i] = 0.0
            continue
        total = 0.0
        div = 0.0
        c0 = c1 = c2 = 0.0
        hi = h[i]
        for j in range(n):
            if not gas[j]:
                continue
            d0 = x[i, 0] - x[j, 0]
            d1 = x[i, 1] - x[j, 1]
            d2 = x[i, 2] - x[j, 2]
            r2 = d0 * d0 + d1 * d1 + d2 * d2
            if r2 >= 4.0 * hi * hi:
                continue
            r = np.sqrt(r2)
            total += m[j] * _w(r / hi, hi)
            if r > 0.0:
                g = m[j] * _dw(r / hi, hi) / r  # m_j grad_i W = g * d
                v0 = v[i, 0] - v[j, 0]
                v1 = v[i, 1] - v[j, 1]
                v2 = v[i, 2] - v[j, 2]
                div -= g * (v0 * d0 + v1 * d1 + v2 * d2)
                c0 += g * (v1 * d2 - v2 * d1)
                c1 += g * (v2 * d0 - v0 * d2)
                c2 += g * (v0 * d1 - v1 * d0)
        rho[i] = total
        div = abs(div) / total
        curl = np.sqrt(c0 * c0 + c1 * c1 + c2 * c2) / total
        c = np.sqrt(GAMMA * (GAMMA - 1.0) * u[i])
        shock[i] = div / (div + curl + 1e-4 * c / hi + 1e-300)


@numba.njit(parallel=True, cache=True)
def forces(x, v, m, h, rho, u, gas, shock, alpha, beta, acc, dudt, dt_i):
    """Pressure and artificial-viscosity accelerations, heating rates, Courant times."""
    n = len(x)
    for i in numba.prange(n):
        acc[i, 0] = acc[i, 1] = acc[i, 2] = 0.0
        dudt[i] = 0.0
        dt_i[i] = np.inf
        if not gas[i]:
            continue
        pi_rho2 = (GAMMA - 1.0) * u[i] / rho[i]
        ci = np.sqrt(GAMMA * (GAMMA - 1.0) * u[i])
        a0 = a1 = a2 = 0.0
        heat = 0.0
        mu_max = 0.0
        for j in range(n):
            if j == i or not gas[j]:
                continue
            d0 = x[i, 0] - x[j, 0]
            d1 = x[i, 1] - x[j, 1]
            d2 = x[i, 2] - x[j, 2]
            r2 = d0 * d0 + d1 * d1 + d2 * d2
            hij = 0.5 * (h[i] + h[j])
            if r2 >= 4.0 * hij * hij or r2 == 0.0:
                continue
            r = np.sqrt(r2)
            vr = (v[i, 0] - v[j, 0]) * d0 + (v[i, 1] - v[j, 1]) * d1 + (v[i, 2] - v[j, 2]) * d2
            visc = 0.0
            if vr < 0.0:
                mu = hij * vr / (r2 + 0.01 * hij * hij)
                cj = np.sqrt(GAMMA * (GAMMA - 1.0) * u[j])
                visc = (-alpha * 0.5 * (ci + cj) * mu + beta * mu * mu) / (0.5 * (rho[i] + rho[j]))
                visc *= 0.5 * (shock[i] + shock[j])
                mu_max = max(mu_max, -mu)
            pj_rho2 = (GAMMA - 1.0) * u[j] / rho[j]
            f = m[j] * (pi_rho2 + pj_rho2 + visc) * _dw(r / hij, hij) / r
            a0 -= f * d0
            a1 -= f * d1
            a2 -= f * d2
            heat += 0.5 * f * vr
        acc[i, 0], acc[i, 1], acc[i, 2] = a0, a1, a2
        dudt[i] = heat
        dt_i[i] = COURANT * h[i] / (ci + 1.2 * (alpha * ci + beta * mu_max) + 1e-300)


# ------------------------------------------------- the same sums, by a tree

LEAF_SIZE = 8  # most particles in a box that is not divided further
_CHUNKS = 48  # the particles are handed to the threads in this many pieces


@numba.njit(cache=True)
def _halve(order, x, axis, a, b, mid):
    """Rearrange order[a:b] so that order[a:mid] are its particles with the smallest x[:, axis].

    Quickselect (Hoare 1961): partition about the middle element's value and
    go on with the side that holds position mid.
    """
    lo, hi = a, b - 1
    while lo < hi:
        pivot = x[order[(lo + hi) // 2], axis]
        i, j = lo, hi
        while i <= j:
            while x[order[i], axis] < pivot:
                i += 1
            while x[order[j], axis] > pivot:
                j -= 1
            if i <= j:
                order[i], order[j] = order[j], order[i]
                i += 1
                j -= 1
        if mid <= j:
            hi = j
        elif mid >= i:
            lo = i
        else:
            return


@numba.njit(cache=True)
def build_tree(x, members):
    """A tree of boxes over the particles `members` (indices into x).

    Box 0 holds all of them; a box of more than LEAF_SIZE is divided at the
    median along its longest side into two boxes of equal numbers (so the
    tree adapts to the density). Returns
        order: the members, rearranged so that every box is a run of it,
        box_lo, box_hi: corners of each box (the extent of its particles),
        first, last: its particles are order[first:last],
        left, right: its two halves (-1: not divided).
    A box's halves come after it in the arrays.
    """
    n = len(members)
    order = members.copy()
    most = 2 * (n // (LEAF_SIZE // 2)) + 2
    box_lo = np.empty((most, 3))
    box_hi = np.empty((most, 3))
    first = np.empty(most, dtype=np.int64)
    last = np.empty(most, dtype=np.int64)
    left = np.full(most, -1, dtype=np.int64)
    right = np.full(most, -1, dtype=np.int64)
    first[0], last[0] = 0, n
    count = 1
    k = 0
    while k < count:
        a, b = first[k], last[k]
        for c in range(3):
            lo = np.inf
            hi = -np.inf
            for q in range(a, b):
                v = x[order[q], c]
                lo = min(lo, v)
                hi = max(hi, v)
            box_lo[k, c] = lo
            box_hi[k, c] = hi
        if b - a > LEAF_SIZE:
            axis = 0
            for c in range(1, 3):
                if box_hi[k, c] - box_lo[k, c] > box_hi[k, axis] - box_lo[k, axis]:
                    axis = c
            mid = (a + b) // 2
            _halve(order, x, axis, a, b, mid)
            left[k], right[k] = count, count + 1
            first[count], last[count] = a, mid
            first[count + 1], last[count + 1] = mid, b
            count += 2
        k += 1
    return (
        order,
        box_lo[:count],
        box_hi[:count],
        first[:count],
        last[:count],
        left[:count],
        right[:count],
    )


@numba.njit(cache=True)
def sphere_boxes(first, last, left, right, x, h):
    """Corners of the box round the spheres |y - x_j| < h_j of each tree box's particles.

    x and h in the tree's order. A particle i can only interact (r < h_i +
    h_j) with a particle of a box whose sphere box is closer to it than h_i.
    """
    count = len(first)
    lo = np.empty((count, 3))
    hi = np.empty((count, 3))
    for k in range(count - 1, -1, -1):
        for c in range(3):
            if left[k] < 0:
                a = np.inf
                b = -np.inf
                for q in range(first[k], last[k]):
                    a = min(a, x[q, c] - h[q])
                    b = max(b, x[q, c] + h[q])
                lo[k, c] = a
                hi[k, c] = b
            else:
                lo[k, c] = min(lo[left[k], c], lo[right[k], c])
                hi[k, c] = max(hi[left[k], c], hi[right[k], c])
    return lo, hi


@numba.njit(cache=True)
def _box_distance2(box_lo, box_hi, k, p0, p1, p2):
    """Squared distance from the point p to box k (0 inside it)."""
    d2 = 0.0
    if p0 < box_lo[k, 0]:
        d2 += (box_lo[k, 0] - p0) ** 2
    elif p0 > box_hi[k, 0]:
        d2 += (p0 - box_hi[k, 0]) ** 2
    if p1 < box_lo[k, 1]:
        d2 += (box_lo[k, 1] - p1) ** 2
    elif p1 > box_hi[k, 1]:
        d2 += (p1 - box_hi[k, 1]) ** 2
    if p2 < box_lo[k, 2]:
        d2 += (box_lo[k, 2] - p2) ** 2
    elif p2 > box_hi[k, 2]:
        d2 += (p2 - box_hi[k, 2]) ** 2
    return d2


@numba.njit(cache=True)
def _within(box_lo, box_hi, first, last, left, right, x, i, radius, stack, found):
    """The particles closer to particle i than `radius`, into `found`; returns how many."""
    p0, p1, p2 = x[i, 0], x[i, 1], x[i, 2]
    r2_max = radius * radius
    count = 0
    stack[0] = 0
    top = 1
    while top > 0:
        top -= 1
        k = stack[top]
        if _box_distance2(box_lo, box_hi, k, p0, p1, p2) >= r2_max:
            continue
        if left[k] >= 0:
            stack[top] = left[k]
            stack[top + 1] = right[k]
            top += 2
            continue
        for j in range(first[k], last[k]):
            d0 = p0 - x[j, 0]
            d1 = p1 - x[j, 1]
            d2 = p2 - x[j, 2]
            if d0 * d0 + d1 * d1 + d2 * d2 < r2_max:
                found[count] = j
                count += 1
    return count


@numba.njit(cache=True)
def _leaves_near(box_lo, box_hi, left, right, q_lo, q_hi, radius, stack, out):
    """The leaves (in the order _within visits them) closer than `radius` to the box q_lo..q_hi.

    Every leaf that _within or forces_tree visits for a particle inside
    that box is among them (a point of the box is no closer to a leaf than
    the box is), so a leaf's particles can share one search: each then
    tests these leaves alone, exactly as it would in its own search.
    """
    r2_max = radius * radius * (1.0 + 1e-9)  # never drop a leaf by rounding
    count = 0
    stack[0] = 0
    top = 1
    while top > 0:
        top -= 1
        k = stack[top]
        d2 = 0.0
        for c in range(3):
            gap = max(q_lo[c] - box_hi[k, c], box_lo[k, c] - q_hi[c], 0.0)
            d2 += gap * gap
        if d2 >= r2_max:
            continue
        if left[k] >= 0:
            stack[top] = left[k]
            stack[top + 1] = right[k]
            top += 2
            continue
        out[count] = k
        count += 1
    return count


@numba.njit(cache=True)
def _within_leaves(box_lo, box_hi, first, last, near, n_near, x, i, radius, found):
    """_within, searching only the leaves near[:n_near] (from _leaves_near): the same particles in the same order."""
    p0, p1, p2 = x[i, 0], x[i, 1], x[i, 2]
    r2_max = radius * radius
    count = 0
    for f in range(n_near):
        k = near[f]
        if _box_distance2(box_lo, box_hi, k, p0, p1, p2) >= r2_max:
            continue
        for j in range(first[k], last[k]):
            d0 = p0 - x[j, 0]
            d1 = p1 - x[j, 1]
            d2 = p2 - x[j, 2]
            if d0 * d0 + d1 * d1 + d2 * d2 < r2_max:
                found[count] = j
                count += 1
    return count


SEARCH_MARGIN = 1.1  # neighbours are looked up this much further out than 2h, see density_tree


@numba.njit(parallel=True, cache=True)
def density_tree(box_lo, box_hi, first, last, left, right, x, v, m, h, u, h_max, passes, rho, shock):  # fmt: skip
    """Smoothing lengths, densities and Balsara factors of gas particles in the order of their tree.

    What Gas did with `density` in passes + 1 sweeps over all particles:
    `passes` times rho_i from h_i, then h_i = ETA (m_i / rho_i)^(1/3) (changed
    by a factor 2 at most, and not above h_max); then rho_i and the Balsara
    factor with the final h_i. All of that depends on h_i alone and not on the other
    particles' h, so it is done here particle by particle, from one list of
    its neighbours: they are looked up out to SEARCH_MARGIN 2 h_i, and again
    only if h_i grows past that. h is updated in place.

    The kernel's factors 1 / (pi h_i^3) and 1 / (pi h_i^4), the same for all
    neighbours of i, multiply the finished sums.

    Speed only: the particles of one leaf share one walk down the tree
    (_leaves_near, out to the largest first search radius in the leaf); a
    search that reaches further (h grew) walks the tree on its own. Either
    way a particle finds the same neighbours in the same order.
    """
    n = len(x)
    leaves = np.flatnonzero(left < 0)
    n_leaves = len(leaves)
    for chunk in numba.prange(_CHUNKS):
        stack = np.empty(128, dtype=np.int64)
        found = np.empty(n, dtype=np.int64)
        near = np.empty(n_leaves, dtype=np.int64)
        for lf in range(chunk * n_leaves // _CHUNKS, (chunk + 1) * n_leaves // _CHUNKS):
            leaf = leaves[lf]
            reach = 0.0
            for i in range(first[leaf], last[leaf]):
                reach = max(reach, SEARCH_MARGIN * 2.0 * h[i])
            n_near = _leaves_near(
                box_lo, box_hi, left, right, box_lo[leaf], box_hi[leaf], reach, stack, near
            )
            for i in range(first[leaf], last[leaf]):
                p0, p1, p2 = x[i, 0], x[i, 1], x[i, 2]
                hi = h[i]
                searched = 0.0
                count = 0
                for _ in range(passes):
                    if 2.0 * hi > searched:
                        searched = SEARCH_MARGIN * 2.0 * hi
                        if searched <= reach:
                            count = _within_leaves(
                                box_lo, box_hi, first, last, near, n_near, x, i, searched, found
                            )
                        else:
                            count = _within(
                                box_lo, box_hi, first, last, left, right, x, i, searched, stack, found
                            )  # fmt: skip
                    total = 0.0
                    for f in range(count):
                        j = found[f]
                        d0 = p0 - x[j, 0]
                        d1 = p1 - x[j, 1]
                        d2 = p2 - x[j, 2]
                        q = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) / hi
                        if q < 1.0:
                            total += m[j] * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
                        elif q < 2.0:
                            total += m[j] * 0.25 * (2.0 - q) ** 3
                    total /= np.pi * hi * hi * hi
                    if total > 0.0:
                        want = ETA * np.cbrt(m[i] / total)
                        hi = min(max(want, 0.5 * hi), min(2.0 * hi, h_max))
                if 2.0 * hi > searched:
                    if 2.0 * hi <= reach:
                        count = _within_leaves(
                            box_lo, box_hi, first, last, near, n_near, x, i, 2.0 * hi, found
                        )
                    else:
                        count = _within(
                            box_lo, box_hi, first, last, left, right, x, i, 2.0 * hi, stack, found
                        )  # fmt: skip
                h[i] = hi
                per_h = 1.0 / hi
                r2_max = 4.0 * hi * hi
                total = 0.0
                div = 0.0
                c0 = c1 = c2 = 0.0
                for f in range(count):
                    j = found[f]
                    d0 = p0 - x[j, 0]
                    d1 = p1 - x[j, 1]
                    d2 = p2 - x[j, 2]
                    r2 = d0 * d0 + d1 * d1 + d2 * d2
                    if r2 >= r2_max:
                        continue
                    r = np.sqrt(r2)
                    q = r * per_h
                    if q < 1.0:
                        total += m[j] * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
                        slope = -3.0 * q + 2.25 * q * q
                    else:
                        total += m[j] * 0.25 * (2.0 - q) ** 3
                        slope = -0.75 * (2.0 - q) ** 2
                    if r > 0.0:
                        g = m[j] * slope / r  # m_j grad_i W = g * d / (pi h^4)
                        v0 = v[i, 0] - v[j, 0]
                        v1 = v[i, 1] - v[j, 1]
                        v2 = v[i, 2] - v[j, 2]
                        div -= g * (v0 * d0 + v1 * d1 + v2 * d2)
                        c0 += g * (v1 * d2 - v2 * d1)
                        c1 += g * (v2 * d0 - v0 * d2)
                        c2 += g * (v0 * d1 - v1 * d0)
                w_norm = per_h * per_h * per_h / np.pi
                total *= w_norm
                rho[i] = total
                div = abs(div) * (w_norm * per_h) / total
                curl = np.sqrt(c0 * c0 + c1 * c1 + c2 * c2) * (w_norm * per_h) / total
                c = np.sqrt(GAMMA * (GAMMA - 1.0) * u[i])
                shock[i] = div / (div + curl + 1e-4 * c / hi + 1e-300)


@numba.njit(parallel=True, cache=True)
def forces_tree(
    reach_lo, reach_hi, first, last, left, right, x, v, m, h, rho, u, shock, alpha, beta,
    acc, dudt, dt_i,
):  # fmt: skip
    """`forces` for gas particles given in the order of their tree; reach_lo, reach_hi from sphere_boxes."""
    n = len(x)
    p_rho2 = np.empty(n)  # P / rho^2
    sound = np.empty(n)
    for i in range(n):
        p_rho2[i] = (GAMMA - 1.0) * u[i] / rho[i]
        sound[i] = np.sqrt(GAMMA * (GAMMA - 1.0) * u[i])
    # Speed only: the particles of one leaf share one walk down the tree
    # (_leaves_near, out to the largest h in the leaf), then each tests the
    # leaves found exactly as its own walk would: the same pairs, in the same order.
    leaves = np.flatnonzero(left < 0)
    n_leaves = len(leaves)
    for chunk in numba.prange(_CHUNKS):
        stack = np.empty(128, dtype=np.int64)
        near = np.empty(n_leaves, dtype=np.int64)
        q_lo = np.empty(3)
        q_hi = np.empty(3)
        for lf in range(chunk * n_leaves // _CHUNKS, (chunk + 1) * n_leaves // _CHUNKS):
            leaf = leaves[lf]
            reach = 0.0
            for c in range(3):
                q_lo[c] = np.inf
                q_hi[c] = -np.inf
            for i in range(first[leaf], last[leaf]):
                reach = max(reach, h[i])
                for c in range(3):
                    q_lo[c] = min(q_lo[c], x[i, c])
                    q_hi[c] = max(q_hi[c], x[i, c])
            n_near = _leaves_near(reach_lo, reach_hi, left, right, q_lo, q_hi, reach, stack, near)
            for i in range(first[leaf], last[leaf]):
                _forces_one(
                    reach_lo, reach_hi, first, last, near, n_near, x, v, m, h, rho, shock,
                    p_rho2, sound, alpha, beta, i, acc, dudt, dt_i,
                )  # fmt: skip


@numba.njit(cache=True)
def _forces_one(
    reach_lo, reach_hi, first, last, near, n_near, x, v, m, h, rho, shock, p_rho2, sound,
    alpha, beta, i, acc, dudt, dt_i,
):  # fmt: skip
    """forces_tree for particle i, from the leaves near[:n_near] that hold its neighbours."""
    pi_rho2 = p_rho2[i]
    ci = sound[i]
    a0 = a1 = a2 = 0.0
    heat = 0.0
    mu_max = 0.0
    hi = h[i]
    p0, p1, p2 = x[i, 0], x[i, 1], x[i, 2]
    for nk in range(n_near):
        k = near[nk]
        if _box_distance2(reach_lo, reach_hi, k, p0, p1, p2) >= hi * hi:
            continue
        for j in range(first[k], last[k]):
            if j == i:
                continue
            d0 = p0 - x[j, 0]
            d1 = p1 - x[j, 1]
            d2 = p2 - x[j, 2]
            r2 = d0 * d0 + d1 * d1 + d2 * d2
            hij = 0.5 * (hi + h[j])
            if r2 >= 4.0 * hij * hij or r2 == 0.0:
                continue
            r = np.sqrt(r2)
            vr = (
                (v[i, 0] - v[j, 0]) * d0
                + (v[i, 1] - v[j, 1]) * d1
                + (v[i, 2] - v[j, 2]) * d2
            )
            visc = 0.0
            if vr < 0.0:
                mu = hij * vr / (r2 + 0.01 * hij * hij)
                visc = (-alpha * 0.5 * (ci + sound[j]) * mu + beta * mu * mu) / (
                    0.5 * (rho[i] + rho[j])
                )
                visc *= 0.5 * (shock[i] + shock[j])
                mu_max = max(mu_max, -mu)
            f = m[j] * (pi_rho2 + p_rho2[j] + visc) * _dw(r / hij, hij) / r
            a0 -= f * d0
            a1 -= f * d1
            a2 -= f * d2
            heat += 0.5 * f * vr
    acc[i, 0], acc[i, 1], acc[i, 2] = a0, a1, a2
    dudt[i] = heat
    dt_i[i] = COURANT * hi / (ci + 1.2 * (alpha * ci + beta * mu_max) + 1e-300)


class Gas:
    """SPH state of the bodies of a BodySystem or KerrBodySystem (one entry per body).

    A body is gas once it has mass (a torn star's particles, see
    blackhole_star) and until it falls into the hole. The body system adds
    accelerations() to the mutual gravity in its kicks, and calls
    heat() with the proper time of each kick.
    """

    def __init__(self, n):
        self.u = np.zeros(n)  # thermal energy per unit mass [c^2]
        self.h = np.full(n, 0.1)
        self.rho = np.zeros(n)
        self.shock = np.ones(n)  # Balsara factors
        self.dudt = np.zeros(n)
        self.dt_i = np.full(n, np.inf)
        self.acc = np.zeros((n, 3))
        self.time_step = np.inf  # shortest Courant time at the last evaluation [M]
        self.cooling = 0.0  # cooling time in units of 1 / Omega (0: none; see BETA_COOL)
        self.shear_viscosity = False  # let the artificial viscosity act on shear (discs)
        self.viscosity = 1.0  # multiplies ALPHA and BETA
        self.disc_aspect = 0.0  # H / r of a locally isothermal disc (0: off)
        self.omega = np.zeros(n)  # orbital frequency r^-1.5 at the last evaluation
        self.started = False
        self.u_start = 0.0
        self.h_max = H_MAX  # [M] largest smoothing length (thin gas has fewer neighbours)

    def resize(self, n):
        if n == len(self.u):
            return
        old = len(self.u)

        def grow(a, fill):
            out = np.full((n, *a.shape[1:]), fill, dtype=a.dtype)
            out[: min(old, n)] = a[: min(old, n)]
            return out

        self.u, self.h, self.rho = grow(self.u, 0.0), grow(self.h, 0.1), grow(self.rho, 0.0)
        self.shock = grow(self.shock, 1.0)
        self.omega = grow(self.omega, 0.0)
        self.dudt, self.dt_i = grow(self.dudt, 0.0), grow(self.dt_i, np.inf)
        self.acc = grow(self.acc, 0.0)

    def start(self, idx, u, spacing):
        """Particles idx become gas with thermal energies u; `spacing` is their typical separation."""
        self.u[idx] = u
        self.u_start = float(np.median(u))  # for the display colour of heated gas
        self.h[idx] = ETA * spacing
        self.started = False

    def accelerations(self, x, v, mass, alive):
        """Pressure and viscosity accelerations of all bodies (zero for those that are not gas)."""
        self.resize(len(x))
        gas = alive & (mass > 0.0)
        members = np.flatnonzero(gas)
        if len(members) < 2:
            self.acc[:] = 0.0
            self.dudt[:] = 0.0
            self.time_step = np.inf
            return self.acc
        x = np.ascontiguousarray(x)
        v = np.ascontiguousarray(v)
        # The gas in the order of its tree of boxes (see the module text):
        # neighbours in space are then neighbours in memory too.
        order, *boxes = build_tree(x, members)
        xs, vs, ms = x[order], v[order], mass[order]
        hs, us = self.h[order], self.u[order]
        rho, shock = np.empty(len(order)), np.empty(len(order))
        # h follows the density from one evaluation to the next; the first
        # time it is iterated from the starting guess.
        passes = 1 if self.started else 6
        density_tree(*boxes, xs, vs, ms, hs, us, float(self.h_max), passes, rho, shock)
        self.started = True
        self.omega = np.sum(x * x, axis=1) ** -0.75
        acc, dudt, dt_i = np.empty((len(order), 3)), np.empty(len(order)), np.empty(len(order))
        links = boxes[2:]  # first, last, left, right
        forces_tree(
            *sphere_boxes(*links, xs, hs),
            *links,
            xs,
            vs,
            ms,
            hs,
            rho,
            us,
            np.ones_like(shock) if self.shear_viscosity else shock,
            ALPHA * self.viscosity,
            BETA * self.viscosity,
            acc,
            dudt,
            dt_i,
        )
        # Back to the bodies' own order; what is not gas has no density or force.
        self.h[order] = hs
        self.rho[:] = 0.0
        self.rho[order] = rho
        self.shock[:] = 1.0
        self.shock[order] = shock
        self.acc[:] = 0.0
        self.acc[order] = acc
        self.dudt[:] = 0.0
        self.dudt[order] = dudt
        self.dt_i[:] = np.inf
        self.dt_i[order] = dt_i
        self.time_step = float(np.min(dt_i))
        return self.acc

    def heat(self, dtau, which):
        """Advance the thermal energies of bodies `which` over their proper times dtau."""
        self.u[which] = np.maximum(self.u[which] + self.dudt[which] * dtau, 0.0)
        if self.disc_aspect > 0.0:
            # omega = r^-1.5, so omega^(2/3) = 1 / r.
            self.u[which] = (
                self.disc_aspect**2 * self.omega[which] ** (2.0 / 3.0) / (GAMMA * (GAMMA - 1.0))
            )
        elif self.cooling > 0.0:
            self.u[which] *= np.exp(-dtau * self.omega[which] / self.cooling)
