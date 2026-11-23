"""Tight bound pairs that cannot be seen: their orbit is advanced analytically.

Written by Claude under an explicit override of CLAUDE.md (2026-11-23).
Review the method and the parameters.

THE PROBLEM. A compact body or a second black hole a few G M / c^2 from a
10 solar-mass hole goes round it millions of times per physics step, on an
orbit far below a pixel. Followed with the leapfrog (nbodystate.Integrator)
it needs ~10^9 substeps per step; the step budget cut every step short and
the rest of the scene stood almost still.

THE METHOD: a splitting of the Wisdom-Holman kind (Wisdom & Holman 1991, AJ
102, 1528), with the pair's own motion averaged over its orbit.

- In the leapfrog the pair is one body: its centre of mass, with its total
  mass (held_force). Both members get the acceleration of that body, so
  they move rigidly with it during the step, and everything else is pulled
  by the total mass at the centre of mass. The pair does not set the
  substep (held_timescale) and its members do not touch each other
  (nbodystate.first_contact, `partner`).
- After the step the pair's relative orbit is advanced by the time the step
  took, in closed form (advance_pairs):

  * Its two first integrals at 1PN order, the energy and angular momentum
    of the relative motion per reduced mass (e.g. Blanchet 2014, Living
    Rev. Relativ. 17, 2, sec. 7.1; x = x_1 - x_2, v = v_1 - v_2,
    M = m_1 + m_2, nu = m_1 m_2 / M^2),
        eps = v^2 / 2 - G M / r + 1/c^2 [3/8 (1 - 3 nu) v^4
              + G M / (2 r) ((3 + nu) v^2 + nu rdot^2) + (G M / r)^2 / 2]
        h   = |x cross v| [1 + 1/c^2 ((1 - 3 nu) v^2 / 2 + (3 + nu) G M / r)],
    are read off the state. eps times the reduced mass is exactly
    postnewtonian.eih_energy of the two bodies in their centre-of-mass
    frame. They fix the orbit's radial range [r_min, r_max] (found
    numerically as the zeros of rdot^2 (r), see _range), hence
    a_r = (r_max + r_min) / 2 and e_r = (r_max - r_min) / (r_max + r_min).
  * The radial motion is the quasi-Keplerian one of Damour & Deruelle
    (1985, Ann. Inst. H. Poincare 43, 107): r = a_r (1 - e_r cos u),
    u - e_r sin u = n (t - t_0), with the 1PN mean motion
        n = (-2 eps)^(3/2) / (G M) [1 - (15 - nu) (-2 eps) / (8 c^2)].
  * The direction of x turns by (1 + k) times the true anomaly that goes
    with u: the periapsis advances by 2 pi k per radial period,
        k = 3 (G M)^2 / (c^2 h^2)  =  3 G M / (c^2 a (1 - e^2)),
    which is the 1PN precession of 6 pi G M / (c^2 a (1 - e^2)) per orbit.
  * Gravitational waves: the orbit averages of the 2.5PN force (Peters
    1964, Phys. Rev. 136, B1224, eqs. 5.6 and 5.7)
        da/dt = -64/5 G^3 m_1 m_2 M / (c^5 a^3 (1 - e^2)^(7/2))
                (1 + 73/24 e^2 + 37/96 e^4)
        de/dt = -304/15 e G^3 m_1 m_2 M / (c^5 a^4 (1 - e^2)^(5/2))
                (1 + 121/304 e^2)
    are integrated over the step (RK4, steps of at most 2% of a) for
    a = a_r and e = e_r, the orbit's actual size and shape, and the new
    eps and h are those of the orbit with the new radial range
    (_from_range). (With a = -G M / (2 eps) and e^2 = 1 + 2 eps h^2 /
    (G M)^2 instead, which differ from a_r and e_r at 1PN order, the
    energy loss came out at 0.38 to 0.96 of that of the direct integration
    with the 2.5PN force for G M / (a c^2) = 0.025 to 0.0015; with a_r and
    e_r at 0.81 to 0.997.) A change of less than DECAY_FLOOR per step is
    not applied: it is below the rounding of that conversion, and such a
    pair takes more than 4e8 years to merge.
  * The new speeds follow from the new r, eps and h (the two equations
    above solved for rdot^2 and the transverse speed, _speeds), so the
    state written back has exactly those first integrals.

  With Newtonian gravity (1 / c^2 = 0) this is the exact Kepler orbit.

WHAT IS KEPT. The pair's centre of mass and total momentum (exactly); its
1PN energy and angular momentum (to rounding, apart from what the waves
take); the precession and the decay, so a tight black-hole binary still
spirals in and merges when Peters says. The state is the only memory: the
orbit is read from the positions and velocities at the start of every step
and written back at its end. Taking a pair and giving it back therefore do
not change the state at all, and no energy is gained or lost by the switch.

ERROR TERMS.
- The pull of the other bodies on the pair's internal orbit (the tide) and
  the pair's quadrupole pull on them are left out. Both are bounded by the
  conditions below (TIDE, QUADRUPOLE). Measured: stars of 1 and 0.1 solar
  masses 0.02 AU apart (nu (r_max / d)^2 = 5.6e-5) move a planet at 1 AU by
  4.8e-4 AU in a third of its orbit compared with the direct integration,
  twice the leapfrog's own error for that planet; the limit is set at half
  of that. The orbit-averaged quadrupole is not applied.
- Periodic 1PN terms in the angle (the difference between the three
  eccentricities of the quasi-Keplerian orbit): the position along the
  orbit is off by ~G M / (c^2 a) radians, without growing. The orbit is
  below what can be seen by construction.
- Decay: Peters' rates are the leading order; their relative error is a
  few G M / (c^2 a) (the 1PN corrections to the flux, which the direct
  integration does not have right either; measured against it above), and
  so is that of the merger time, most of which is spent at the largest a.
- The members' velocity through the scene, V, is taken as Galilean: terms
  of order V v / c^2 of the pair's binding energy (v the orbital speed)
  come and go with the orbital phase in the 1PN energy of the whole system.
- The pair weighs m_1 + m_2 for the others; its binding energy (a 1PN
  three-body term, ~nu G M / (2 a c^2) of the mass) is left out.

WHICH PAIRS (find_pairs). Two bodies that are not debris, when all of:
- they are bound (eps < 0) and stay apart: r_min is outside the distance at
  which they would merge, collide or the one would be torn (nbodystate.
  pair_reach with every body counted as heavy; with relativity also outside
  6 G M / c^2, where the expansion ends);
- the orbit cannot be seen: r_max is below one pixel, or, if one of the
  two is a black hole, below the smallest radius a hole is drawn with (the
  companion is under the hole's dot); AND its period is below
  UNRESOLVED_PERIOD physics steps (it cannot be drawn as a curve). A
  resolved orbit is never taken;
- nothing else is close: for every other body k at distance d from the
  pair's centre of mass (less what it can travel in one step),
      d > CLEARANCE r_max,
      nu (r_max / d)^2 < QUADRUPOLE   (the pair as a point mass),
      2 m_k / M (r_max / d)^3 < the tide limit: TIDE, and where relativity
      turns the orbit faster than the tide can (Kozai-Lidov cycles are
      then quenched, e.g. Naoz 2016, ARA&A 54, 441, sec. 5) the smaller of
      TIDE_FAST and QUENCH k; otherwise the tide would build up over the
      many orbits within one step.
The conditions depend on the state only, not on its history or on the
direction of time, so a pair is taken and given back at the same distance
of a perturber, coming and going. A body belongs to one pair at most (the
tightest); a pair is not itself a member of a wider unresolved pair.

WHEN THE PAIR MEETS. If the decay brings r_min to the reach above, the pair
is put at that separation (at its closest approach) and reported (advance_pairs): the
module then applies its usual rules (merger with its mass loss, capture,
tear, collision).
"""

import numba
import numpy as np

from physengine.modules.nbodystate import NBodyState, pair_reach

# The orbit cannot be drawn as a curve if it has fewer than this many physics
# steps per revolution (two frames at four steps per frame).
UNRESOLVED_PERIOD = 8.0
CLEARANCE = 3.0
QUADRUPOLE = 3e-5
TIDE = 1e-6
TIDE_FAST = 1e-4
QUENCH = 1e-3
VALID_1PN = 6.0  # G M / c^2: no pair is followed closer than this
DECAY_FLOOR = 1e-11  # smallest relative change of the orbit per step that is applied
# A decaying orbit is kept at least this eccentric (r_max / r_min - 1 of twice
# this): the conversion from its radial range needs the two ends apart.
MIN_ECCENTRICITY = 5e-5


@numba.njit(cache=True)
def invariants(x, v, gm, nu, ic2):
    """(eps, h, r, rdot) of a relative orbit; see the module docstring."""
    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    rd = (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]) / r
    w = gm / r
    eps = (
        0.5 * v2
        - w
        + ic2
        * (
            0.375 * (1.0 - 3.0 * nu) * v2 * v2
            + 0.5 * w * ((3.0 + nu) * v2 + nu * rd * rd)
            + 0.5 * w * w
        )
    )
    cx = x[1] * v[2] - x[2] * v[1]
    cy = x[2] * v[0] - x[0] * v[2]
    cz = x[0] * v[1] - x[1] * v[0]
    h = np.sqrt(cx * cx + cy * cy + cz * cz) * (
        1.0 + ic2 * (0.5 * (1.0 - 3.0 * nu) * v2 + (3.0 + nu) * w)
    )
    return eps, h, r, rd


@numba.njit(cache=True)
def _speeds(r, eps, h, gm, nu, ic2):
    """(rdot^2, transverse speed) at separation r of the orbit with first integrals
    eps and h. rdot^2 is negative outside the orbit's radial range."""
    w = gm / r
    v2 = 2.0 * (eps + w)
    if ic2 == 0.0:
        vt = h / r
        return v2 - vt * vt, vt
    a1, a3 = 1.0 - 3.0 * nu, 3.0 + nu
    vt = h / r
    for _ in range(200):
        vt = h / (r * (1.0 + ic2 * (0.5 * a1 * v2 + a3 * w)))
        rd2 = v2 - vt * vt
        pn = ic2 * (0.375 * a1 * v2 * v2 + 0.5 * w * (a3 * v2 + nu * rd2) + 0.5 * w * w)
        res = 2.0 * (eps + w - pn) - v2
        # Newton step, with the derivative taken at fixed transverse speed.
        step = res / (1.0 + ic2 * (1.5 * a1 * v2 + w * (a3 + nu)))
        v2 += step
        if abs(step) <= 1e-15 * abs(v2):
            break
    vt = h / (r * (1.0 + ic2 * (0.5 * a1 * v2 + a3 * w)))
    return v2 - vt * vt, vt


@numba.njit(cache=True)
def _range(eps, h, gm, nu, ic2, r_floor):
    """Radial range of the orbit: (code, r_min, r_max). code 0: not a bound orbit;
    1: bound, between r_min and r_max; 2: it comes within r_floor."""
    if not (eps < 0.0) or not (h > 0.0):
        return 0, 0.0, 0.0
    p = h * h / gm
    if r_floor > 0.0:
        if r_floor >= 0.9 * p:
            return 2, r_floor, r_floor
        g, _ = _speeds(r_floor, eps, h, gm, nu, ic2)
        if g >= 0.0:
            return 2, r_floor, r_floor
    # rdot^2 (r) has one maximum, at h^2 / (G M) in Newtonian gravity.
    lo, hi = max(r_floor, 0.3 * p), 3.0 * p
    golden = 0.6180339887498949
    c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
    gc, _ = _speeds(c, eps, h, gm, nu, ic2)
    gd, _ = _speeds(d, eps, h, gm, nu, ic2)
    a, b = lo, hi
    for _ in range(60):
        if gc > gd:
            b, d, gd = d, c, gc
            c = b - golden * (b - a)
            gc, _ = _speeds(c, eps, h, gm, nu, ic2)
        else:
            a, c, gc = c, d, gd
            d = a + golden * (b - a)
            gd, _ = _speeds(d, eps, h, gm, nu, ic2)
    top = 0.5 * (a + b)
    g_top, _ = _speeds(top, eps, h, gm, nu, ic2)
    if g_top <= 0.0:
        return 1, top, top  # circular to rounding
    g, _ = _speeds(lo, eps, h, gm, nu, ic2)
    if g >= 0.0:
        return 0, 0.0, 0.0
    a, b = lo, top
    for _ in range(64):
        mid = 0.5 * (a + b)
        g, _ = _speeds(mid, eps, h, gm, nu, ic2)
        if g < 0.0:
            a = mid
        else:
            b = mid
    r_min = b
    a, b = top, 2.0 * top
    for _ in range(400):
        g, _ = _speeds(b, eps, h, gm, nu, ic2)
        if g < 0.0:
            break
        a, b = b, 2.0 * b
    else:
        return 0, 0.0, 0.0
    for _ in range(64):
        mid = 0.5 * (a + b)
        g, _ = _speeds(mid, eps, h, gm, nu, ic2)
        if g < 0.0:
            b = mid
        else:
            a = mid
    return 1, r_min, a


@numba.njit(cache=True)
def _from_range(r_min, r_max, gm, nu, ic2):
    """(ok, eps, h) of the bound orbit that turns round at r_min and r_max: the
    transverse speed at r_min for which rdot^2 (r_max) = 0, by bisection."""
    x = np.array([r_min, 0.0, 0.0])
    lo, hi = 0.5 * np.sqrt(gm / r_min), 2.0 * np.sqrt(gm / r_min)  # Newtonian: 1 to 1.41
    for k in range(2):
        speed = lo if k == 0 else hi
        eps, h, _, _ = invariants(x, np.array([0.0, speed, 0.0]), gm, nu, ic2)
        g, _ = _speeds(r_max, eps, h, gm, nu, ic2)
        if (g < 0.0) != (k == 0):
            return False, 0.0, 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        eps, h, _, _ = invariants(x, np.array([0.0, mid, 0.0]), gm, nu, ic2)
        g, _ = _speeds(r_max, eps, h, gm, nu, ic2)
        if g < 0.0:
            lo = mid
        else:
            hi = mid
    eps, h, _, _ = invariants(x, np.array([0.0, 0.5 * (lo + hi), 0.0]), gm, nu, ic2)
    return eps < 0.0, eps, h


@numba.njit(cache=True)
def mean_motion(eps, gm, nu, ic2):
    """2 pi / radial period of the bound orbit of energy eps (1PN)."""
    return (-2.0 * eps) ** 1.5 / gm * (1.0 - (15.0 - nu) * (-2.0 * eps) * ic2 / 8.0)


@numba.njit(cache=True)
def _peters_rates(a, e, kgw):
    q = 1.0 - e * e
    da = -12.8 * kgw / (a**3 * q**3.5) * (1.0 + (73.0 / 24.0) * e * e + (37.0 / 96.0) * e**4)
    de = -(304.0 / 15.0) * e * kgw / (a**4 * q**2.5) * (1.0 + (121.0 / 304.0) * e * e)
    return da, de


@numba.njit(cache=True)
def peters(a, e, dt, kgw, r_stop):
    """Semi-major axis and eccentricity after dt of gravitational-wave decay (Peters
    1964), kgw = G^3 m_1 m_2 M / c^5. Stops when the periapsis a (1 - e) reaches
    r_stop. Returns (a, e, time used, whether it stopped there)."""
    t = 0.0
    while t < dt:
        if a * (1.0 - e) <= r_stop:
            return a, e, t, True
        da1, de1 = _peters_rates(a, e, kgw)
        step = dt - t
        if da1 != 0.0:
            step = min(step, 0.02 * a / abs(da1))
        a2, e2 = a + 0.5 * step * da1, max(e + 0.5 * step * de1, 0.0)
        da2, de2 = _peters_rates(a2, e2, kgw)
        a3, e3 = a + 0.5 * step * da2, max(e + 0.5 * step * de2, 0.0)
        da3, de3 = _peters_rates(a3, e3, kgw)
        a4, e4 = a + step * da3, max(e + step * de3, 0.0)
        da4, de4 = _peters_rates(a4, e4, kgw)
        a += step * (da1 + 2.0 * da2 + 2.0 * da3 + da4) / 6.0
        e = max(e + step * (de1 + 2.0 * de2 + 2.0 * de3 + de4) / 6.0, 0.0)
        t += step
    return a, e, dt, a * (1.0 - e) <= r_stop


@numba.njit(cache=True)
def _kepler(mean, e):
    """Eccentric anomaly u in [0, 2 pi) with u - e sin u = mean (mod 2 pi)."""
    two_pi = 2.0 * np.pi
    mean = mean - two_pi * np.floor(mean / two_pi)
    u = np.pi  # Newton from pi converges for every e < 1
    for _ in range(100):
        step = (u - e * np.sin(u) - mean) / (1.0 - e * np.cos(u))
        u -= step
        if abs(step) < 1e-15:
            break
    return u


@numba.njit(cache=True)
def _true_minus_eccentric(u, e):
    """True anomaly minus eccentric anomaly (periodic in u)."""
    beta = e / (1.0 + np.sqrt(1.0 - e * e))
    return 2.0 * np.arctan2(beta * np.sin(u), 1.0 - beta * np.cos(u))


@numba.njit(cache=True)
def advance_relative(x, v, m1, m2, G, ic2, dt, r_floor, radiate):
    """The relative position and velocity x = x_1 - x_2, v = v_1 - v_2 of a bound
    pair after dt (module docstring). Returns (x, v, code): code 1: advanced;
    2: the pair has come within r_floor (it is put at its closest approach,
    or left as it is if it was already that close); 0: not a bound orbit,
    nothing changed."""
    total = m1 + m2
    gm = G * total
    nu = m1 * m2 / (total * total)
    eps, h, r, rd = invariants(x, v, gm, nu, ic2)
    code, r_min, r_max = _range(eps, h, gm, nu, ic2, r_floor)
    if code != 1:
        return x.copy(), v.copy(), code
    a_r, e_r = 0.5 * (r_max + r_min), (r_max - r_min) / (r_max + r_min)
    u0 = 0.0
    if e_r > 1e-13:
        cu = min(max((1.0 - r / a_r) / e_r, -1.0), 1.0)
        su = np.sqrt(1.0 - cu * cu)
        u0 = np.arctan2(-su if rd < 0.0 else su, cu)
    n0 = mean_motion(eps, gm, nu, ic2)
    k0 = 3.0 * gm * gm * ic2 / (h * h)

    eps1, h1, used, met = eps, h, dt, False
    a_r1, e_r1 = a_r, e_r
    if radiate and ic2 > 0.0 and m1 > 0.0 and m2 > 0.0 and dt > 0.0:
        a, e, t, stopped = peters(a_r, e_r, dt, G**3 * m1 * m2 * total * ic2**2.5, r_floor)
        if abs(a - a_r) > DECAY_FLOOR * a_r or abs(e - e_r) > DECAY_FLOOR:
            e = max(e, MIN_ECCENTRICITY)
            ok, eps_new, h_new = _from_range(a * (1.0 - e), a * (1.0 + e), gm, nu, ic2)
            if ok:
                eps1, h1, a_r1, e_r1, used, met = eps_new, h_new, a, e, t, stopped
    n1 = mean_motion(eps1, gm, nu, ic2)
    k1 = 3.0 * gm * gm * ic2 / (h1 * h1)

    turn = 0.5 * (n0 + n1) * used  # mean anomaly gained
    u1 = _kepler(u0 - e_r * np.sin(u0) + turn, e_r1)
    # True anomaly gained, whole revolutions included, times (1 + k).
    angle = turn + e_r1 * np.sin(u1) - e_r * np.sin(u0)
    angle += _true_minus_eccentric(u1, e_r1) - _true_minus_eccentric(u0, e_r)
    angle *= 1.0 + 0.5 * (k0 + k1)
    if met:
        r1 = a_r1 * (1.0 - e_r1)  # at its closest: the reach, or just inside
        rd2, vt = _speeds(r1, eps1, h1, gm, nu, ic2)
        rd1 = 0.0
    else:
        r1 = a_r1 * (1.0 - e_r1 * np.cos(u1))
        rd2, vt = _speeds(r1, eps1, h1, gm, nu, ic2)
        rd1 = np.sqrt(max(rd2, 0.0))
        if np.sin(u1) < 0.0:
            rd1 = -rd1

    # In the orbital plane: e1 along x, e2 a quarter turn ahead.
    e1 = x / r
    cx = x[1] * v[2] - x[2] * v[1]
    cy = x[2] * v[0] - x[0] * v[2]
    cz = x[0] * v[1] - x[1] * v[0]
    norm = np.sqrt(cx * cx + cy * cy + cz * cz)
    normal = np.array([cx / norm, cy / norm, cz / norm])
    e2 = np.array(
        [
            normal[1] * e1[2] - normal[2] * e1[1],
            normal[2] * e1[0] - normal[0] * e1[2],
            normal[0] * e1[1] - normal[1] * e1[0],
        ]
    )
    cos, sin = np.cos(angle), np.sin(angle)
    along = cos * e1 + sin * e2
    across = cos * e2 - sin * e1
    return r1 * along, rd1 * along + vt * across, 2 if met else 1


@numba.njit(cache=True)
def _floor(i, j, radius, mass, black_hole, debris, G, ic2, capture):
    """(closest the pair may come while it is followed here, whether it has a reach at all)."""
    reach = pair_reach(i, j, radius, mass, black_hole, debris, capture, 0.0, -1.0)
    return max(reach, VALID_1PN * G * (mass[i] + mass[j]) * ic2), reach > 0.0


@numba.njit(cache=True)
def find_pairs(
    pos,
    vel,
    mass,
    radius,
    black_hole,
    debris,
    G,
    ic2,
    max_period,
    max_size,
    max_size_hole,
    step,
    capture,
):
    """The pairs to follow analytically (module docstring, WHICH PAIRS), as rows (i, j)
    with i < j. `max_size`: the largest r_max, `max_size_hole` for a pair
    with a black hole in it; `step`: the time one physics step may advance."""
    n = len(mass)
    found = np.empty((n, 2), dtype=np.int64)
    count = 0
    taken = np.zeros(n, dtype=np.bool_)
    if not (max_size_hole > 0.0) or n < 2:
        return found[:0]
    two_pi = 2.0 * np.pi
    room = 4 * n  # candidates kept (more than a few per body cannot all be isolated)
    cand = np.empty((room, 2), dtype=np.int64)
    cand_p = np.empty(room)
    n_cand = 0
    for i in range(n):
        if debris[i]:
            continue
        for j in range(i + 1, n):
            if debris[j]:
                continue
            total = mass[i] + mass[j]
            if not (total > 0.0):
                continue
            gm = G * total
            x = pos[i] - pos[j]
            v = vel[i] - vel[j]
            # Newtonian first: most pairs are unbound, slow or wide.
            r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            size = max_size_hole if black_hole[i] or black_hole[j] else max_size
            if not (r > 0.0) or r >= size:
                continue
            energy = 0.5 * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) - gm / r
            if energy >= 0.0:
                continue
            a = -gm / (2.0 * energy)
            if a >= size or two_pi * np.sqrt(a**3 / gm) >= 2.0 * max_period:
                continue
            nu = mass[i] * mass[j] / (total * total)
            eps, h, r, rd = invariants(x, v, gm, nu, ic2)
            r_floor, _ = _floor(i, j, radius, mass, black_hole, debris, G, ic2, capture)
            code, r_min, r_max = _range(eps, h, gm, nu, ic2, r_floor)
            if code != 1 or r_max >= size:
                continue
            motion = mean_motion(eps, gm, nu, ic2)
            if not (motion > 0.0) or two_pi / motion >= max_period:
                continue
            # Nothing else close.
            com = (mass[i] * pos[i] + mass[j] * pos[j]) / total
            com_v = (mass[i] * vel[i] + mass[j] * vel[j]) / total
            tide = TIDE
            if ic2 > 0.0:
                tide = max(tide, min(TIDE_FAST, QUENCH * 3.0 * gm * gm * ic2 / (h * h)))
            alone = True
            for k in range(n):
                if k == i or k == j:
                    continue
                d = 0.0
                speed = 0.0
                for c in range(3):
                    d += (pos[k, c] - com[c]) ** 2
                    speed += (vel[k, c] - com_v[c]) ** 2
                d = np.sqrt(d) - np.sqrt(speed) * step
                if d <= CLEARANCE * r_max:
                    alone = False
                    break
                ratio = r_max / d
                if nu * ratio * ratio >= QUADRUPOLE:
                    alone = False
                    break
                if 2.0 * mass[k] / total * ratio**3 >= tide:
                    alone = False
                    break
            if alone and n_cand < room:
                cand[n_cand, 0], cand[n_cand, 1] = i, j
                cand_p[n_cand] = two_pi / motion
                n_cand += 1
    if n_cand == 0:
        return found[:0]
    order = np.argsort(cand_p[:n_cand])  # the tightest first
    for q in order:
        i, j = cand[q, 0], cand[q, 1]
        if taken[i] or taken[j]:
            continue
        taken[i] = taken[j] = True
        found[count, 0], found[count, 1] = i, j
        count += 1
    return found[:count]


@numba.njit(cache=True)
def _advance_pairs(pos, vel, mass, radius, black_hole, debris, pairs, G, ic2, dt, capture):
    met = np.zeros(len(pairs), dtype=np.bool_)
    for q in range(len(pairs)):
        i, j = pairs[q, 0], pairs[q, 1]
        total = mass[i] + mass[j]
        r_floor, radiate = _floor(i, j, radius, mass, black_hole, debris, G, ic2, capture)
        x, v, code = advance_relative(
            pos[i] - pos[j], vel[i] - vel[j], mass[i], mass[j], G, ic2, dt, r_floor, radiate
        )
        if code == 0:
            continue
        com = (mass[i] * pos[i] + mass[j] * pos[j]) / total
        com_v = (mass[i] * vel[i] + mass[j] * vel[j]) / total
        pos[i] = com + mass[j] / total * x
        pos[j] = com - mass[i] / total * x
        vel[i] = com_v + mass[j] / total * v
        vel[j] = com_v - mass[i] / total * v
        met[q] = code == 2
    return met


def advance_pairs(state, pairs, G, ic2, dt, capture):
    """Advance the relative orbits of `pairs` by dt, in place. Returns the rows of
    `pairs` that have met (they are left at their meeting distance)."""
    if len(pairs) == 0 or dt == 0.0:
        return pairs[:0]
    met = _advance_pairs(
        state.pos,
        state.vel,
        state.mass,
        state.radius,
        state.black_hole,
        state.debris,
        pairs,
        G,
        ic2,
        dt,
        capture,
    )
    return pairs[met]


NO_PAIRS = np.zeros((0, 2), dtype=np.int64)


class Held:
    """The pairs an integrator currently treats as single bodies: rows (i, j), i < j,
    of the state it advances. Shared by its force, its timescale and its
    contact check."""

    def __init__(self, G, ic2):
        self.G, self.ic2 = G, ic2
        self.pairs = NO_PAIRS

    @property
    def pairs(self):
        return self._pairs

    @pairs.setter
    def pairs(self, pairs):
        self._pairs = pairs
        self._top = int(pairs.max()) if len(pairs) else -1

    def active(self, n):
        """The pairs, or None if there are none (or they belong to another state)."""
        if self._top < 0 or self._top >= n:
            return None
        return self._pairs

    def partner(self, n):
        """Per body: the other member of its pair, or -1."""
        out = np.full(n, -1, dtype=np.int64)
        pairs = self.active(n)
        if pairs is not None:
            out[pairs[:, 0]] = pairs[:, 1]
            out[pairs[:, 1]] = pairs[:, 0]
        return out


@numba.njit(cache=True)
def _merged(pairs, pos, vel, mass):
    """Positions, velocities and masses with each pair replaced by its centre of mass
    (in the row of its first member; the second member's row is dropped), and
    the mask of the rows kept."""
    n = len(mass)
    keep = np.ones(n, dtype=np.bool_)
    p, v, m = pos.copy(), vel.copy(), mass.copy()
    for q in range(len(pairs)):
        i, j = pairs[q, 0], pairs[q, 1]
        total = mass[i] + mass[j]
        for c in range(3):
            p[i, c] = (mass[i] * pos[i, c] + mass[j] * pos[j, c]) / total
            v[i, c] = (mass[i] * vel[i, c] + mass[j] * vel[j, c]) / total
        m[i] = total
        keep[j] = False
    k = n - len(pairs)
    p_out, v_out, m_out = np.empty((k, 3)), np.empty((k, 3)), np.empty(k)
    row = 0
    for a in range(n):
        if keep[a]:
            p_out[row], v_out[row], m_out[row] = p[a], v[a], m[a]
            row += 1
    return p_out, v_out, m_out, keep


@numba.njit(cache=True)
def _spread(acc, keep, pairs):
    """Accelerations of all bodies from those of the merged ones: both members of a
    pair get their centre's."""
    out = np.empty((len(keep), 3))
    row = 0
    for a in range(len(keep)):
        if keep[a]:
            out[a] = acc[row]
            row += 1
    for q in range(len(pairs)):
        out[pairs[q, 1]] = out[pairs[q, 0]]
    return out


def held_force(force, held, with_velocity):
    """`force` (as nbodystate.Integrator takes it) with the pairs of `held` acting, and
    being acted on, as single bodies at their centres of mass."""

    def wrapped(pos, mass, vel=None, debris=None):
        pairs = held.active(len(mass))
        extra = {} if debris is None else {"debris": debris}
        if pairs is None:
            return force(pos, mass, vel, **extra) if with_velocity else force(pos, mass, **extra)
        p, v, m, keep = _merged(pairs, pos, vel if with_velocity else pos, mass)
        if debris is not None:
            extra = {"debris": debris[keep]}
        acc = force(p, m, v, **extra) if with_velocity else force(p, m, **extra)
        return _spread(acc, keep, pairs)

    return wrapped


def held_timescale(timescale, held):
    """`timescale(pos, vel, mass)` with each pair of `held` counted as one body."""

    def wrapped(pos, vel, mass, debris=None):
        pairs = held.active(len(mass))
        extra = {} if debris is None else {"debris": debris}
        if pairs is None:
            return timescale(pos, vel, mass, **extra)
        p, v, m, keep = _merged(pairs, pos, vel, mass)
        if debris is not None:
            extra = {"debris": debris[keep]}
        return timescale(p, v, m, **extra)

    return wrapped


def merged_state(state, pairs, drawn=None):
    """A state with each pair replaced by one body at its centre of mass (a black hole
    if either member is one, with the horizons added), and the drawn radii to
    go with it. For deciding the pace of a step."""
    if len(pairs) == 0:
        return state, drawn
    i, j = pairs[:, 0], pairs[:, 1]
    pos, vel, mass, keep = _merged(pairs, state.pos, state.vel, state.mass)
    out = NBodyState(len(mass))
    out.pos[:], out.vel[:], out.mass[:] = pos, vel, mass
    radius = state.radius.copy()
    both = state.black_hole[i] == state.black_hole[j]
    radius[i] = np.where(
        both & state.black_hole[i],
        radius[i] + radius[j],
        np.where(
            both,
            np.maximum(radius[i], radius[j]),
            np.where(state.black_hole[i], radius[i], radius[j]),
        ),
    )
    hole = state.black_hole.copy()
    hole[i] |= hole[j]
    out.radius[:], out.black_hole[:] = radius[keep], hole[keep]
    out.debris[:], out.gas[:] = state.debris[keep], state.gas[keep]
    if drawn is not None:
        drawn = np.array(drawn, dtype=float)
        drawn[i] = np.maximum(drawn[i], drawn[j])
        drawn = drawn[keep]
    return out, drawn


def internal_acc(state, pairs, G):
    """The Newtonian acceleration of each pair's members towards each other (zero for
    all other bodies): what the held force leaves out, for the quadrupole."""
    out = np.zeros_like(state.pos)
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        x = state.pos[i] - state.pos[j]
        inv_r3 = np.sum(x * x, axis=1) ** -1.5
        out[i] = -(G * state.mass[j] * inv_r3)[:, None] * x
        out[j] = (G * state.mass[i] * inv_r3)[:, None] * x
    return out
