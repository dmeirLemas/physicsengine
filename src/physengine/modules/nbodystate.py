import numba
import numpy as np


class NBodyState:
    def __init__(self, N) -> None:
        self.N = N

        self.pos = np.zeros(shape=(N, 3), dtype=np.float64)
        self.vel = np.zeros(shape=(N, 3), dtype=np.float64)
        self.mass = np.zeros(shape=(N,), dtype=np.float64)
        # Physical radius, used for collisions. 0 means a point mass that
        # never collides.
        self.radius = np.zeros(shape=(N,), dtype=np.float64)
        # Black holes. Their radius is the horizon, 2 G M / c^2; see
        # contact_radius() and merge_pair() for how they differ from bodies.
        self.black_hole = np.zeros(shape=(N,), dtype=bool)
        # Debris: pieces of a torn or shattered body. Two pieces of debris do
        # not pull on each other, do not collide and do not set the substep
        # (see the note at DEBRIS below); with everything else they interact
        # as usual.
        self.debris = np.zeros(shape=(N,), dtype=bool)
        # Debris of a torn star: gas. Only changes how it is drawn.
        self.gas = np.zeros(shape=(N,), dtype=bool)

    def contact_radius(self):
        """Radius used for contacts: the physical radius, or for a black hole
        its capture radius (CAPTURE_RADIUS horizons)."""
        return np.where(self.black_hole, CAPTURE_RADIUS * self.radius, self.radius)


# A black hole captures what comes within 3 horizon radii = 6 G M / c^2, the
# innermost stable circular orbit of a non-rotating hole: nothing orbits
# inside it. For two black holes the reach is the sum, 6 G (m1 + m2) / c^2.
# (Unbound bodies passing between 4 and 6 G M / c^2 would in fact escape;
# that range is ignored.) The 1PN force is not valid this deep either: its
# leading correction, 1 - 4 G M / (r c^2), turns gravity into a repulsion
# inside 4 G M / c^2, so bodies must be captured before they get there.
CAPTURE_RADIUS = 3.0
# A body much lighter than the hole (at most LIGHT_BODY of its mass) is
# instead reported at HANDOVER_RADIUS horizons = 20 G M / c^2, where the
# module hands it to the exact orbit round the hole (nbody_plunge.py). That
# far out the 1PN motion is still good to ~(G M / (r c^2))^2 = 0.25%.
HANDOVER_RADIUS = 10.0
LIGHT_BODY = 0.01

# DEBRIS. A body pulled apart by a black hole's tide, or shattered in a
# collision, is replaced by a number of pieces (nbody_collisions.py). Pieces
# that start a body's radius apart have, between them, the dynamical time of
# that body (half an hour for the Sun), thousands of times shorter than a
# step, so pairs of debris are left out of the forces, the substep criterion
# and the contacts. For a torn star this is what the tide does anyway (inside
# the tidal radius it beats the star's own gravity); for collision fragments
# it means they cannot gather into a new body among themselves, only fall
# back onto the remnant.

# A black hole's tide pulls a body of mass m and radius R apart inside
# r_t = R (M / m)^(1/3) (Rees 1988, Nature 333, 523): first_contact reports
# the pair there, if that is outside the capture radius.


# ---------------------------------------------------------------------------
# Adaptive substepping for close encounters.
#
# Written by Claude under an explicit one-off override of CLAUDE.md
# (2026-11-20): review the method and the parameters.
#
# Each outer step dt is cut into kick-drift-kick substeps of dt / 2^k, so
# that every substep is at most ETA times the shortest dynamical timescale of
# any pair (see min_timescale). Within a substep the method is the ordinary
# leapfrog. Without close encounters k stays 0 and this is exactly the plain
# leapfrog.
#
# The level k can change inside a step (since 2026-11-22; before, all of dt
# used one k, capped at 256 substeps, which a deep encounter such as two
# black holes falling together exceeded by many orders of magnitude, so the
# bodies jumped through it). Substeps stay on the binary grid of the step: a
# substep of level k starts at a multiple of dt / 2^k. With the substep
# following the timescale, a fall from distance R to distance r costs about
# ln(R / r) / ETA substeps however deep it goes.
#
# Changing k makes the method lose its time symmetry, and then the energy
# error drifts instead of staying bounded. To keep the selection time
# symmetric, a substep's k must satisfy the states at both its start and its
# end: a substep whose end state asks for a higher level is redone at that
# level (Hut, Makino & McMillan 1995, ApJ 443, L93; Hairer, Lubich & Wanner
# 2006, ch. VIII.3), and the level only drops again at a grid point of the
# coarser level.
#
# Reference for the timescale criterion: the pairwise free-fall and crossing
# times are the usual choice for shared adaptive steps in few-body codes,
# e.g. Aarseth, Gravitational N-Body Simulations (2003), ch. 2.
# ---------------------------------------------------------------------------

# Substep / shortest timescale. Energy error scales roughly as ETA^2; 0.02
# keeps the preset binaries below ~1e-4 over 10 years at an average of 3 to 7
# substeps per step.
ETA = 0.02
MAX_LEVEL = 40  # smallest substep dt / 2^40


def min_timescale(pos, vel, mass, G, debris=None):
    """Shortest pairwise dynamical timescale of the system.

    For each pair, the smaller of
        free-fall time  sqrt(r^3 / (G (m_i + m_j)))   (for a circular orbit,
                                                       period / 2 pi)
        crossing time   r / |v_i - v_j|
    Returns inf for fewer than two bodies. Pairs of debris are left out.
    """
    if len(mass) < 2:
        return np.inf
    if debris is None:
        debris = np.zeros(len(mass), dtype=np.bool_)
    tau = pair_timescales(pos, vel, mass, G, debris)[0]
    # No pairs left when every body is debris.
    return float(np.min(tau)) if len(tau) else np.inf


@numba.njit(cache=True)
def pair_timescales(pos, vel, mass, G, debris):
    """Dynamical timescale of every pair (see min_timescale), with the pairs'
    separations and summed masses. Pairs of debris are left out altogether,
    so with K ordinary bodies among N the cost is K N, not N^2. Compiled: as
    NumPy array expressions this took most of a step's time for a few bodies."""
    n = len(mass)
    n_debris = 0
    for i in range(n):
        if debris[i]:
            n_debris += 1
    count = n * (n - 1) // 2 - n_debris * (n_debris - 1) // 2
    tau = np.empty(count)
    sep = np.empty(count)
    msum = np.empty(count)
    k = 0
    for i in range(n):
        if debris[i]:
            continue
        for j in range(n):
            # Each pair once: an ordinary body with every later ordinary
            # body, and with every piece of debris.
            if j == i or (not debris[j] and j < i):
                continue
            r2 = 0.0
            v2 = 0.0
            for d in range(3):
                r2 += (pos[j, d] - pos[i, d]) ** 2
                v2 += (vel[j, d] - vel[i, d]) ** 2
            r = np.sqrt(r2)
            m = mass[i] + mass[j]
            free_fall = np.sqrt(r**3 / (G * m)) if m > 0.0 else np.inf
            crossing = r / np.sqrt(v2) if v2 > 0.0 else np.inf
            tau[k] = min(free_fall, crossing)
            sep[k] = r
            msum[k] = m
            k += 1
    return tau, sep, msum


@numba.njit(cache=True)
def pair_reach(i, j, radius, mass, black_hole, debris, capture, handover, light):
    """Distance at which bodies i and j touch (see first_contact); 0 if they never do."""
    if radius[i] <= 0.0 or radius[j] <= 0.0 or (debris[i] and debris[j]):
        return 0.0
    ri = capture * radius[i] if black_hole[i] else radius[i]
    rj = capture * radius[j] if black_hole[j] else radius[j]
    reach = ri + rj
    if black_hole[i] != black_hole[j]:
        hole, body = (i, j) if black_hole[i] else (j, i)
        if mass[body] <= light * mass[hole]:
            reach = handover * radius[hole] + radius[body]
        if not debris[body] and mass[body] > 0.0:
            reach = max(reach, radius[body] * (mass[hole] / mass[body]) ** (1.0 / 3.0))
    return reach


@numba.njit(cache=True)
def hole_pace(
    pos,
    vel,
    mass,
    G,
    black_hole,
    radius,
    debris,
    floor,
    unseen,
    travel,
    zone,
    fast_period,
    capture,
    handover,
    light,
):
    """Shortest timescale worth slowing the display down for, among the pairs
    that include a black hole; inf if there is none.

    Two black holes: their dynamical timescale (see min_timescale), with the
    separation counted as at least `floor` and at least the handover radius.

    A black hole i and an ordinary body or piece of debris j: the same, but
    only if the pair is something to watch. From the relative position and
    velocity, as a two-body orbit with mu = G (m_i + m_j):
        energy       E = v^2 / 2 - mu / r          (bound if E < 0)
        ang. mom.    h = |r x v|
        eccentricity e = sqrt(1 + 2 E h^2 / mu^2)
        pericentre   q = (h^2 / mu) / (1 + e)
        period       P = 2 pi sqrt(a^3 / mu),  a = -mu / (2 E)
    (e.g. Murray & Dermott, Solar System Dynamics, 1999, sec. 2.3-2.4). The
    pair is an event, which ends by itself, if the body is unbound (it
    passes and leaves) or if q is inside the pair's reach (pair_reach: it
    will be captured, handed to the exact orbit or torn; debris bound to the
    hole also when q is inside `unseen[i]`, where it is counted as accreted).
    A bound orbit that is not an event counts only if its period is at
    least `fast_period`: a faster one is an ordinary fast orbit, like a
    moon's, and slowing down for it would never end. A body closer to the
    hole than `unseen[i]` (what cannot be told apart from the hole on
    screen; 0 without a view) does not count at all.

    For an event within `zone` of the hole, the time the body takes to move
    `travel` relative to the hole also counts (0: not used).
    """
    n = len(mass)
    best = np.inf
    for i in range(n):
        if not black_hole[i]:
            continue
        for j in range(n):
            if j == i or (black_hole[j] and j < i):
                continue
            r2 = 0.0
            v2 = 0.0
            for d in range(3):
                r2 += (pos[j, d] - pos[i, d]) ** 2
                v2 += (vel[j, d] - vel[i, d]) ** 2
            sep = np.sqrt(r2)
            m = mass[i] + mass[j]
            event = False
            if not black_hole[j]:
                if sep < unseen[i]:
                    continue
                mu = G * m
                if mu > 0.0 and sep > 0.0:
                    e_orbit = 0.5 * v2 - mu / sep
                    cx = (pos[j, 1] - pos[i, 1]) * (vel[j, 2] - vel[i, 2]) - (
                        pos[j, 2] - pos[i, 2]
                    ) * (vel[j, 1] - vel[i, 1])
                    cy = (pos[j, 2] - pos[i, 2]) * (vel[j, 0] - vel[i, 0]) - (
                        pos[j, 0] - pos[i, 0]
                    ) * (vel[j, 2] - vel[i, 2])
                    cz = (pos[j, 0] - pos[i, 0]) * (vel[j, 1] - vel[i, 1]) - (
                        pos[j, 1] - pos[i, 1]
                    ) * (vel[j, 0] - vel[i, 0])
                    h2 = cx * cx + cy * cy + cz * cz
                    ecc = np.sqrt(max(1.0 + 2.0 * e_orbit * h2 / (mu * mu), 0.0))
                    q = h2 / mu / (1.0 + ecc)
                    reach = pair_reach(
                        i, j, radius, mass, black_hole, debris, capture, handover, light
                    )
                    if debris[j] and e_orbit < 0.0:
                        reach = max(reach, unseen[i])
                    event = e_orbit >= 0.0 or q < reach
                    if not event:
                        a = -mu / (2.0 * e_orbit)
                        if 2.0 * np.pi * np.sqrt(a**3 / mu) < fast_period:
                            continue
            r = max(sep, floor, handover * radius[i])
            if m > 0.0:
                best = min(best, np.sqrt(r**3 / (G * m)))
            if v2 > 0.0:
                best = min(best, r / np.sqrt(v2))
                if event and travel > 0.0 and r2 < zone * zone:
                    best = min(best, travel / np.sqrt(v2))
    return best


def first_contact(pos0, v, radius, h, mass=None, black_hole=None, debris=None, partner=None):
    """First pair of bodies that touch during a drift.

    In a leapfrog drift every body moves in a straight line, pos0 + v t for
    0 <= t <= h. For each pair the closest approach on that segment is at
    t* = clip(-(r0 . dv) / |dv|^2, 0, h); the pair touches if the distance
    there is below their reach. Checking the whole segment, not only its
    end, catches bodies that would pass through each other within one
    substep. Bodies with radius 0 never touch.

    Reach of a pair: the sum of the radii, a black hole counting with its
    capture radius (CAPTURE_RADIUS horizons), or with HANDOVER_RADIUS
    horizons for a body much lighter than it; for a black hole and a body
    that is not debris, the body's tidal radius R (M / m)^(1/3) if that is
    larger. Two pieces of debris never touch, and neither do a body and its
    `partner` (per body, the index of the other member of a pair whose orbit
    is followed analytically, nbody_binary.py, or -1).

    Returns (i, j) for the touching pair with the earliest t*, or None.
    """
    n = len(radius)
    mass = np.ones(n) if mass is None else mass
    black_hole = np.zeros(n, dtype=np.bool_) if black_hole is None else black_hole
    debris = np.zeros(n, dtype=np.bool_) if debris is None else debris
    partner = np.full(n, -1, dtype=np.int64) if partner is None else partner
    i, j = _first_contact(
        pos0,
        v,
        radius,
        h,
        mass,
        black_hole,
        debris,
        CAPTURE_RADIUS,
        HANDOVER_RADIUS,
        LIGHT_BODY,
        partner,
    )
    return None if i < 0 else (i, j)


@numba.njit(cache=True)
def _first_contact(pos0, v, radius, h, mass, black_hole, debris, capture, handover, light, partner):
    n = len(radius)
    best_i, best_j, best_t = -1, -1, np.inf
    for i in range(n):
        if debris[i] or radius[i] <= 0.0:
            continue  # pairs of debris never touch: start from the ordinary bodies
        for j in range(n):
            if j == i or (not debris[j] and j < i) or radius[j] <= 0.0 or partner[i] == j:
                continue
            reach = pair_reach(i, j, radius, mass, black_hole, debris, capture, handover, light)
            if reach <= 0.0:
                continue
            rdv = 0.0
            dv2 = 0.0
            for d in range(3):
                dv = v[j, d] - v[i, d]
                rdv += (pos0[j, d] - pos0[i, d]) * dv
                dv2 += dv * dv
            t = -rdv / dv2 if dv2 > 0.0 else 0.0
            t = min(max(t, 0.0), h)
            c2 = 0.0
            for d in range(3):
                c2 += (pos0[j, d] - pos0[i, d] + (v[j, d] - v[i, d]) * t) ** 2
            if np.sqrt(c2) < reach and t < best_t:
                best_i, best_j, best_t = min(i, j), max(i, j), t
    return best_i, best_j


def merge_pair(state, keep, gone):
    """Merge body `gone` into body `keep` (perfectly inelastic collision).

    Conserves mass and linear momentum; the merged body sits at the pair's
    centre of mass with its velocity. Kinetic energy of the relative motion
    is lost, as in a real merger. Two ordinary bodies give a body with the
    volume of both (radius (R1^3 + R2^3)^(1/3)). If either is a black hole
    the result is a black hole whose horizon, 2 G M / c^2, is proportional
    to its mass. Two black holes radiate part of their mass as
    gravitational waves in the merger (postnewtonian.merger_mass: 4.8% for
    equal masses). Only body `keep` is changed; the caller removes `gone`.
    """
    m1, m2 = state.mass[keep], state.mass[gone]
    m = m1 + m2
    if m > 0.0:
        state.pos[keep] = (m1 * state.pos[keep] + m2 * state.pos[gone]) / m
        state.vel[keep] = (m1 * state.vel[keep] + m2 * state.vel[gone]) / m
    holes = [i for i in (keep, gone) if state.black_hole[i]]
    if holes:
        hole_mass = sum(state.mass[i] for i in holes)
        hole_radius = sum(state.radius[i] for i in holes)
        if len(holes) == 2 and m > 0.0:
            from physengine.modules.postnewtonian import merger_mass

            m = merger_mass(m1, m2)
        state.radius[keep] = hole_radius * m / hole_mass if hole_mass > 0.0 else hole_radius
        state.black_hole[keep] = True
    else:
        state.radius[keep] = np.cbrt(state.radius[keep] ** 3 + state.radius[gone] ** 3)
    state.mass[keep] = m


class Integrator:
    """Kick-drift-kick leapfrog with adaptive substeps for close encounters.

    `force_func(pos, mass)` returns accelerations. With `timescale_func(pos,
    vel, mass)` (see min_timescale) each step is cut into substeps of
    dt / 2^k; without it every step is a single plain leapfrog step.

    With velocity_dependent=True the force is `force_func(pos, mass, vel)`
    (for example 1PN gravity). The closing half-kick then needs the velocity
    it produces; it uses a predicted one, v_half + h/2 a(x, v_half), which
    keeps the method second order (added by Claude, 2026-11-20, under the
    override for the 1PN work; it is no longer exactly symplectic).
    """

    def __init__(
        self,
        force_func,
        timescale_func=None,
        eta=ETA,
        max_level=MAX_LEVEL,
        velocity_dependent=False,
    ):
        self.velocity_dependent = velocity_dependent
        self.a = force_func
        self.timescale = timescale_func
        self.eta = eta
        self.max_level = max_level
        self.acc_cache = None
        self.last_level = 0  # deepest substep level k used by the last step
        self.last_substeps = 0  # substeps computed by the last step
        self.last_time = 0.0  # time covered by the last step (set by nbodymodule.advance_step)
        self.debris = None  # the state's debris flags during advance(), if it has any
        # nbody_binary.Held, if the force treats some pairs as single bodies:
        # their members do not touch each other.
        self.held = None

    def invalidate(self):
        self.acc_cache = None

    def _acc(self, pos, mass, vel):
        # The debris flags are only passed when there is debris, so force
        # functions that do not know about it keep working.
        extra = {} if self.debris is None else {"debris": self.debris}
        if self.velocity_dependent:
            return self.a(pos, mass, vel, **extra)
        return self.a(pos, mass, **extra)

    def level(self, state, dt):
        """Substep level k for a step of dt wanted by the current state."""
        return self._level(state.pos, state.vel, state.mass, dt)

    def _level(self, pos, vel, mass, dt):
        if self.timescale is None or len(mass) < 2:
            return 0
        extra = {} if self.debris is None else {"debris": self.debris}
        tau = self.timescale(pos, vel, mass, **extra)
        if not np.isfinite(tau):
            return 0
        if tau <= 0.0:
            return self.max_level
        k = int(np.ceil(np.log2(dt / (self.eta * tau))))
        return min(max(k, 0), self.max_level)

    def advance(self, state, dt, check_contact=False, max_substeps=None):
        """Advance the state by dt, or less if two bodies touch.

        With check_contact, stops after the first substep in which two
        bodies touch (see first_contact, NBodyState.contact_radius). With
        max_substeps, stops at the end of a substep once that many have been
        computed (counting redone ones), so that a very deep encounter
        cannot take unbounded time. Returns (pair, t): the touching pair or
        None, and the time advanced.
        """
        mass = state.mass
        self.debris = state.debris if state.debris.any() else None
        if self.acc_cache is None or self.acc_cache.shape != state.pos.shape:
            self.acc_cache = self._acc(state.pos, mass, state.vel)
        acc = self.acc_cache
        pos, vel = state.pos.copy(), state.vel.copy()
        partner = None if self.held is None else self.held.partner(len(mass))

        # Time within the step in ticks of dt / 2^max_level.
        total = 1 << self.max_level
        done = 0
        k = self._level(pos, vel, mass, dt)
        deepest, count, pair = k, 0, None
        while done < total:
            if max_substeps is not None and count >= max_substeps:
                break
            h = dt / (1 << k)
            new_pos, new_vel, new_acc, v_half = self._kdk(pos, vel, acc, mass, h)
            count += 1
            k_end = self._level(new_pos, new_vel, mass, dt)
            if k_end > k:
                k = k_end  # redo from the same point with the end state's level
                continue
            if check_contact:
                pair = first_contact(
                    pos, v_half, state.radius, h, mass, state.black_hole, state.debris, partner
                )
            pos, vel, acc = new_pos, new_vel, new_acc
            done += total >> k
            deepest = max(deepest, k)
            if pair is not None:
                break
            # Coarsen where the state allows it and the grid lines up.
            while k > k_end and done % (total >> (k - 1)) == 0:
                k -= 1

        self.last_level = deepest
        self.last_substeps = count
        self.acc_cache = acc
        state.pos[:, :] = pos
        state.vel[:, :] = vel
        return pair, (dt if done == total else dt * done / total)

    def _kdk(self, pos, vel, acc, mass, h):
        """One kick-drift-kick substep; returns new pos, vel, acc and the drift velocity."""
        v_half = vel + (h / 2) * acc
        pos = pos + h * v_half
        if self.velocity_dependent:
            v_pred = v_half + (h / 2) * self._acc(pos, mass, v_half)
            acc = self._acc(pos, mass, v_pred)
        else:
            acc = self._acc(pos, mass, None)
        return pos, v_half + (h / 2) * acc, acc, v_half

    def step(self, state, dt, n):
        """n steps of dt, without collisions."""
        for _ in range(n):
            self.advance(state, dt)


def acc(pos, mass, G, eps, debris=None):
    diff = pos[None, :, :] - pos[:, None, :]  # (N, N, 3)
    dist2 = np.sum(diff**2, axis=-1) + eps**2  # (N, N)

    # A body does not pull on itself, nor on one at exactly the same point.
    same = dist2 == 0.0
    inv_d3 = np.where(same, 0.0, np.where(same, 1.0, dist2) ** -1.5)
    if debris is not None:
        inv_d3[np.ix_(debris, debris)] = 0.0  # debris does not pull on debris

    return G * np.einsum("ij,ijk->ik", inv_d3 * mass[None, :], diff)


def sun_earth_scenario():
    G = 4.0 * np.pi**2  # AU^3 / (M_sun yr^2)
    M_sun, m_earth, r = 1.0, 3.0e-6, 1.0

    # Circular two-body orbit: relative speed sqrt(G (M + m) / r), split by
    # mass ratio so the total momentum is zero.
    v_rel = np.sqrt(G * (M_sun + m_earth) / r)
    v_sun = -m_earth / (M_sun + m_earth) * v_rel
    v_earth = M_sun / (M_sun + m_earth) * v_rel

    pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64)
    vel = np.array([[0.0, v_sun, 0.0], [0.0, v_earth, 0.0]], dtype=np.float64)
    mass = np.array([M_sun, m_earth], dtype=np.float64)

    nb = NBodyState(2)
    nb.pos[:, :] = pos
    nb.vel[:, :] = vel
    nb.mass[:] = mass

    return nb


def total_momentum(state):
    return state.mass[:] @ state.vel


def total_angular_momentum(state):
    return state.mass[:] @ np.cross(state.pos, state.vel)


def energy(pos, vel, mass, G, eps):
    kin = 0.5 * mass * np.sum(vel**2, axis=-1)
    N = mass.shape[0]

    i, j = np.triu_indices(N, k=1)

    dpos = pos[i] - pos[j]
    dist_sq = np.sum(dpos**2, axis=-1)

    dist = np.sqrt(dist_sq + eps**2)
    pairwise_pot = np.where(
        dist > 0.0, -G * mass[i] * mass[j] / np.where(dist > 0.0, dist, 1.0), 0.0
    )
    pot = np.sum(pairwise_pot)

    return np.sum(kin) + pot
