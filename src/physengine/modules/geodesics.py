"""Massive bodies around a Schwarzschild black hole: exact geodesics.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). Review the physics.

Units G = c = M = 1, hole at the origin. Positions x = r n with r the areal
(Schwarzschild) radius and n the direction, so x is the hole's spherical
coordinates written as a vector. For a free body with proper time tau,
energy E = (1 - 2/r) dt/dtau and angular momentum L = |x x dx/dtau|, the
geodesic equations reduce to (e.g. Hartle 2003, Gravity, ch. 9)

    (dr/dtau)^2 = E^2 - V(r),   V(r) = (1 - 2/r) (1 + L^2 / r^2),
    dphi/dtau = L / r^2.

Differentiating, the radial acceleration is -1/r^2 + L^2/r^3 - 3 L^2/r^4, and
the angular motion conserves L as for any central force. Both together are
the vector equation

    d^2x/dtau^2 = -x / r^3 (1 + 3 L^2 / r^2),

Newtonian gravity plus a 1/r^4 term; it gives the Binet equation
u'' + u = 1/L^2 + 3 u^2 (u = 1/r, ' = d/dphi) of the Schwarzschild orbit,
with its perihelion precession, innermost stable circular orbit at r = 6 and
the plunge inside it. E follows from the state:
    E^2 = (dr/dtau)^2 + (1 - 2/r)(1 + L^2/r^2).

All bodies share one clock, the coordinate time t (the time of a distant
observer), and each takes its own RK4 steps in it: with w = dx/dtau,
    dx/dt = w (1 - 2/r) / E,    dw/dt = a_tau (1 - 2/r) / E.
A body falling in therefore slows down and freezes just outside r = 2 in t,
as a distant observer sees it; it is removed once r < 2 (1 + CAPTURE_MARGIN).

Bodies also pull on each other with Newtonian gravity (softened), added to
d^2x/dtau^2. This is the weak-field, slow-motion limit of their mutual
gravity and assumes the bodies are much lighter than the hole.
"""

import numba
import numpy as np

CAPTURE_MARGIN = 1e-3  # removed at r < 2 (1 + this): already frozen and faded


@numba.njit(cache=True)
def energy(x, w):
    """E = (1 - 2/r) dt/dtau of each body, from position and w = dx/dtau."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        r = np.sqrt(x[i, 0] ** 2 + x[i, 1] ** 2 + x[i, 2] ** 2)
        vr = (x[i, 0] * w[i, 0] + x[i, 1] * w[i, 1] + x[i, 2] * w[i, 2]) / r
        w2 = w[i, 0] ** 2 + w[i, 1] ** 2 + w[i, 2] ** 2
        out[i] = np.sqrt(max(vr * vr + (1.0 - 2.0 / r) * (1.0 + w2 - vr * vr), 0.0))
    return out


def angular_momentum(x, w):
    """L = |x x dx/dtau| of each body."""
    return np.linalg.norm(np.cross(x, w), axis=1)


@numba.njit(parallel=True, cache=True)
def newtonian_mutual(x, mass, alive, eps, out):
    """Softened Newtonian pull of the bodies on each other (direct sum)."""
    n = len(x)
    e2 = eps * eps
    for i in numba.prange(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        if alive[i]:
            for j in range(n):
                if j == i or not alive[j] or mass[j] == 0.0:
                    continue
                dx = x[j, 0] - x[i, 0]
                dy = x[j, 1] - x[i, 1]
                dz = x[j, 2] - x[i, 2]
                d2 = dx * dx + dy * dy + dz * dz + e2
                f = mass[j] / (d2 * np.sqrt(d2))
                ax += f * dx
                ay += f * dy
                az += f * dz
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az


@numba.njit(cache=True)
def _deriv_one(x0, x1, x2, w0, w1, w2):
    """d/dt of (x, w) of one body: dx/dt = w (1 - 2/r) / E, dw/dt = a_tau (1 - 2/r) / E."""
    r2 = x0 * x0 + x1 * x1 + x2 * x2
    r = np.sqrt(r2)
    lx = x1 * w2 - x2 * w1
    ly = x2 * w0 - x0 * w2
    lz = x0 * w1 - x1 * w0
    l2 = lx * lx + ly * ly + lz * lz
    vr = (x0 * w0 + x1 * w1 + x2 * w2) / r
    w2s = w0 * w0 + w1 * w1 + w2 * w2
    e = np.sqrt(max(vr * vr + (1.0 - 2.0 / r) * (1.0 + w2s - vr * vr), 1e-300))
    lapse = (1.0 - 2.0 / r) / e
    f = -(1.0 + 3.0 * l2 / r2) / (r2 * r) * lapse
    return w0 * lapse, w1 * lapse, w2 * lapse, f * x0, f * x1, f * x2


@numba.njit(parallel=True, cache=True)
def _drift(x, w, alive, duration, eta, max_step, capture_r, t0, captured_at):
    """Advance every live body along its geodesic by `duration` of coordinate time.

    Each body takes its own RK4 steps of eta r^1.5 (at most max_step), so a
    particle near the hole does not slow down the others. Bodies that come
    within capture_r are marked captured (time in captured_at).
    """
    for i in numba.prange(len(x)):
        if not alive[i]:
            continue
        s = np.empty(6)
        s[0], s[1], s[2] = x[i, 0], x[i, 1], x[i, 2]
        s[3], s[4], s[5] = w[i, 0], w[i, 1], w[i, 2]
        done = 0.0
        while duration - done > 1e-12:
            r = np.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
            h = min(eta * r * np.sqrt(r), max_step, duration - done)
            k1 = np.array(_deriv_one(s[0], s[1], s[2], s[3], s[4], s[5]))
            m = s + 0.5 * h * k1
            k2 = np.array(_deriv_one(m[0], m[1], m[2], m[3], m[4], m[5]))
            m = s + 0.5 * h * k2
            k3 = np.array(_deriv_one(m[0], m[1], m[2], m[3], m[4], m[5]))
            m = s + h * k3
            k4 = np.array(_deriv_one(m[0], m[1], m[2], m[3], m[4], m[5]))
            s = s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            done += h
            if s[0] * s[0] + s[1] * s[1] + s[2] * s[2] < capture_r * capture_r:
                alive[i] = False
                captured_at[i] = t0 + done
                break
        x[i, 0], x[i, 1], x[i, 2] = s[0], s[1], s[2]
        w[i, 0], w[i, 1], w[i, 2] = s[3], s[4], s[5]


class BodySystem:
    """Bodies moving on geodesics of a Schwarzschild hole, pulling on each other.

    Arrays (one row per body): x positions, w = dx/dtau, mass (for the mutual
    pull only; the hole's mass is 1), alive (False once captured).
    """

    ETA = 0.01  # each body's step: this fraction of its orbital time scale r^1.5
    MAX_STEP = 2.0  # [M]
    # Mutual pulls are applied as kicks this far apart [M]: short against the
    # slowest time scale they act on (a star's own dynamical time, ~300 M).
    OUTER_STEP = 1.0

    def __init__(self, x, w, mass=None, eps=0.05):
        self.x = np.array(x, dtype=float).reshape(-1, 3)
        self.w = np.array(w, dtype=float).reshape(-1, 3)
        n = len(self.x)
        self.mass = np.zeros(n) if mass is None else np.array(mass, dtype=float)
        self.alive = np.ones(n, dtype=bool)
        self.eps = eps  # softening of the mutual pull [M]
        self.t = 0.0  # coordinate time [M]
        self.captured_at = np.full(n, np.nan)  # t when each body was removed
        self._acc = np.zeros_like(self.x)
        self._acc_fresh = False  # _acc belongs to the current positions
        self.history = None  # a History, if attached: sampled during advance()
        self.hydro = None  # a blackhole_sph.Gas, if the bodies are gas

    @property
    def n(self):
        return len(self.x)

    def add(self, x, w, mass=0.0):
        self.x = np.vstack([self.x, np.reshape(x, (1, 3))])
        self.w = np.vstack([self.w, np.reshape(w, (1, 3))])
        self.mass = np.append(self.mass, mass)
        self.alive = np.append(self.alive, True)
        self.captured_at = np.append(self.captured_at, np.nan)
        self._acc = np.zeros_like(self.x)
        self._acc_fresh = False
        if self.history is not None:
            self.history.add_body(self, len(self.x) - 1)

    def r(self):
        return np.linalg.norm(self.x, axis=1)

    def _kick(self, dt):
        """Mutual Newtonian pull over dt (in each body's proper time, dtau = dt (1-2/r)/E)."""
        if not np.any(self.mass[self.alive] > 0.0):
            return
        if not self._acc_fresh:
            newtonian_mutual(self.x, self.mass, self.alive, self.eps, self._acc)
            if self.hydro is not None:
                self._acc += self.hydro.accelerations(self.x, self.w, self.mass, self.alive)
            self._acc_fresh = True
        live = self.alive
        lapse = (1.0 - 2.0 / self.r()[live]) / energy(self.x[live], self.w[live])
        self.w[live] += (dt * lapse)[:, None] * self._acc[live]
        if self.hydro is not None:
            self.hydro.heat(dt * lapse, live)

    def advance(self, duration):
        """Advance all bodies by `duration` of coordinate time.

        Each outer step: half a kick of mutual gravity, every body along its
        exact geodesic (own RK4 steps), half a kick (Strang splitting: second
        order in the mutual pull, fourth order in the hole's).
        """
        end = self.t + duration
        while end - self.t > 1e-12:
            h = min(self.OUTER_STEP, end - self.t)
            if self.history is not None:  # land exactly on the next sample time
                h = min(h, self.history.next_time() - self.t)
            self._kick(0.5 * h)
            _drift(
                self.x,
                self.w,
                self.alive,
                h,
                self.ETA,
                self.MAX_STEP,
                2.0 * (1.0 + CAPTURE_MARGIN),
                self.t,
                self.captured_at,
            )
            self._acc_fresh = False
            self._kick(0.5 * h)
            self.t += h
            if self.history is not None and self.t >= self.history.next_time() - 1e-9:
                self.history.record(self)

    def past(self, which, spacing, count):
        """(x, w) of bodies `which` at t - spacing, t - 2 spacing, ... (count samples).

        Integrates them together backwards (reversing w retraces a free path;
        their mutual pulls are included); a body whose past would come out of
        the hole is held at its present state instead.
        """
        back = BodySystem(self.x[which].copy(), -self.w[which].copy(), self.mass[which], self.eps)
        back.OUTER_STEP = max(self.OUTER_STEP, min(spacing, 4.0))
        held = ~back.alive
        for _ in range(count):
            back.advance(spacing)
            held |= ~back.alive
            yield (
                np.where(held[:, None], self.x[which], back.x),
                np.where(held[:, None], self.w[which], -back.w),
            )

    def dx_dt(self, x=None, w=None):
        """Coordinate velocity dx/dt = w (1 - 2/r) / E."""
        x = self.x if x is None else x
        w = self.w if w is None else w
        r = np.linalg.norm(x, axis=1)
        return w * ((1.0 - 2.0 / r) / np.maximum(energy(x, w), 1e-300))[:, None]


class History:
    """Past states of the bodies of a BodySystem, for light-travel delays.

    A ring buffer of (x, w) sampled every `spacing` of coordinate time, the
    last `length` samples (float32). Before t = 0 (and before a body was
    added) it is filled by integrating the bodies backwards (reversing w
    retraces a free path; their mutual pulls are included), so a body has a
    past to be seen from; a body whose past would come out of the hole is held
    at its starting state instead.
    """

    def __init__(self, system, spacing=0.5, length=1600):
        self.spacing = float(spacing)
        self.length = int(length)
        n = system.n
        self.x = np.full((self.length, n, 3), np.nan, dtype=np.float32)
        self.w = np.full((self.length, n, 3), np.nan, dtype=np.float32)
        self.k_last = int(np.floor(system.t / self.spacing + 1e-9))  # newest sample index
        self._prefill(system, np.arange(n))
        self._store(self.k_last, system.x, system.w, np.arange(n))
        system.history = self

    def next_time(self):
        return (self.k_last + 1) * self.spacing

    def _slot(self, k):
        return k % self.length

    def _store(self, k, x, w, which):
        s = self._slot(k)
        self.x[s, which] = x
        self.w[s, which] = w

    def record(self, system):
        self.k_last += 1
        s = self._slot(self.k_last)
        self.x[s] = system.x
        self.w[s] = system.w

    def _prefill(self, system, which):
        """The past of bodies `which`, from system.past (integrated backwards from now)."""
        if len(which) == 0:
            return
        for m, (x, w) in enumerate(system.past(which, self.spacing, self.length - 1), start=1):
            self._store(self.k_last - m, x, w, which)

    def add_body(self, system, j):
        pad = np.full((self.length, 1, 3), np.nan, dtype=np.float32)
        self.x = np.concatenate([self.x, pad], axis=1)
        self.w = np.concatenate([self.w, pad.copy()], axis=1)
        self._prefill(system, np.array([j]))
        self._store(self.k_last, system.x[j], system.w[j], j)

    def state_at(self, system, which, t):
        """(x, w) of bodies `which` at coordinate times t (cubic Hermite in dx/dt), and whether known.

        Between the newest sample and now, the current state is the right end.
        Times older than the buffer are unknown.
        """
        which = np.asarray(which)
        t = np.minimum(np.asarray(t, dtype=float), system.t)
        k0 = np.floor(t / self.spacing).astype(int)
        known = k0 > self.k_last - self.length
        k0 = np.where(known, k0, self.k_last)
        x0 = self.x[self._slot(k0), which].astype(float)
        w0 = self.w[self._slot(k0), which].astype(float)
        t0 = k0 * self.spacing
        newest = k0 >= self.k_last
        k1 = np.minimum(k0 + 1, self.k_last)
        x1 = np.where(newest[:, None], system.x[which], self.x[self._slot(k1), which])
        w1 = np.where(newest[:, None], system.w[which], self.w[self._slot(k1), which])
        t1 = np.where(newest, system.t, (k0 + 1) * self.spacing)
        span = np.maximum(t1 - t0, 1e-12)
        f = np.clip((t - t0) / span, 0.0, 1.0)[:, None]
        v0 = system.dx_dt(x0, w0) * span[:, None]
        v1 = system.dx_dt(x1, w1) * span[:, None]
        f2, f3 = f * f, f * f * f
        x = (
            (2 * f3 - 3 * f2 + 1) * x0
            + (f3 - 2 * f2 + f) * v0
            + (-2 * f3 + 3 * f2) * x1
            + (f3 - f2) * v1
        )
        w = (1 - f) * w0 + f * w1
        known &= np.all(np.isfinite(x), axis=1)
        return x, w, known


# ------------------------------------------------------- initial conditions


def effective_potential(r, L):
    """V(r) = (1 - 2/r)(1 + L^2 / r^2): E^2 - V = (dr/dtau)^2."""
    return (1.0 - 2.0 / r) * (1.0 + L * L / (r * r))


def circular_orbit(r):
    """(E, L) of the circular orbit at r > 3: L^2 = r^2 / (r - 3), E = (1 - 2/r) / sqrt(1 - 3/r).

    Stable for r > 6 (the ISCO), unstable between 3 and 6.
    """
    if r <= 3.0:
        raise ValueError("no circular orbits inside the photon sphere (r = 3)")
    L = r / np.sqrt(r - 3.0)
    E = (1.0 - 2.0 / r) / np.sqrt(1.0 - 3.0 / r)
    return E, L


def bound_orbit(r_p, r_a):
    """(E, L) of the bound orbit with turning points r_p < r_a.

    From V(r_p) = V(r_a) = E^2:
        L^2 = (2/r_p - 2/r_a) / ((1 - 2/r_p)/r_p^2 - (1 - 2/r_a)/r_a^2).
    Raises ValueError if no orbit oscillates between them (r_p inside the
    potential barrier, so the body would plunge instead).
    """
    if r_a < r_p:
        r_p, r_a = r_a, r_p
    if np.isclose(r_p, r_a):
        return circular_orbit(r_p)
    num = 2.0 / r_p - 2.0 / r_a
    den = (1.0 - 2.0 / r_p) / r_p**2 - (1.0 - 2.0 / r_a) / r_a**2
    if r_p <= 2.0 or den <= 0.0:
        raise ValueError("no bound orbit with these turning points")
    L = np.sqrt(num / den)
    E2 = effective_potential(r_a, L)
    between = np.linspace(r_p, r_a, 2001)[1:-1]
    if np.any(effective_potential(between, L) > E2 * (1.0 + 1e-12)):
        raise ValueError("no bound orbit with these turning points")
    # r_p must be a turning point from outside: V rising inward just inside it.
    if effective_potential(r_p * (1.0 - 1e-6), L) < E2:
        raise ValueError("pericentre inside the potential barrier: the body would plunge")
    return np.sqrt(E2), L


def orbit_state(r, L, E, inward=True, normal=(0.0, 1.0, 0.0), direction=(1.0, 0.0, 0.0)):
    """Position and w = dx/dtau at radius r on an orbit with energy E, angular momentum L.

    The orbit plane has unit normal `normal` (the body goes counter-clockwise
    about it); the body is at r along `direction` (projected into the plane).
    Radial speed from (dr/dtau)^2 = E^2 - V(r), inward or outward.
    """
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    d = np.asarray(direction, dtype=float)
    d = d - (d @ n) * n
    d /= np.linalg.norm(d)
    t = np.cross(n, d)
    vr2 = E * E - effective_potential(r, L)
    vr = np.sqrt(max(vr2, 0.0)) * (-1.0 if inward else 1.0)
    return r * d, vr * d + (L / r) * t


def orbit_normal(inclination, node=0.0):
    """Unit normal of an orbit plane tilted by `inclination` (radians) from the disc plane.

    The disc lies in the xz-plane and spins about +y; inclination 0 is an
    orbit going the same way as the disc. `node` turns the tilt axis about y.
    """
    n = np.array([0.0, np.cos(inclination), np.sin(inclination)])
    c, s = np.cos(node), np.sin(node)
    return np.array([c * n[0] + s * n[2], n[1], -s * n[0] + c * n[2]])


def precession_per_orbit(r_p, r_a, n=400):
    """Exact advance of the pericentre per radial period, radians.

    Delta phi = 2 integral_{r_p}^{r_a} (L / r^2) / sqrt(E^2 - V(r)) dr - 2 pi,
    by Gauss-Legendre quadrature with r = r_p + (r_a - r_p)(1 - cos s)/2,
    which removes the inverse square roots at both turning points. Weak-field
    limit: 6 pi M / (a (1 - e^2)).
    """
    E, L = bound_orbit(r_p, r_a)
    x, wts = np.polynomial.legendre.leggauss(n)
    s = 0.5 * np.pi * (x + 1.0)
    r = r_p + 0.5 * (r_a - r_p) * (1.0 - np.cos(s))
    dr_ds = 0.5 * (r_a - r_p) * np.sin(s)
    radicand = np.maximum(E * E - effective_potential(r, L), 1e-300)
    integral = 0.5 * np.pi * np.sum(wts * (L / r**2) / np.sqrt(radicand) * dr_ds)
    return 2.0 * integral - 2.0 * np.pi
