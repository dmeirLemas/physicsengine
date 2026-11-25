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

Speed (2026-11-23). The kernels below run every frame on a few thousand
bodies, which is little work per call: what they cost is set as much by how
they are written as by the physics. They avoid allocating inside loops, the
mutual pull is a branch-free sum the compiler can vectorise, and they run on
KERNEL_THREADS threads (see few_threads).
"""

from contextlib import contextmanager

import numba
import numpy as np

CAPTURE_MARGIN = 1e-3  # removed at r < 2 (1 + this): already frozen and faded

# Threads for the black-hole module's parallel kernels. Each call is a
# fraction of a millisecond of work; on more threads than this the time to
# wake them (and the cores they keep busy while they wait for the next call)
# costs more than the extra threads save, and on a laptop with integrated
# graphics the busy cores slow the GPU's ray trace down (measured on the
# 14-core dev machine: 6 threads against the default 20, see TODO.md).
KERNEL_THREADS = max(1, min(6, numba.config.NUMBA_NUM_THREADS))


@contextmanager
def few_threads(count=None):
    """Run Numba's parallel loops on at most `count` (KERNEL_THREADS) threads inside the block.

    Not physics: the results do not depend on the number of threads.
    """
    before = numba.get_num_threads()
    numba.set_num_threads(max(1, min(KERNEL_THREADS if count is None else count, before)))
    try:
        yield
    finally:
        numba.set_num_threads(before)


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


@numba.njit(parallel=True, cache=True, fastmath={"reassoc", "contract"})
def newtonian_mutual(x, mass, alive, eps, out):
    """Softened Newtonian pull of the bodies on each other (direct sum).

    out[i] = sum_j m_j d / (|d|^2 + eps^2)^1.5, d = x_j - x_i, over the live
    bodies j with mass; zero for bodies that are not alive. Every pair is
    summed (no approximation). The sources are first packed into contiguous
    arrays and the inner loop has no branches: a body's own term is exactly
    zero (d = 0), so it need not be skipped, and with eps = 0 a pair at zero
    distance is left out. The sum may be reordered by the compiler (to use
    vector instructions), which changes the result in the last digits only.
    """
    n = len(x)
    e2 = eps * eps
    sx = np.empty(n)
    sy = np.empty(n)
    sz = np.empty(n)
    sm = np.empty(n)
    ns = 0
    for j in range(n):
        if alive[j] and mass[j] != 0.0:
            sx[ns] = x[j, 0]
            sy[ns] = x[j, 1]
            sz[ns] = x[j, 2]
            sm[ns] = mass[j]
            ns += 1
    for i in numba.prange(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        if alive[i]:
            xi = x[i, 0]
            yi = x[i, 1]
            zi = x[i, 2]
            for j in range(ns):
                dx = sx[j] - xi
                dy = sy[j] - yi
                dz = sz[j] - zi
                d2 = dx * dx + dy * dy + dz * dz + e2
                f = sm[j] / (d2 * np.sqrt(d2)) if d2 > 0.0 else 0.0
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
        s0, s1, s2 = x[i, 0], x[i, 1], x[i, 2]
        s3, s4, s5 = w[i, 0], w[i, 1], w[i, 2]
        done = 0.0
        while duration - done > 1e-12:
            r = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
            h = min(eta * r * np.sqrt(r), max_step, duration - done)
            hh = 0.5 * h
            a0, a1, a2, a3, a4, a5 = _deriv_one(s0, s1, s2, s3, s4, s5)
            b0, b1, b2, b3, b4, b5 = _deriv_one(
                s0 + hh * a0, s1 + hh * a1, s2 + hh * a2, s3 + hh * a3, s4 + hh * a4, s5 + hh * a5
            )
            c0, c1, c2, c3, c4, c5 = _deriv_one(
                s0 + hh * b0, s1 + hh * b1, s2 + hh * b2, s3 + hh * b3, s4 + hh * b4, s5 + hh * b5
            )
            d0, d1, d2, d3, d4, d5 = _deriv_one(
                s0 + h * c0, s1 + h * c1, s2 + h * c2, s3 + h * c3, s4 + h * c4, s5 + h * c5
            )
            h6 = h / 6.0
            s0 = s0 + h6 * (a0 + 2.0 * b0 + 2.0 * c0 + d0)
            s1 = s1 + h6 * (a1 + 2.0 * b1 + 2.0 * c1 + d1)
            s2 = s2 + h6 * (a2 + 2.0 * b2 + 2.0 * c2 + d2)
            s3 = s3 + h6 * (a3 + 2.0 * b3 + 2.0 * c3 + d3)
            s4 = s4 + h6 * (a4 + 2.0 * b4 + 2.0 * c4 + d4)
            s5 = s5 + h6 * (a5 + 2.0 * b5 + 2.0 * c5 + d5)
            done += h
            if s0 * s0 + s1 * s1 + s2 * s2 < capture_r * capture_r:
                alive[i] = False
                captured_at[i] = t0 + done
                break
        x[i, 0], x[i, 1], x[i, 2] = s0, s1, s2
        w[i, 0], w[i, 1], w[i, 2] = s3, s4, s5


@numba.njit(cache=True)
def _apply_kick(x, w, acc, alive, dt, dtau):
    """w += acc dtau for the live bodies, dtau = dt (1 - 2/r) / E their proper time (kept in dtau)."""
    for i in range(len(x)):
        dtau[i] = 0.0
        if not alive[i]:
            continue
        r = np.sqrt(x[i, 0] ** 2 + x[i, 1] ** 2 + x[i, 2] ** 2)
        vr = (x[i, 0] * w[i, 0] + x[i, 1] * w[i, 1] + x[i, 2] * w[i, 2]) / r
        w2 = w[i, 0] ** 2 + w[i, 1] ** 2 + w[i, 2] ** 2
        e = np.sqrt(max(vr * vr + (1.0 - 2.0 / r) * (1.0 + w2 - vr * vr), 0.0))
        dtau[i] = dt * ((1.0 - 2.0 / r) / e)
        w[i, 0] += dtau[i] * acc[i, 0]
        w[i, 1] += dtau[i] * acc[i, 1]
        w[i, 2] += dtau[i] * acc[i, 2]


@numba.njit(cache=True)
def _state_at(hx, hw, k_last, spacing, now, sx, sw, which, t, geodesic, x, w, known):
    """History.state_at for every row: body which[i] at time t[i] (see there).

    hx, hw: the ring buffer; sx, sw: the bodies now, at time `now`. geodesic:
    the stored w is dx/dtau and dx/dt = w (1 - 2/r) / E (a BodySystem);
    otherwise w is dx/dt itself.
    """
    length = hx.shape[0]
    p0 = np.empty(3)
    p1 = np.empty(3)
    q0 = np.empty(3)
    q1 = np.empty(3)
    for i in range(len(which)):
        j = which[i]
        ti = min(t[i], now)
        k0 = int(np.floor(ti / spacing))
        ok = k0 > k_last - length
        if not ok:
            k0 = k_last
        slot = k0 % length
        t0 = k0 * spacing
        newest = k0 >= k_last
        slot1 = slot
        t1 = now
        if not newest:
            t1 = (k0 + 1) * spacing
            slot1 = min(k0 + 1, k_last) % length
        for c in range(3):
            p0[c] = hx[slot, j, c]
            q0[c] = hw[slot, j, c]
            if newest:
                p1[c] = sx[j, c]
                q1[c] = sw[j, c]
            else:
                p1[c] = hx[slot1, j, c]
                q1[c] = hw[slot1, j, c]
        span = max(t1 - t0, 1e-12)
        f = min(max((ti - t0) / span, 0.0), 1.0)
        g0 = span
        g1 = span
        if geodesic:
            r = np.sqrt(p0[0] ** 2 + p0[1] ** 2 + p0[2] ** 2)
            vr = (p0[0] * q0[0] + p0[1] * q0[1] + p0[2] * q0[2]) / r
            w2 = q0[0] ** 2 + q0[1] ** 2 + q0[2] ** 2
            e = np.sqrt(max(vr * vr + (1.0 - 2.0 / r) * (1.0 + w2 - vr * vr), 0.0))
            g0 = span * ((1.0 - 2.0 / r) / max(e, 1e-300))
            r = np.sqrt(p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2)
            vr = (p1[0] * q1[0] + p1[1] * q1[1] + p1[2] * q1[2]) / r
            w2 = q1[0] ** 2 + q1[1] ** 2 + q1[2] ** 2
            e = np.sqrt(max(vr * vr + (1.0 - 2.0 / r) * (1.0 + w2 - vr * vr), 0.0))
            g1 = span * ((1.0 - 2.0 / r) / max(e, 1e-300))
        f2 = f * f
        f3 = f2 * f
        for c in range(3):
            x[i, c] = (
                (2 * f3 - 3 * f2 + 1) * p0[c]
                + (f3 - 2 * f2 + f) * (q0[c] * g0)
                + (-2 * f3 + 3 * f2) * p1[c]
                + (f3 - f2) * (q1[c] * g1)
            )
            w[i, c] = (1 - f) * q0[c] + f * q1[c]
            ok = ok and np.isfinite(x[i, c])
        known[i] = ok


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
    # A remainder shorter than this fraction of OUTER_STEP is folded into the
    # step before it (which is then up to 5% longer than OUTER_STEP) rather
    # than taken as a step of its own: each step costs a full evaluation of
    # the mutual pull (and the gas forces), however short it is. Measured
    # 2026-11-23: in "Star torn apart, with gas" a frame was 1.006 OUTER_STEP
    # long, so every frame paid for a second evaluation over 0.6% of it.
    STEP_SLACK = 0.05

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
        # An outer step carried across calls (advance(carry=True)): its
        # length, what is left of it, and whether it opened with a kick.
        self._open_h = 0.0
        self._open_left = 0.0
        self._open_kicked = False
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
            return False
        if not self._acc_fresh:
            v = self.w
            if self.hydro is not None:
                # The gas forces depend on the velocities (viscosity, heating,
                # Balsara factors), and after a drift w is half a kick behind
                # the positions. They are evaluated with w predicted to the end
                # of the kick with the previous accelerations, as SPH codes do
                # (e.g. GADGET-2, Springel 2005, MNRAS 364, 1105); otherwise
                # the scheme is only first order in the gas forces (measured
                # 2026-11-23 on the torn star: half the error of the scheme
                # without it at the same step, with half the evaluations).
                v = self.w.copy()
                _apply_kick(self.x, v, self._acc, self.alive, dt, np.empty(len(self.x)))
            newtonian_mutual(self.x, self.mass, self.alive, self.eps, self._acc)
            if self.hydro is not None:
                self._acc += self.hydro.accelerations(self.x, v, self.mass, self.alive)
            self._acc_fresh = True
        live = self.alive
        dtau = np.empty(len(self.x))
        _apply_kick(self.x, self.w, self._acc, live, dt, dtau)
        if self.hydro is not None:
            self.hydro.heat(dtau[live], live)
        return True

    def advance(self, duration, carry=False):
        """Advance all bodies by `duration` of coordinate time.

        Each outer step: half a kick of mutual gravity, every body along its
        exact geodesic (own RK4 steps), half a kick (Strang splitting: second
        order in the mutual pull, fourth order in the hole's).

        carry=False: the outer steps fit into this call, so at its end the
        velocities belong to the positions. Each call then costs at least one
        evaluation of the mutual pull, however short it is.

        carry=True (the module's frames): an outer step is OUTER_STEP long
        whatever the length of the call, and one that is not finished when
        the call ends is carried on by the next (see _advance_carried). The
        scheme is the same; only the places where it is interrupted differ.
        """
        if carry:
            self._advance_carried(duration)
            return
        self._close_open_step()
        end = self.t + duration
        while end - self.t > 1e-12:
            h = min(self.OUTER_STEP, end - self.t)
            if end - self.t - h < self.STEP_SLACK * self.OUTER_STEP:  # no sliver left over
                h = end - self.t
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

    def _close_open_step(self):
        """End an outer step left open by advance(carry=True) where it stands.

        The step opened with half a kick for its full length but has drifted
        only part of it: the closing kick makes the two add up to the time
        drifted (first order for this one step; it happens only when a
        carried advance is followed by one that is not).
        """
        if self._open_left > 0.0:
            if self._open_kicked:
                self._kick(0.5 * self._open_h - self._open_left)
            self._open_left = 0.0

    def _advance_carried(self, duration):
        """advance() with outer steps that run on across calls.

        The same kick-drift-kick steps, each OUTER_STEP long (as set when the
        step opens). The drift of a step is exact along the geodesics, so it
        may be cut anywhere: at the end of a call and at the history's sample
        times, without evaluating the mutual pull again. A call shorter than
        OUTER_STEP therefore costs a drift only. Between the opening and the
        closing kick of a step, w is half a kick behind the positions (what
        the display and the history then read); the positions are those of
        the scheme. A step that opened while no body had mass (the star
        still whole) closes without a kick as well.
        """
        end = self.t + duration
        while end - self.t > 1e-12:
            if self._open_left <= 0.0:
                self._open_h = self._open_left = float(self.OUTER_STEP)
                self._open_kicked = self._kick(0.5 * self._open_h)
            d = min(self._open_left, end - self.t)
            if self.history is not None:
                d = min(d, self.history.next_time() - self.t)
            _drift(
                self.x,
                self.w,
                self.alive,
                d,
                self.ETA,
                self.MAX_STEP,
                2.0 * (1.0 + CAPTURE_MARGIN),
                self.t,
                self.captured_at,
            )
            self._acc_fresh = False
            self.t += d
            self._open_left -= d
            if self._open_left <= 1e-12 * self._open_h:
                if self._open_kicked:
                    self._kick(0.5 * self._open_h)
                self._open_left = 0.0
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
        own = type(system).dx_dt
        if own is BodySystem.dx_dt or getattr(system, "W_IS_VELOCITY", False):
            # The same as below, in one compiled loop (no temporary arrays).
            which = np.ascontiguousarray(np.broadcast_to(which, np.shape(t)), dtype=np.int64)
            x = np.empty((len(which), 3))
            w = np.empty((len(which), 3))
            known = np.empty(len(which), dtype=np.bool_)
            _state_at(
                self.x,
                self.w,
                self.k_last,
                self.spacing,
                float(system.t),
                np.ascontiguousarray(system.x, dtype=float),
                np.ascontiguousarray(system.w, dtype=float),
                which,
                np.ascontiguousarray(t, dtype=float),
                own is BodySystem.dx_dt,
                x,
                w,
                known,
            )
            return x, w, known
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
