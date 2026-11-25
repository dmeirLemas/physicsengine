"""Bodies around a rotating (Kerr) black hole: geodesics and lensed images.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). Review the physics; the equations are in kerr.py.

Motion. Each body has Boyer-Lindquist (r, theta, phi), the Mino-time
velocities rdot = dr/dlambda, thetadot = dtheta/dlambda, and its constants
E, L, Q (mass 1). It is advanced in coordinate time t (shared by all bodies)
with RK4 on d/dt = (dt/dlambda)^-1 d/dlambda and its own steps; after every
step rdot and thetadot are put back onto rdot^2 = R(r), thetadot^2 =
Theta(theta) (signs kept), so the constants hold exactly. Bodies closer than
the horizon r_+ (1 + CAPTURE_MARGIN) are removed (frozen there in t anyway).

Setting a body up: from its world position and its velocity v measured by
the zero-angular-momentum observer (ZAMO) there, gamma = 1 / sqrt(1 - v^2):
    u_r = e^mu1 gamma v_r, u_theta = e^mu2 gamma v_theta,
    L = u_phi = e^psi gamma v_phi, E = e^nu gamma + omega L,
    Q = u_theta^2 + cos^2(theta) (a^2 (1 - E^2) + L^2 / sin^2(theta)),
    rdot = Delta u_r, thetadot = u_theta.
At a = 0 the ZAMO is the static observer, so the Schwarzschild scenes carry
over through their local velocities.

Mutual pulls (a star's self-gravity): Newtonian accelerations, added to each
body's local velocity over its proper time dtau = dt / u^t, then E, L, Q are
recomputed. As in geodesics.py, valid for bodies much lighter than the hole.

Images. A photon from the body reaches the camera (a ZAMO) along a pixel
direction d. For each body and image (direct, and secondary round the other
side), Newton's method on d: follow the photon back from the camera
(kerr.trace_photon's equations) and take the point of its path closest to
the body; the residual is that offset, across the path (2 numbers). The start
is the Schwarzschild image (blackhole_images.py) or the last frame's answer.
At the solution:
- the Jacobian of the residual over the two screen angles gives the image's
  solid angle per unit cross-section of the body, 1 / |det J|;
- the light-travel time is the coordinate time along the path;
- the redshift g = nu_obs / nu_emit = (1 / E0) / (-p.u) with the photon's
  p_t = -1, p_phi = L, p_r = rdot / Delta, p_theta = thetadot and the
  body's four-velocity u.
"""

import numba
import numpy as np

from physengine.modules import kerr

CAPTURE_MARGIN = 1e-3


# ------------------------------------------------------------- set-up


def _metric_v(r, th, a):
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


def _bl_v(p, a):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rho2 = x * x + y * y + z * z
    r = np.sqrt(0.5 * (rho2 - a * a + np.sqrt((rho2 - a * a) ** 2 + 4.0 * a * a * y * y)))
    return r, np.arccos(np.clip(y / r, -1.0, 1.0)), np.arctan2(-z, x)


def _axes_v(r, th, ph, a):
    R = np.sqrt(r * r + a * a)
    s, c, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    er = np.stack([r / R * s * cp, c, -r / R * s * sp], 1)
    et = np.stack([R * c * cp, -r * s, -R * c * sp], 1)
    ep = np.stack([-sp, np.zeros_like(sp), -cp], 1)
    er /= np.linalg.norm(er, axis=1, keepdims=True)
    et /= np.linalg.norm(et, axis=1, keepdims=True)
    return er, et, ep


def states_from_local(pos, v_local, a):
    """Vectorised state_from_local: (n, 5) states and (n, 3) constants (E, L, Q)."""
    pos = np.atleast_2d(np.asarray(pos, dtype=float))
    v = np.atleast_2d(np.asarray(v_local, dtype=float))
    r, th, ph = _bl_v(pos, a)
    er, et, ep = _axes_v(r, th, ph, a)
    vr, vt, vp = np.sum(v * er, 1), np.sum(v * et, 1), np.sum(v * ep, 1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - (vr * vr + vt * vt + vp * vp), 1e-12))
    sig, dl, A, om, enu, epsi, emu1, emu2 = _metric_v(r, th, a)
    u_r = emu1 * g * vr
    u_t = emu2 * g * vt
    L = epsi * g * vp
    E = enu * g + om * L
    s, c = np.sin(th), np.cos(th)
    Q = u_t**2 + c * c * (a * a * (1.0 - E * E) + L * L / np.maximum(s * s, 1e-300))
    return np.stack([r, th, ph, dl * u_r, u_t], 1), np.stack([E, L, Q], 1)


def state_from_local(pos, v_local, a):
    """(r, theta, phi, rdot, thetadot), (E, L, Q) of a body at world `pos` with ZAMO-frame velocity v_local (world axes)."""
    st, c = states_from_local(np.reshape(pos, (1, 3)), np.reshape(v_local, (1, 3)), a)
    return st[0], c[0]


def local_velocities(st, consts, a):
    """ZAMO-frame velocities (world axes) of bodies with states st, constants consts."""
    r, th, ph, rd, td = st.T
    E, L = consts[:, 0], consts[:, 1]
    sig, dl, A, om, enu, epsi, emu1, emu2 = _metric_v(r, th, a)
    gamma = (E - om * L) / enu
    er, et, ep = _axes_v(r, th, ph, a)
    return (
        ((rd / dl / emu1) / gamma)[:, None] * er
        + ((td / emu2) / gamma)[:, None] * et
        + ((L / epsi) / gamma)[:, None] * ep
    )


def local_velocity_from_schwarzschild(x, w):
    """Static-frame velocity of a body given Schwarzschild x and w = dx/dtau (geodesics.py)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    r = np.linalg.norm(x)
    n = x / r
    lapse = np.sqrt(1.0 - 2.0 / r)
    vr = n @ w
    w_perp = w - vr * n
    E = np.sqrt(vr * vr + lapse**2 * (1.0 + w @ w - vr * vr))
    gamma = E / lapse
    return (n * vr / lapse + w_perp) / gamma


def kerr_circular_state(r, a, inclination=0.0, nudge=0.0):
    """A body on the prograde circular equatorial orbit at r (nudged inward by L (1 - nudge))."""
    E, L, Om = kerr.circular_orbit(r, a)
    th = np.pi / 2
    state = np.array([r, th, 0.0, 0.0, 0.0])
    L = L * (1.0 - nudge)
    if inclination:
        # Same speed, orbit tilted: the ZAMO-frame velocity turned out of the plane.
        sig, dl, A, om, enu, epsi, emu1, emu2 = kerr.metric(r, th, a)
        gv = L / epsi
        vloc = gv / np.sqrt(1.0 + gv * gv)
        pos = kerr.to_world(r, th, 0.0, a)
        er, et, ep = kerr.local_axes(r, th, 0.0, a)
        v = vloc * (np.cos(inclination) * ep - np.sin(inclination) * et)
        return state_from_local(pos, v, a)
    return state, np.array([E, L, 0.0])


def kerr_capture_L(a, inclination_deg, r0, iterations=30):
    """The angular-momentum parameter of blackhole_scenes.infalling (E = 1 from r0) at which
    the body just gets captured by a Kerr hole of spin a, found by bisection.

    (For a = 0 this is L = 4: Chandrasekhar 1983, sec. 19.) The scene's body
    is set up in Schwarzschild terms and carried over by its local velocity,
    so the threshold is found for that same set-up.
    """
    from physengine.modules.blackhole_scenes import infalling

    def captured(L):
        spec = infalling("probe", r0, L, inclination_deg=inclination_deg)
        st, c = states_from_local(
            spec.x[None], local_velocity_from_schwarzschild(spec.x, spec.w)[None], a
        )
        system = KerrBodySystem(st, c, a)
        system.advance(40.0 * r0)  # long enough to pass pericentre or fall in
        return not system.alive[0]

    lo, hi = 0.5, 6.0  # captured at lo, escapes at hi
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if captured(mid):
            lo = mid
        else:
            hi = mid
    return lo


# ------------------------------------------------------------ dynamics


@numba.njit(cache=True)
def _rates(y, E, L, Q, a):
    """d/dlambda of (r, theta, phi, rdot, thetadot) and dt/dlambda for a body (mu = 1)."""
    r, th = y[0], y[1]
    s = np.sin(th)
    c = np.cos(th)
    s2 = max(s * s, 1e-24)
    dl = r * r - 2.0 * r + a * a
    P = E * (r * r + a * a) - a * L
    K = (L - a * E) ** 2 + Q
    dR = 4.0 * E * r * P - (2.0 * r - 2.0) * (r * r + K) - dl * 2.0 * r
    out = np.empty(5)
    out[0] = y[3]
    out[1] = y[4]
    out[2] = -(a * E - L / s2) + a * P / dl
    out[3] = 0.5 * dR
    out[4] = s * c * a * a * (1.0 - E * E) + L * L * c / (s2 * s)
    tdot = -a * (a * E * s2 - L) + (r * r + a * a) * P / dl
    return out, tdot


@numba.njit(cache=True)
def _project(y, E, L, Q, a):
    """rdot, thetadot back onto R(r) and Theta(theta), keeping their signs."""
    r, th = y[0], y[1]
    dl = r * r - 2.0 * r + a * a
    P = E * (r * r + a * a) - a * L
    R = P * P - dl * (r * r + (L - a * E) ** 2 + Q)
    s, c = np.sin(th), np.cos(th)
    Th = Q - c * c * (a * a * (1.0 - E * E) + L * L / max(s * s, 1e-24))
    y[3] = np.sqrt(max(R, 0.0)) * (1.0 if y[3] >= 0.0 else -1.0)
    y[4] = np.sqrt(max(Th, 0.0)) * (1.0 if y[4] >= 0.0 else -1.0)


@numba.njit(parallel=True, cache=True)
def _drift(st, consts, alive, duration, a, eta, max_step, r_capture, t0, captured_at):
    for i in numba.prange(len(st)):
        if not alive[i]:
            continue
        E, L, Q = consts[i, 0], consts[i, 1], consts[i, 2]
        y = st[i].copy()
        done = 0.0
        while duration - done > 1e-12:
            r = y[0]
            h = min(eta * r * np.sqrt(r), max_step, duration - done)
            k1, t1 = _rates(y, E, L, Q, a)
            k1 /= t1
            k2, t2 = _rates(y + 0.5 * h * k1, E, L, Q, a)
            k2 /= t2
            k3, t3 = _rates(y + 0.5 * h * k2, E, L, Q, a)
            k3 /= t3
            k4, t4 = _rates(y + h * k3, E, L, Q, a)
            k4 /= t4
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            _project(y, E, L, Q, a)
            done += h
            if y[0] < r_capture:
                alive[i] = False
                captured_at[i] = t0 + done
                break
        st[i] = y


@numba.njit(cache=True)
def world_velocity(st, consts, a):
    """dx/dt in world coordinates of each body."""
    out = np.zeros((len(st), 3))
    for i in range(len(st)):
        rates, tdot = _rates(st[i], consts[i, 0], consts[i, 1], consts[i, 2], a)
        r, th, ph = st[i, 0], st[i, 1], st[i, 2]
        rr, tt, pp = rates[0] / tdot, rates[1] / tdot, rates[2] / tdot
        R = np.sqrt(r * r + a * a)
        s, c, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        out[i, 0] = (r / R) * s * cp * rr + R * c * cp * tt - R * s * sp * pp
        out[i, 1] = c * rr - r * s * tt
        out[i, 2] = -(r / R) * s * sp * rr - R * c * sp * tt - R * s * cp * pp
    return out


def four_velocity_from_world(pos, vel, a):
    """(u^t, u^r, u^theta, u^phi) of a body at world pos moving with world dx/dt = vel."""
    r, th, ph = kerr.from_world(pos, a)
    R = np.sqrt(r * r + a * a)
    s, c, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    # Jacobian d(x, y, z)/d(r, theta, phi), solved for the coordinate rates.
    J = np.array(
        [
            [(r / R) * s * cp, R * c * cp, -R * s * sp],
            [c, -r * s, 0.0],
            [-(r / R) * s * sp, -R * c * sp, -R * s * cp],
        ]
    )
    rr, tt, pp = np.linalg.solve(J, vel)
    sig = r * r + a * a * c * c
    dl = r * r - 2.0 * r + a * a
    g_tt = -(1.0 - 2.0 * r / sig)
    g_tp = -2.0 * a * r * s * s / sig
    g_pp = ((r * r + a * a) ** 2 - a * a * dl * s * s) / sig * s * s
    norm = g_tt + 2.0 * g_tp * pp + g_pp * pp * pp + sig / dl * rr * rr + sig * tt * tt
    ut = 1.0 / np.sqrt(max(-norm, 1e-300))
    return ut, ut * rr, ut * tt, ut * pp


class KerrBodySystem:
    """Bodies on Kerr geodesics; same interface as geodesics.BodySystem (x, w, alive, t, advance...).

    Here w is the world coordinate velocity dx/dt (what the History stores).
    """

    ETA = 0.01
    MAX_STEP = 2.0
    OUTER_STEP = 1.0

    def __init__(self, states, consts, a, mass=None, eps=0.05):
        self.a = float(a)
        self.st = np.array(states, dtype=float).reshape(-1, 5)
        self.consts = np.array(consts, dtype=float).reshape(-1, 3)
        n = len(self.st)
        self.mass = np.zeros(n) if mass is None else np.array(mass, dtype=float)
        self.alive = np.ones(n, dtype=bool)
        self.captured_at = np.full(n, np.nan)
        self.eps = eps
        self.t = 0.0
        self.history = None
        self.hydro = None  # a blackhole_sph.Gas, if the bodies are gas
        # An outer step carried across calls (advance(carry=True)): its
        # length, what is left of it, whether it opened with a kick, and a
        # closing half kick still to be applied with the next opening one.
        self._open_h = 0.0
        self._open_left = 0.0
        self._open_kicked = False
        self._owed = 0.0
        self.r_capture = kerr.horizon(self.a) * (1.0 + CAPTURE_MARGIN)

    @property
    def n(self):
        return len(self.st)

    @property
    def x(self):
        r, th, ph = self.st[:, 0], self.st[:, 1], self.st[:, 2]
        R = np.sqrt(r * r + self.a * self.a)
        return np.stack(
            [R * np.sin(th) * np.cos(ph), r * np.cos(th), -R * np.sin(th) * np.sin(ph)], 1
        )

    @property
    def w(self):
        return world_velocity(self.st, self.consts, self.a)

    def r(self):
        return self.st[:, 0].copy()

    def dx_dt(self, x=None, w=None):
        return self.w if w is None else w

    def add_state(self, state, const, mass=0.0):
        self.st = np.vstack([self.st, state])
        self.consts = np.vstack([self.consts, const])
        self.mass = np.append(self.mass, mass)
        self.alive = np.append(self.alive, True)
        self.captured_at = np.append(self.captured_at, np.nan)
        if self.history is not None:
            self.history.add_body(self, self.n - 1)

    def _kick(self, dt):
        """Mutual Newtonian pull over coordinate time dt, applied in each body's ZAMO frame.

        The local velocity changes by a dtau, dtau = dt / u^t with u^t = gamma / e^nu
        for a body moving with Lorentz factor gamma relative to the ZAMO.
        """
        live = self.alive & (self.mass > 0.0)
        if live.sum() < 2:
            return False
        from physengine.modules.geodesics import newtonian_mutual

        x = self.x
        acc = np.zeros_like(x)
        newtonian_mutual(x, self.mass, self.alive, self.eps, acc)
        idx = np.flatnonzero(self.alive)
        st, c = self.st[idx], self.consts[idx]
        r, th = st[:, 0], st[:, 1]
        _, _, _, om, enu, _, _, _ = _metric_v(r, th, self.a)
        gamma = (c[:, 0] - om * c[:, 1]) / enu
        dtau = dt * enu / gamma
        v = local_velocities(st, c, self.a)
        if self.hydro is not None:
            v_all = np.zeros_like(x)
            v_all[idx] = v
            acc = acc + self.hydro.accelerations(x, v_all, self.mass, self.alive)
            self.hydro.heat(dtau, idx)
        v = v + acc[idx] * dtau[:, None]
        self.st[idx], self.consts[idx] = states_from_local(x[idx], v, self.a)
        return True

    def local_velocity(self, i):
        """ZAMO-frame velocity of body i (world axes)."""
        return local_velocities(self.st[i : i + 1], self.consts[i : i + 1], self.a)[0]

    def past(self, which, spacing, count):
        """(x, w) of bodies `which` at t - spacing, ... (count samples).

        Kerr is symmetric under (t, phi) -> (-t, -phi): the past of a body is
        the future of its mirror image with phi, rdot and thetadot negated.
        A body whose past would come out of the hole is held at its present
        state.
        """
        mirror = self.st[which].copy()
        mirror[:, 2] *= -1.0
        mirror[:, 3:] *= -1.0
        back = KerrBodySystem(mirror, self.consts[which].copy(), self.a, self.mass[which], self.eps)
        back.OUTER_STEP = max(self.OUTER_STEP, min(spacing, 4.0))
        x_now, w_now = self.x[which], self.w[which]
        held = ~back.alive
        for _ in range(count):
            back.advance(spacing)
            held |= ~back.alive
            real = back.st.copy()
            real[:, 2] *= -1.0
            real[:, 3:] *= -1.0
            R = np.sqrt(real[:, 0] ** 2 + self.a**2)
            x = np.stack(
                [
                    R * np.sin(real[:, 1]) * np.cos(real[:, 2]),
                    real[:, 0] * np.cos(real[:, 1]),
                    -R * np.sin(real[:, 1]) * np.sin(real[:, 2]),
                ],
                1,
            )
            w = world_velocity(real, back.consts, self.a)
            yield np.where(held[:, None], x_now, x), np.where(held[:, None], w_now, w)

    def _close_open_step(self):
        """End an outer step left open by advance(carry=True) where it stands (see BodySystem)."""
        if self._open_left > 0.0:
            if self._open_kicked:
                self._kick(0.5 * self._open_h - self._open_left)
            self._open_left = 0.0

    def _advance_carried(self, duration):
        """advance() with outer steps that run on across calls (see BodySystem._advance_carried).

        The two half kicks that meet between one step and the next are
        applied as one kick: the positions are the same for both.
        """
        end = self.t + duration
        while end - self.t > 1e-12:
            if self._open_left <= 0.0:
                h = float(self.OUTER_STEP)
                # (the closing half kick of the step before, if it is still owed)
                self._open_kicked = self._kick(0.5 * h + self._owed)
                self._owed = 0.0
                self._open_h = self._open_left = h
            d = min(self._open_left, end - self.t)
            if self.history is not None:
                d = min(d, self.history.next_time() - self.t)
            _drift(
                self.st,
                self.consts,
                self.alive,
                d,
                self.a,
                self.ETA,
                self.MAX_STEP,
                self.r_capture,
                self.t,
                self.captured_at,
            )
            self.t += d
            self._open_left -= d
            if self._open_left <= 1e-12 * self._open_h:
                self._open_left = 0.0
                if self._open_kicked:
                    if end - self.t > 1e-12:
                        self._owed = 0.5 * self._open_h  # with the next step's opening kick
                    else:
                        self._kick(0.5 * self._open_h)
            if self.history is not None and self.t >= self.history.next_time() - 1e-9:
                self.history.record(self)

    def advance(self, duration, carry=False):
        """Advance by `duration`; carry=True as in geodesics.BodySystem.advance."""
        if carry:
            self._advance_carried(duration)
            return
        self._close_open_step()
        end = self.t + duration
        while end - self.t > 1e-12:
            h = min(self.OUTER_STEP, end - self.t)
            if self.history is not None:
                h = min(h, self.history.next_time() - self.t)
            self._kick(0.5 * h)
            _drift(
                self.st,
                self.consts,
                self.alive,
                h,
                self.a,
                self.ETA,
                self.MAX_STEP,
                self.r_capture,
                self.t,
                self.captured_at,
            )
            self._kick(0.5 * h)
            self.t += h
            if self.history is not None and self.t >= self.history.next_time() - 1e-9:
                self.history.record(self)


# --------------------------------------------------------------- images


@numba.njit(cache=True)
def _to_world(r, th, ph, a):
    R = np.sqrt(r * r + a * a)
    s = np.sin(th)
    return np.array([R * s * np.cos(ph), r * np.cos(th), -R * s * np.sin(ph)])


@numba.njit(cache=True)
def _photon_rates(y, a, L, Q):
    r, th = y[0], y[1]
    s = np.sin(th)
    c = np.cos(th)
    s2 = max(s * s, 1e-24)
    dl = r * r - 2.0 * r + a * a
    P = r * r + a * a - a * L
    out = np.empty(6)
    out[0] = y[4]
    out[1] = y[5]
    out[2] = -(a - L / s2) + a * P / dl
    out[3] = -a * (a * s2 - L) + (r * r + a * a) * P / dl
    out[4] = 0.5 * (4.0 * r * P - (2.0 * r - 2.0) * ((L - a) ** 2 + Q))
    out[5] = -a * a * s * c + L * L * c / (s2 * s)
    return out


@numba.njit(cache=True)
def _closest(r0, th0, ph0, rd0, td0, L, Q, a, body, r_far, eta, max_steps, r_in, r_out, disc_on):
    """Follow a photon back; closest approach to `body`.

    Returns (distance, offset vector (3), path direction (3), t there, rdot,
    thetadot, r, theta, hidden by the disc before that point).
    """
    y = np.array([r0, th0, ph0, 0.0, rd0, td0])
    r_h = 1.0 + np.sqrt(1.0 - a * a)
    best = 1e300
    best_off = np.zeros(3)
    best_dir = np.zeros(3)
    best_t = 0.0
    best_y = y.copy()
    best_hidden = False
    hidden = False
    prev = _to_world(y[0], y[1], y[2], a)
    for _ in range(max_steps):
        k1 = _photon_rates(y, a, L, Q)
        scale = max(abs(k1[0]) / y[0], abs(k1[1]), abs(k1[2]), 1e-9)
        h = -eta / scale
        k2 = _photon_rates(y + 0.5 * h * k1, a, L, Q)
        k3 = _photon_rates(y + 0.5 * h * k2, a, L, Q)
        k4 = _photon_rates(y + h * k3, a, L, Q)
        yn = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # back onto the first integrals
        dl = yn[0] * yn[0] - 2.0 * yn[0] + a * a
        P = yn[0] * yn[0] + a * a - a * L
        Rv = P * P - dl * ((L - a) ** 2 + Q)
        sn, cn = np.sin(yn[1]), np.cos(yn[1])
        Th = Q + a * a * cn * cn - L * L * cn * cn / max(sn * sn, 1e-24)
        yn[4] = np.sqrt(max(Rv, 0.0)) * (1.0 if yn[4] >= 0.0 else -1.0)
        yn[5] = np.sqrt(max(Th, 0.0)) * (1.0 if yn[5] >= 0.0 else -1.0)
        if disc_on:
            c0 = np.cos(y[1])
            if (c0 > 0.0) != (cn > 0.0):
                f = c0 / (c0 - cn)
                rc = y[0] + f * (yn[0] - y[0])
                if rc >= r_in and rc <= r_out:
                    hidden = True
        cur = _to_world(yn[0], yn[1], yn[2], a)
        seg = cur - prev
        sl2 = seg[0] ** 2 + seg[1] ** 2 + seg[2] ** 2
        if sl2 > 0.0:
            f = (
                (body[0] - prev[0]) * seg[0]
                + (body[1] - prev[1]) * seg[1]
                + (body[2] - prev[2]) * seg[2]
            ) / sl2
            f = min(max(f, 0.0), 1.0)
            q = prev + f * seg
            off = q - body
            d = np.sqrt(off[0] ** 2 + off[1] ** 2 + off[2] ** 2)
            if d < best:
                best = d
                best_off = off
                best_dir = seg / np.sqrt(sl2)
                best_t = y[3] + f * (yn[3] - y[3])
                best_y = y + f * (yn - y)
                best_hidden = hidden
        prev = cur
        y = yn
        if y[0] <= r_h * 1.0001:
            break
        if y[0] >= r_far and y[4] < 0.0:
            break
    return best, best_off, best_dir, best_t, best_y[4], best_y[5], best_y[0], best_y[1], best_hidden


@numba.njit(cache=True)
def _camera_photon(cam_r, cam_th, cam_ph, er, et, ep, d, a):
    """(rdot, thetadot, L, Q, E0) of the photon a ZAMO camera sees along d (it arrives along -d)."""
    s, c = np.sin(cam_th), np.cos(cam_th)
    r = cam_r
    sig = r * r + a * a * c * c
    dl = r * r - 2.0 * r + a * a
    A = (r * r + a * a) ** 2 - a * a * dl * s * s
    om = 2.0 * a * r / A
    enu = np.sqrt(sig * dl / A)
    epsi = np.sqrt(A / sig) * s
    nr = -(d[0] * er[0] + d[1] * er[1] + d[2] * er[2])
    nt = -(d[0] * et[0] + d[1] * et[1] + d[2] * et[2])
    np_ = -(d[0] * ep[0] + d[1] * ep[1] + d[2] * ep[2])
    L0 = epsi * np_
    E0 = enu + om * L0
    L = L0 / E0
    pr = np.sqrt(sig / dl) * nr / E0
    pt = np.sqrt(sig) * nt / E0
    Q = pt * pt + c * c * (L * L / max(s * s, 1e-24) - a * a)
    return dl * pr, pt, L, Q, E0


@numba.njit(cache=True)
def _residual(cam, er, et, ep, d, a, body, r_far, eta, max_steps, r_in, r_out, disc_on, b1, b2):
    rd0, td0, L, Q, E0 = _camera_photon(cam[0], cam[1], cam[2], er, et, ep, d, a)
    res = _closest(
        cam[0], cam[1], cam[2], rd0, td0, L, Q, a, body, r_far, eta, max_steps, r_in, r_out, disc_on
    )
    off = res[1]
    return (
        np.array(
            [
                off[0] * b1[0] + off[1] * b1[1] + off[2] * b1[2],
                off[0] * b2[0] + off[1] * b2[1] + off[2] * b2[2],
            ]
        ),
        res,
        L,
        Q,
        E0,
    )


@numba.njit(parallel=True, cache=True)
def _solve_images(
    cam,
    er,
    et,
    ep,
    a,
    bodies,
    guesses,
    r_far,
    eta,
    max_steps,
    r_in,
    r_out,
    disc_on,
    iters,
    out_dir,
    out_info,
):
    """Newton on each guessed direction. out_info: visible, solid per area, delay, E0, L, rdot, thetadot, r, theta."""
    n = len(guesses)
    for k in numba.prange(n):
        out_info[k, 0] = 0.0
        d = guesses[k].copy()
        nd = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if nd == 0.0:
            continue
        d /= nd
        body = bodies[k]
        # Screen basis around d.
        up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        s1 = np.cross(d, up)
        s1 /= np.sqrt(s1[0] ** 2 + s1[1] ** 2 + s1[2] ** 2)
        s2 = np.cross(d, s1)
        ok = False
        J = np.zeros((2, 2))
        t_hit = 0.0
        rd_hit = 0.0
        td_hit = 0.0
        r_hit = 0.0
        th_hit = 0.0
        hidden = False
        for it in range(iters + 1):
            # Path direction near the body sets the residual basis.
            rd0, td0, L, Q, E0 = _camera_photon(cam[0], cam[1], cam[2], er, et, ep, d, a)
            base = _closest(
                cam[0],
                cam[1],
                cam[2],
                rd0,
                td0,
                L,
                Q,
                a,
                body,
                r_far,
                eta,
                max_steps,
                r_in,
                r_out,
                disc_on,
            )
            t_dir = base[2]
            b1 = np.cross(t_dir, up)
            nb = np.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
            if nb < 1e-9:
                b1 = np.cross(t_dir, np.array([0.0, 0.0, 1.0]))
                nb = np.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
            b1 /= nb
            b2 = np.cross(t_dir, b1)
            o = base[1]
            f0 = np.array(
                [
                    o[0] * b1[0] + o[1] * b1[1] + o[2] * b1[2],
                    o[0] * b2[0] + o[1] * b2[1] + o[2] * b2[2],
                ]
            )
            eps = 1e-5
            fa = _residual(
                cam,
                er,
                et,
                ep,
                d + eps * s1,
                a,
                body,
                r_far,
                eta,
                max_steps,
                r_in,
                r_out,
                disc_on,
                b1,
                b2,
            )[0]
            fb = _residual(
                cam,
                er,
                et,
                ep,
                d + eps * s2,
                a,
                body,
                r_far,
                eta,
                max_steps,
                r_in,
                r_out,
                disc_on,
                b1,
                b2,
            )[0]
            J[0, 0] = (fa[0] - f0[0]) / eps
            J[1, 0] = (fa[1] - f0[1]) / eps
            J[0, 1] = (fb[0] - f0[0]) / eps
            J[1, 1] = (fb[1] - f0[1]) / eps
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            t_hit, rd_hit, td_hit, r_hit, th_hit, hidden = (
                base[3],
                base[4],
                base[5],
                base[6],
                base[7],
                base[8],
            )
            if base[0] < 1e-4 * max(1.0, body[0] ** 2 + body[1] ** 2 + body[2] ** 2) ** 0.5:
                ok = True
                break
            if it == iters or abs(det) < 1e-30:
                break
            dx = (J[1, 1] * f0[0] - J[0, 1] * f0[1]) / det
            dy = (-J[1, 0] * f0[0] + J[0, 0] * f0[1]) / det
            step = np.sqrt(dx * dx + dy * dy)
            if step > 0.1:  # damped: stay in the basin
                dx *= 0.1 / step
                dy *= 0.1 / step
            d = d - dx * s1 - dy * s2
            d /= np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            s1 = np.cross(d, up)
            s1 /= np.sqrt(s1[0] ** 2 + s1[1] ** 2 + s1[2] ** 2)
            s2 = np.cross(d, s1)
        if not ok or hidden:
            out_dir[k] = d
            continue
        rd0, td0, L, Q, E0 = _camera_photon(cam[0], cam[1], cam[2], er, et, ep, d, a)
        out_dir[k] = d
        out_info[k, 0] = 1.0
        out_info[k, 1] = 1.0 / max(abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]), 1e-30)
        out_info[k, 2] = -t_hit
        out_info[k, 3] = E0
        out_info[k, 4] = L
        out_info[k, 5] = rd_hit
        out_info[k, 6] = td_hit
        out_info[k, 7] = r_hit
        out_info[k, 8] = th_hit


def kerr_images(cam_world, bodies, vel, guesses, a, r_in, r_out, disc_on, r_far=None, iters=12):
    """Images of bodies (world positions, world velocities dx/dt) seen by a ZAMO camera at cam_world.

    guesses: (n, 3) starting pixel directions, one per wanted image (pass the
    same body twice for its two images). Returns (directions, visible, solid
    angle per unit cross-section, g, delay).
    """
    bodies = np.ascontiguousarray(bodies, dtype=float).reshape(-1, 3)
    guesses = np.ascontiguousarray(guesses, dtype=float).reshape(-1, 3)
    n = len(bodies)
    out_dir = np.zeros((n, 3))
    info = np.zeros((n, 9))
    if n == 0:
        return out_dir, np.zeros(0, bool), np.zeros(0), np.zeros(0), np.zeros(0)
    r, th, ph = kerr.from_world(np.asarray(cam_world, dtype=float), a)
    er, et, ep = kerr.local_axes(r, th, ph, a)
    r_far = max(2.0 * r, 300.0) if r_far is None else r_far
    _solve_images(
        np.array([r, th, ph]),
        er,
        et,
        ep,
        float(a),
        bodies,
        guesses,
        r_far,
        0.01,
        20000,
        float(r_in),
        float(r_out),
        bool(disc_on),
        iters,
        out_dir,
        info,
    )
    vis = info[:, 0] > 0.5
    g = np.zeros(n)
    for k in np.flatnonzero(vis):
        E0, L, rd, td, rb = info[k, 3], info[k, 4], info[k, 5], info[k, 6], info[k, 7]
        ut, ur, uth, uph = four_velocity_from_world(
            bodies[k], np.asarray(vel, dtype=float).reshape(-1, 3)[k], a
        )
        dl = rb * rb - 2.0 * rb + a * a
        minus_pu = ut - L * uph - (rd / dl) * ur - td * uth
        g[k] = (1.0 / E0) / minus_pu if minus_pu > 0 else 0.0
    return out_dir, vis, info[:, 1], g, info[:, 2]
