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
"""

import numba
import numpy as np

GAMMA = 5.0 / 3.0
ETA = 1.2
ALPHA = 1.0
BETA = 2.0
COURANT = 0.3
H_MAX = 8.0  # [M] smoothing length of an isolated particle
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

    def _update_density(self, x, v, m, gas, passes):
        for _ in range(passes):
            density(x, v, m, self.h, self.u, gas, self.rho, self.shock)
            ok = gas & (self.rho > 0.0)
            want = ETA * np.cbrt(m[ok] / self.rho[ok])
            self.h[ok] = np.clip(want, 0.5 * self.h[ok], np.minimum(2.0 * self.h[ok], H_MAX))
        density(x, v, m, self.h, self.u, gas, self.rho, self.shock)

    def accelerations(self, x, v, mass, alive):
        """Pressure and viscosity accelerations of all bodies (zero for those that are not gas)."""
        self.resize(len(x))
        gas = alive & (mass > 0.0)
        if gas.sum() < 2:
            self.acc[:] = 0.0
            self.dudt[:] = 0.0
            self.time_step = np.inf
            return self.acc
        x = np.ascontiguousarray(x)
        v = np.ascontiguousarray(v)
        # h follows the density from one evaluation to the next; the first
        # time it is iterated from the starting guess.
        self._update_density(x, v, mass, gas, 1 if self.started else 6)
        self.started = True
        self.omega = np.sum(x * x, axis=1) ** -0.75
        forces(
            x,
            v,
            mass,
            self.h,
            self.rho,
            self.u,
            gas,
            np.ones_like(self.shock) if self.shear_viscosity else self.shock,
            ALPHA * self.viscosity,
            BETA * self.viscosity,
            self.acc,
            self.dudt,
            self.dt_i,
        )
        self.time_step = float(np.min(self.dt_i[gas]))
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
