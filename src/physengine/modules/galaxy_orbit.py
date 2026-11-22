"""The approach orbit of two extended galaxies.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20). Review
the physics.

Two galaxies that start 100 kpc apart are not point masses: their dark haloes
already overlap. Three ways of setting up the orbit were measured on the
default prograde encounter, pericentre set to 10 kpc, by where the two
bulges (what one sees) actually passed each other:

- Kepler orbit of two point masses: 18.7 kpc. The galaxies attract each
  other less than point masses, but were given the point-mass angular
  momentum.
- Two rigid spheres with their exact interaction energy
  W(d) = integral of rho_1(x) Phi_2(|x - d|) d^3x: 4.5 kpc. This predicts
  the motion of the centres of MASS well (they passed at 9.0 kpc), but the
  outer haloes lag behind, and the visible centres run up to 6 kpc ahead of
  their galaxy's centre of mass.
- Used here: each galaxy's centre falls like a test body in the field of the
  other galaxy, taken as spherical. The relative acceleration of the two
  centres is then
      g(d) = G (M1(<d) + M2(<d)) / d^2,
  only the mass inside the separation pulling (Newton's shell theorem). Far
  apart this is the Kepler problem with the total mass. Measured bulge
  passages: 7.8 kpc for 10 (4.1 for 5, 17.9 for 20, 37.8 for 40, 8.7 for a
  companion of 30% mass), 5% later than predicted. What remains is dynamical
  friction, and the 1-3 kpc by which a bulge wanders around the centre of a
  halo of only 4000 particles.

M(<r) comes from the galaxy model: the truncated Hernquist halo and bulge
exactly, the disc with its cylindrical mass profile used as a spherical one.
That last step underestimates the pull within the disc plane close in (by
30% at 10 kpc for the default model, 3% at 30 kpc; along the disc axis it
overestimates by 9% and 4%), which only affects the last stretch of the
approach.

The motion of the separation vector is that of a body of unit mass in the
potential Psi(d) = -integral_d^inf g: energy E = v^2 / 2 + Psi(r) and angular
momentum L = r v_t per unit mass are conserved. (The classes below use the
general names W, dW and mu, with E = mu v^2 / 2 + W; here mu = 1.)

The orbit is given by its pericentre r_p and an eccentricity e:

- e < 1, bound: e = (r_a - r_p) / (r_a + r_p) with r_a the apocentre, the
  usual definition for orbits in a galaxy potential (Binney & Tremaine 2008,
  sec. 3.1). So r_a = r_p (1 + e) / (1 - e), as for a Kepler ellipse. E and L
  follow from v_r = 0 at both r_p and r_a:
      L^2 = 2 mu (W(r_a) - W(r_p)) / (1 / r_p^2 - 1 / r_a^2).
- e >= 1, unbound: E = (e - 1) / 2 * |W(r_p)|, and L from v_r = 0 at r_p.
  e = 1 is the parabolic orbit (E = 0, at rest at infinity), the limit of the
  bound case as r_a goes to infinity.

For point masses both cases are exactly the Kepler orbit of eccentricity e.

Model units: G = 1, kpc, 1e10 M_sun, 4.715 Myr.
"""

import numpy as np

# Separations [kpc] where g and Psi are tabulated. The galaxy models span
# 0.1 kpc (softening) to a few hundred kpc (halo cut-off).
_D_GRID = np.geomspace(1e-3, 1e5, 20000)


def _hernquist_mass(comp, r):
    """Mass of a truncated Hernquist component inside r, as galacy_ics.ics samples it."""
    u_max = (comp.r_max / (comp.r_max + comp.a)) ** 2
    rc = np.minimum(r, comp.r_max)
    return comp.mass * (rc / (rc + comp.a)) ** 2 / u_max


def _disc_mass(disc, r):
    """Mass of the truncated exponential disc inside cylindrical radius r."""
    x, x_max = np.minimum(r, disc.R_max) / disc.R_d, disc.R_max / disc.R_d
    return disc.mass * (1.0 - (1.0 + x) * np.exp(-x)) / (1.0 - (1.0 + x_max) * np.exp(-x_max))


def enclosed_mass(model, r):
    """Mass of a GalaxyModel inside radius r, the disc counted as if spherical."""
    r = np.asarray(r, dtype=float)
    spheres = _hernquist_mass(model.halo, r) + _hernquist_mass(model.bulge, r)
    return spheres + _disc_mass(model.disc, r)


class PointMasses:
    """Interaction of two point masses: W = -G m1 m2 / r (for tests and comparison)."""

    finite_centre = False  # W(0) is infinite

    def __init__(self, m1, m2, G=1.0):
        self.m1, self.m2, self.G = m1, m2, G
        self.mu = m1 * m2 / (m1 + m2)

    def W(self, r):
        return -self.G * self.m1 * self.m2 / r

    def dW(self, r):
        return self.G * self.m1 * self.m2 / r**2


class GalaxyPair:
    """Relative motion of the centres of two galaxy models (see the module docstring).

    W is Psi, the potential per unit mass of the separation vector; mu = 1.
    """

    finite_centre = True
    mu = 1.0

    def __init__(self, model1, model2, G=1.0):
        d = _D_GRID
        self._d = d
        self._g = G * (enclosed_mass(model1, d) + enclosed_mass(model2, d)) / d**2
        # Psi(d) = -integral_d^inf g: trapezoid on the grid, Kepler beyond it.
        beyond = G * (model1.total_mass() + model2.total_mass()) / d[-1]
        segments = 0.5 * (self._g[1:] + self._g[:-1]) * np.diff(d)
        self._psi = -(np.append(np.cumsum(segments[::-1])[::-1], 0.0) + beyond)

    def W(self, r):
        return np.interp(r, self._d, self._psi)

    def dW(self, r):
        return np.interp(r, self._d, self._g)


# ------------------------------------------------------------------ the orbit


def orbit_apocentre(r_p, e):
    """Largest distance on the orbit: r_p (1 + e) / (1 - e); infinite for e >= 1."""
    return np.inf if e >= 1.0 else r_p * (1.0 + e) / (1.0 - e)


def orbit_constants(pair, r_p, e):
    """(E, L): energy and angular momentum of the orbit with pericentre r_p, eccentricity e."""
    if r_p < 0.0 or e < 0.0:
        raise ValueError("pericentre and eccentricity must be >= 0")
    mu = pair.mu
    if r_p == 0.0:
        # Head-on: no angular momentum, and an "apocentre" of r_p (1 + e) /
        # (1 - e) = 0 means nothing, so only unbound energies are defined.
        if e < 1.0 or not pair.finite_centre:
            raise ValueError("a head-on orbit needs eccentricity >= 1 and extended galaxies")
        return 0.5 * (e - 1.0) * -float(pair.W(1e-3)), 0.0
    W_p = float(pair.W(r_p))
    if e >= 1.0:
        E = 0.5 * (e - 1.0) * -W_p
        return E, np.sqrt(2.0 * mu * (E - W_p)) * r_p
    if e < 1e-6:  # circular: centripetal force = attraction
        L2 = mu * r_p**3 * float(pair.dW(r_p))
    else:
        r_a = orbit_apocentre(r_p, e)
        L2 = 2.0 * mu * (float(pair.W(r_a)) - W_p) / (1.0 / r_p**2 - 1.0 / r_a**2)
    return W_p + L2 / (2.0 * mu * r_p**2), np.sqrt(L2)


def _radial_energy(pair, E, L, r):
    """mu v_r^2 / 2 at distance r: what is left of E after W and the centrifugal term."""
    return E - pair.W(r) - L * L / (2.0 * pair.mu * r * r)


def encounter_orbit(pair, r_p, e, r0, eta=0.01):
    """Incoming relative position and velocity at distance r0, and the time to pericentre.

    Body 2 relative to body 1, in the xy-plane, angular momentum along +z,
    pericentre on the +x axis: the same frame as a Kepler orbit. The orbit is
    integrated from the pericentre outwards (Runge-Kutta 4, step eta r / v)
    until it reaches r0; mirrored in the x-axis and reversed in time, that is
    the incoming state. r0 must lie between r_p and the apocentre.

    Head-on (r_p = 0): body 2 starts on the -y axis and moves along +y. That
    is the limit of small r_p for extended galaxies, whose centres go
    straight through each other a quarter turn before the nominal pericentre
    direction, so the disc angles keep their meaning. (Point masses swing
    half a turn around each other instead; see kepler_orbit.)

    Returns (pos, vel, t_pericentre).
    """
    E, L = orbit_constants(pair, r_p, e)
    mu = pair.mu
    r_a = orbit_apocentre(r_p, e)
    tol = 1e-9
    if r0 < r_p * (1.0 - tol) or r0 > r_a * (1.0 + tol):
        raise ValueError(f"start distance {r0} is outside the orbit's range [{r_p}, {r_a}]")
    r0 = min(r0, r_a)

    def speed(r):
        return np.sqrt(max(2.0 * float(_radial_energy(pair, E, 0.0, r)) / mu, 0.0))

    if r_p == 0.0:
        # Straight line: time from the centre out to r0 by the same stepping.
        t, r = 0.0, 1e-3
        while r < r0:
            v = speed(r)
            h = min(eta * max(r, 0.05) / v, (r0 - r) / v + 1e-12)
            # midpoint rule on dr/dt = v(r)
            r_new = r + h * speed(r + 0.5 * h * v)
            t += h
            r = r_new
        return np.array([0.0, -r0, 0.0]), np.array([0.0, speed(r0), 0.0]), t

    def deriv(y):
        r = np.hypot(y[0], y[1])
        a = -float(pair.dW(r)) / (mu * r)
        return np.array([y[2], y[3], a * y[0], a * y[1]])

    y = np.array([r_p, 0.0, 0.0, L / (mu * r_p)])
    t = 0.0
    while r0 > r_p * (1.0 + 1e-12):  # (starting at the pericentre itself: nothing to integrate)
        r = np.hypot(y[0], y[1])
        v = np.hypot(y[2], y[3])
        h = eta * r / v
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * h * k1)
        k3 = deriv(y + 0.5 * h * k2)
        k4 = deriv(y + h * k3)
        y_new = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        r_new = np.hypot(y_new[0], y_new[1])
        if r_new >= r0:
            # Land on r0: r is close to linear in the step over this last bit.
            frac = (r0 - r) / (r_new - r)
            h *= frac
            k2 = deriv(y + 0.5 * h * k1)
            k3 = deriv(y + 0.5 * h * k2)
            k4 = deriv(y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            break
        if r_new <= r:
            # Turned around just short of r0 (r0 at the apocentre): stop here.
            break
        y, t = y_new, t + h

    # Put the state exactly on r0 with the exact conserved quantities: the
    # direction comes from the integration, the speeds from E and L.
    phi = np.arctan2(y[1], y[0])
    v_t = L / (mu * r0)
    v_r = np.sqrt(max(2.0 * float(_radial_energy(pair, E, L, r0)) / mu, 0.0))
    c, s = np.cos(phi), np.sin(phi)
    # Outgoing state at angle +phi; incoming is its mirror image in the x-axis.
    pos = r0 * np.array([c, -s, 0.0])
    vel = np.array([-(v_r * c - v_t * s), v_r * s + v_t * c, 0.0])
    return pos, vel, t
