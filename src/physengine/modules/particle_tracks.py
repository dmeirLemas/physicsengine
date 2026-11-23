"""Paths of particles through the detector of the particle-physics module.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
4). Review the physics.

Units: metres, nanoseconds, GeV, tesla; c = 0.299792458 m/ns. The beams run
along z; the detector is a set of coaxial cylinders round it.

Motion in the magnet. A charge q (in units of e) with momentum p and energy
E in a uniform field B along z obeys dp/dt = q v x B: p_z is constant and
the transverse momentum turns at the angular rate
    omega = K q B c / E,   K = 0.299792458 GeV / (T m),
clockwise seen from +z for q B > 0, so the path is a helix of radius
    R = p_T / (K |q| B)     (1 GeV in 1 T: 3.34 m).
With a = K q B and (p_x0, p_y0) at t = 0:
    p_x = p_x0 cos(omega t) + p_y0 sin(omega t),
    p_y = -p_x0 sin(omega t) + p_y0 cos(omega t),
    x = x0 + (p_x0 sin(omega t) + p_y0 (1 - cos(omega t))) / a,
    y = y0 + (p_y0 sin(omega t) - p_x0 (1 - cos(omega t))) / a,
    z = z0 + p_z c t / E.

The detector's dimensions are close to those of CMS (CMS Collaboration 2008,
JINST 3, S08004). Inside the coil the field is uniform; in the iron yoke
outside it the returning flux points the other way, RETURN_FIELD of the
central value (1.8 T against 3.8 T in CMS), so a muon's path bends back and
makes an S. What the material does (energy loss, scattering, radiation,
showers, nuclear interactions) is in particle_material.py.

Exact here: the path between interactions, and where and when unstable
particles decay: a particle of lifetime tau travels for a time gamma t with
t drawn from exp(-t / tau), i.e. a mean distance beta gamma c tau (its
energy loss on the way is taken into account step by step for the path,
not for the clock).
"""

from dataclasses import dataclass

import numpy as np

from physengine.modules.particle_material import (
    BREM_PHOTON_MIN,
    Shower,
    bethe,
    bremsstrahlung_fraction,
    em_shower,
    hadronic_shower,
    region,
    scatter,
    scattering_step,
)

C = 0.299792458  # m/ns
K = 0.299792458  # GeV / (T m)

TRACKER_RADIUS = 1.2
ECAL_RADIUS, ECAL_Z = 1.3, 3.0  # front faces of the electromagnetic calorimeter
HCAL_RADIUS, HCAL_Z = 1.9, 3.6  # and of the hadron calorimeter
HCAL_OUTER = 2.9
SOLENOID_RADIUS = 3.0  # the central field fills r < this
YOKE_INNER = 3.3  # the return field fills this out to OUTER_RADIUS
RETURN_FIELD = -0.47
MUON_LAYERS = (4.4, 5.3, 6.2, 7.0)  # radii of the muon stations
OUTER_RADIUS, OUTER_Z = 7.3, 6.5  # where the display ends
STEP_LENGTH = 0.03  # m between stored points inside the coil
STEP_OUTSIDE = 0.08  # m outside it, where paths are nearly straight
MAX_PATH = 40.0  # m: a particle curling up in the field is dropped after this
STOPPED = 0.005  # GeV of kinetic energy below which a particle is taken as stopped
# No path can take more steps than this (MAX_PATH in the shortest steps, twice
# over): a last guard against a loop that would never end.
MAX_STEPS = int(2 * MAX_PATH / STEP_LENGTH)


@dataclass
class Track:
    name: str
    p4: np.ndarray  # four-momentum at the start (GeV)
    points: np.ndarray  # (n, 3) positions [m]
    times: np.ndarray  # (n,) times [ns]
    # "decay", "shower", "stopped", "conversion", "exit" or "lost"
    end: str
    p4_end: np.ndarray  # four-momentum at the end
    parent: int = -1  # index of the parent track in the event, -1: from the collision
    hits: list = None  # (position, time) where a muon crosses a muon station
    ionisation: float = 0.0  # GeV left along the path by ionisation
    shower: Shower | None = None  # the shower it ends in, if any
    origin: str = "collision"  # or "decay", or "radiated" (a photon from an electron)


def helix_step(pos, p4, q, field, dt):
    """Position and four-momentum after dt in a uniform field along z (exact)."""
    energy, px, py, pz = p4
    a = K * q * field
    if a == 0.0:
        return pos + p4[1:] * (C * dt / energy), p4
    w = a * C / energy * dt
    s, c = np.sin(w), np.cos(w)
    new_pos = np.array(
        [
            pos[0] + (px * s + py * (1.0 - c)) / a,
            pos[1] + (py * s - px * (1.0 - c)) / a,
            pos[2] + pz * C * dt / energy,
        ]
    )
    return new_pos, np.array([energy, px * c + py * s, -px * s + py * c, pz])


def field_at(pos, central):
    """Field along z at `pos` for a central field `central` (see the module docstring)."""
    r = np.hypot(pos[0], pos[1])
    if abs(pos[2]) >= OUTER_Z:
        return 0.0
    if r < SOLENOID_RADIUS:
        return central
    if YOKE_INNER <= r < OUTER_RADIUS:
        return RETURN_FIELD * central
    return 0.0


def propagate(name, kind, charge, p4, start, t0, field, lifetime=None, rng=None, material=True):
    """Follow a particle from `start` at time t0 until it decays (after `lifetime` ns
    in the lab, if given), showers, stops or leaves.

    Returns (Track, photons): the photons an electron radiated on the way,
    as (four-momentum, position, time). With material=False the detector is
    empty space with a field (for tests of the motion alone).

    Raises ValueError for a four-momentum, position or time that is not
    finite, or an energy that is not positive or is below the momentum
    (such a particle has no defined speed, and following it would never end).
    """
    pos = np.array(start, dtype=float)
    p = np.array(p4, dtype=float)
    if p.shape != (4,) or pos.shape != (3,):
        raise ValueError(f"{name}: expected a four-momentum and a position")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(pos)) and np.isfinite(t0)):
        raise ValueError(f"{name}: four-momentum, position or time is not finite")
    if p[0] <= 0.0 or p[1:] @ p[1:] > p[0] ** 2 * (1.0 + 1e-9):
        raise ValueError(f"{name}: not a physical four-momentum {p}")
    if lifetime is not None and not lifetime >= 0.0:
        raise ValueError(f"{name}: lifetime {lifetime} is not a time")
    m = max(float(np.sqrt(max(p[0] ** 2 - p[1:] @ p[1:], 0.0))), 0.0)
    points, times, hits, photons = [pos.copy()], [t0], [], []
    track = Track(name, np.array(p4), None, None, "", p, hits=hits)
    t, path, steps = t0, 0.0, 0
    crossed = 0.0  # radiation lengths of material crossed so far (for the scattering)
    heavy = charge != 0 and kind in ("muon", "hadron")
    hadron = kind in ("hadron", "neutral")
    while not track.end:
        momentum = np.linalg.norm(p[1:])
        speed = momentum / p[0] * C
        steps += 1
        if steps > MAX_STEPS or not np.isfinite(speed):
            track.end = "lost"
            break
        if speed <= 0.0:
            track.end = "decay" if lifetime is not None else "stopped"
            break
        r_before = np.hypot(pos[0], pos[1])
        dt = (STEP_LENGTH if r_before < SOLENOID_RADIUS else STEP_OUTSIDE) / speed
        if lifetime is not None and t - t0 + dt >= lifetime:
            dt, track.end = max(lifetime - (t - t0), 0.0), "decay"
        new_pos, p = helix_step(pos, p, charge, field_at(pos, field), dt)
        r_after = np.hypot(new_pos[0], new_pos[1])
        if kind == "muon":
            for layer in MUON_LAYERS:
                if r_before < layer <= r_after:
                    f = (layer - r_before) / (r_after - r_before)
                    hits.append((pos + f * (new_pos - pos), t + f * dt))
        length = speed * dt
        pos, t, path = new_pos, t + dt, path + length
        points.append(pos.copy())
        times.append(t)
        if track.end:
            break
        here = region(pos) if material else None
        if here is not None and kind != "neutrino":
            direction = p[1:] / momentum
            dense = here.name != "tracker"
            if kind == "electron":
                if dense:
                    track.shower = em_shower(here, pos, direction, t, p[0], False)
                    track.end = "shower"
                    break
                keep = bremsstrahlung_fraction(here, length, rng)
                radiated = (1.0 - keep) * p[0]
                if radiated > BREM_PHOTON_MIN:
                    photons.append((np.array([radiated, *(radiated * direction)]), pos.copy(), t))
                    energy = p[0] - radiated
                    p = np.array([energy, *(np.sqrt(max(energy**2 - m * m, 0.0)) * direction)])
            elif kind == "photon":
                if rng.uniform() < 1.0 - np.exp(-length / (9.0 / 7.0 * here.x0 / 100.0)):
                    if dense:
                        track.shower = em_shower(here, pos, direction, t, p[0], True)
                        track.end = "shower"
                    else:
                        track.end = "conversion"
                    break
            elif hadron and dense:
                if rng.uniform() < 1.0 - np.exp(-length / (here.lambda_i / 100.0)):
                    track.shower = hadronic_shower(
                        here, pos, direction, t, p[0] - (m if name in ("p", "n") else 0.0)
                    )
                    track.end = "shower"
                    break
            if heavy:
                lost = min(bethe(here, m, momentum, charge) * length, p[0] - m)
                track.ionisation += lost
                energy = p[0] - lost
                if energy - m < STOPPED:
                    track.ionisation += energy - m
                    p = np.array([m, 0.0, 0.0, 0.0])
                    track.end = "stopped" if lifetime is None else "decay"
                    break
                momentum = np.sqrt(energy**2 - m * m)
                p = np.array([energy, *(momentum * direction)])
            if charge != 0:
                after = crossed + length * 100.0 / here.x0
                theta0 = scattering_step(crossed, after, momentum, momentum / p[0], charge)
                crossed = after
                p = np.array([p[0], *scatter(p[1:], theta0, rng)])
        if abs(pos[2]) >= OUTER_Z or r_after >= OUTER_RADIUS:
            track.end = "exit"
        elif path > MAX_PATH:
            track.end = "lost"
    track.points, track.times, track.p4_end = np.array(points), np.array(times), p
    return track, photons


def helix_radius(pt, charge, field):
    """Radius of curvature [m] of transverse momentum pt [GeV] in `field` [T]."""
    return pt / (K * abs(charge) * field)
