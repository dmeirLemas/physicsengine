from copy import deepcopy

import numpy as np


class NBodyState:
    def __init__(self, N) -> None:
        self.N = N

        self.pos = np.zeros(shape=(N, 3), dtype=np.float64)
        self.vel = np.zeros(shape=(N, 3), dtype=np.float64)
        self.mass = np.zeros(shape=(N,), dtype=np.float64)


class Integrator:
    def __init__(self, force_func):
        self.a = force_func
        self.acc_cache = None

    def invalidate(self):
        self.acc_cache = None

    def step(self, state, dt, n):
        pos = deepcopy(state.pos)
        vel = deepcopy(state.vel)
        mass = deepcopy(state.mass)

        if self.acc_cache is None or self.acc_cache.shape != pos.shape:
            self.acc_cache = self.a(pos, mass)

        for _ in range(n):
            v_half = vel + (dt / 2) * self.acc_cache
            pos += dt * v_half
            self.acc_cache = self.a(pos, mass)
            vel[:, :] = v_half + (dt / 2) * self.acc_cache

        state.pos[:, :] = pos
        state.vel[:, :] = vel


def acc(pos, mass, G, eps):
    diff = pos[None, :, :] - pos[:, None, :]  # (N, N, 3)
    dist2 = np.sum(diff**2, axis=-1) + eps**2  # (N, N)

    np.fill_diagonal(dist2, 1.0)
    inv_d3 = dist2**-1.5
    np.fill_diagonal(inv_d3, 0.0)

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

    pairwise_pot = -G * mass[i] * mass[j] / np.sqrt(dist_sq + eps**2)
    pot = np.sum(pairwise_pot)

    return np.sum(kin) + pot
