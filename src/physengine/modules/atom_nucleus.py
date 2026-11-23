"""The nucleus: size, a packed arrangement of nucleons to draw, the binding-energy curve.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, atom
module). Review the physics.

Units: fm, MeV.

Physics: the radius R = r0 A^(1/3) with r0 = 1.2 fm (nuclear matter has a
nearly constant density, so the volume grows with A), binding energies from
the mass table and from the semi-empirical formula (atom_data), the Coulomb
energy of two touching nuclei.

Illustration: where the nucleons are drawn. A nucleus is not a bag of small
hard balls at fixed places; pack() only produces a tidy cluster of the right
overall size, with the right numbers of protons and neutrons.
"""

import numpy as np
from numba import njit

from physengine.modules.atom_data import E2, EXCESS_H1, EXCESS_N, R0_FM, semf_binding

NUCLEON_DRAW_RADIUS = 1.0  # fm: the size nucleons are drawn (display)


def radius(A):
    """Nuclear radius [fm]: R = r0 A^(1/3)."""
    return R0_FM * np.asarray(A, dtype=float) ** (1.0 / 3.0)


def coulomb_barrier(Z1, A1, Z2, A2):
    """Coulomb energy [MeV] of two nuclei just touching: Z1 Z2 e^2 / (R1 + R2)."""
    return Z1 * Z2 * E2 / (radius(A1) + radius(A2))


@njit(cache=True)
def _relax(points, limit, spacing, iterations):
    n = len(points)
    for _ in range(iterations):
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[j, 0] - points[i, 0]
                dy = points[j, 1] - points[i, 1]
                dz = points[j, 2] - points[i, 2]
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d < spacing and d > 1e-9:
                    push = 0.5 * (spacing - d) / d
                    points[i, 0] -= push * dx
                    points[i, 1] -= push * dy
                    points[i, 2] -= push * dz
                    points[j, 0] += push * dx
                    points[j, 1] += push * dy
                    points[j, 2] += push * dz
        for i in range(n):
            # A weak pull to the centre keeps the cluster compact; the wall keeps it round.
            r = np.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2 + points[i, 2] ** 2)
            scale = 0.97
            if r * scale > limit:
                scale = limit / r
            points[i, 0] *= scale
            points[i, 1] *= scale
            points[i, 2] *= scale
    return points


def pack(Z, A, seed=None):
    """Positions [fm] of A nucleons in a ball of radius R = r0 A^(1/3), and
    which of them are protons. For drawing only (see the module docstring).

    Random points are pushed apart until they sit about one nucleon spacing
    from each other. The same Z and A always give the same picture.
    """
    A, Z = int(A), int(Z)
    rng = np.random.default_rng(1000 * Z + A if seed is None else seed)
    if A == 1:
        return np.zeros((1, 3)), np.array([Z == 1])
    R = float(radius(A))
    direction = rng.normal(size=(A, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points = direction * (R * rng.random(A) ** (1.0 / 3.0))[:, None]
    # Spacing of a dense packing of A balls in the volume of radius R.
    spacing = 2.0 * R0_FM * 0.74 ** (1.0 / 3.0) * 0.98
    limit = max(R - 0.5 * spacing, 0.5 * spacing if A > 1 else 0.0)
    points = _relax(points, limit, spacing, 120)
    points -= points.mean(axis=0)
    protons = np.zeros(A, dtype=bool)
    protons[rng.permutation(A)[:Z]] = True
    return points, protons


def valley_z(A):
    """The most stable Z for mass number A according to the semi-empirical
    formula: the Z with the smallest atomic mass."""
    A = int(A)
    candidates = np.arange(max(1, A // 4), A)
    if len(candidates) == 0:
        return 1
    # M(Z, A) = Z M(H-1) + (A - Z) m_n - B, so the smallest atomic mass is
    # the largest B + Z (m_n - M(H-1)).
    score = semf_binding(candidates, A) + candidates * (EXCESS_N - EXCESS_H1)
    return int(candidates[np.argmax(score)])


def binding_curve(table, a_max=270):
    """The curve of binding energy per nucleon.

    Returns (A, measured, formula): for each mass number 1..a_max the largest
    B / A among the nuclides in the table (nan where the table has none or
    only estimates), and the formula's B / A at its most stable Z (from A = 8).
    """
    A = np.arange(1, a_max + 1)
    measured = np.full(len(A), np.nan)
    for (Z, a), (_, estimated) in table.masses.items():
        if estimated or a > a_max or Z == 0:
            continue
        value = table.binding(Z, a) / a
        if np.isnan(measured[a - 1]) or value > measured[a - 1]:
            measured[a - 1] = value
    formula = np.full(len(A), np.nan)
    for a in A[7:]:
        formula[a - 1] = float(semf_binding(valley_z(a), a)) / a
    return A, measured, formula
