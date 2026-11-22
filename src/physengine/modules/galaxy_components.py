"""The parts of a galaxy: labels per particle, and the model of each part.

The labels are shared by the galaxy simulation code (which assigns them) and
the galaxy renderer (which colours by them). Rename or extend as you like;
keep the renderer's colour table in step.

The dataclasses describe one galaxy model (written by Claude under a one-off
override of CLAUDE.md, 2026-11-20; the values in DEFAULT_MODEL are the ones
the module used before). Model units: G = 1, kpc, 1e10 M_sun.
"""

from dataclasses import dataclass, field

HALO = 0
BULGE = 1
DISC = 2


@dataclass
class Hernquist:
    """Hernquist sphere (Hernquist 1990, ApJ 356, 359), cut off at r_max."""

    n: int  # particle count
    mass: float  # [1e10 M_sun]
    a: float  # scale radius [kpc]
    r_max: float  # cut-off radius [kpc]


@dataclass
class ExponentialDisc:
    """Exponential disc with a sech^2 vertical profile, cut off at R_max."""

    n: int  # particle count
    mass: float  # [1e10 M_sun]
    R_d: float  # scale length [kpc]
    z0: float  # scale height [kpc]
    R_max: float  # cut-off radius [kpc]
    # Toomre stability parameter at 2.5 R_d. Above 1 the disc is stable
    # against clumping; 1.5 still lets it form spiral structure.
    Q: float = 1.5


def _default_halo():
    return Hernquist(n=4000, mass=50.0, a=30.0, r_max=300.0)


def _default_bulge():
    return Hernquist(n=1000, mass=1.0, a=0.6, r_max=6.0)


def _default_disc():
    return ExponentialDisc(n=4000, mass=5.0, R_d=3.0, z0=0.3, R_max=18.0, Q=1.5)


@dataclass
class GalaxyModel:
    """One galaxy: dark halo, bulge and disc. The defaults are DEFAULT_MODEL."""

    halo: Hernquist = field(default_factory=_default_halo)
    bulge: Hernquist = field(default_factory=_default_bulge)
    disc: ExponentialDisc = field(default_factory=_default_disc)

    def total_mass(self):
        return self.halo.mass + self.bulge.mass + self.disc.mass

    def particle_count(self):
        """Massive particles (tracer stars not included)."""
        return self.halo.n + self.bulge.n + self.disc.n


DEFAULT_MODEL = GalaxyModel()
