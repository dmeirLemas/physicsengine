"""The stops of the guided tour (core/tour.py): which scenes are shown, for how long,
and what the caption says. Display choices only."""

from physengine.core.tour import Stop
from physengine.modules import atom_scenes

BODIES, GALAXIES, HOLE, PARTICLES, ATOM = (
    "Massive bodies",
    "Galaxies",
    "Black hole",
    "Particles",
    "Atom",
)


def _orbital():
    atom_scenes.SETTINGS["orbital"].update(n=3, l=2, m=0, Z=1, real=True)


def _fission():
    atom_scenes.SETTINGS["fission"].update(system="U-235", continuous=True)


def _fill(count):
    return lambda module: module.request_fill(count)


# Control panels folded away during the tour. The particle module's "Event" window
# and the "Atom" window stay: they hold the histogram and the numbers being shown.
COLLAPSE = ("Simulation", "Scene editor", "Galaxy setup", "Bodies")

STOPS = (
    Stop(
        BODIES,
        "Massive bodies: the inner solar system",
        "Direct-summation gravity with the first relativistic correction, leapfrog integration.",
        seconds=18.0,
        params={"scene_name": "Inner solar system"},
    ),
    Stop(
        BODIES,
        "Massive bodies: the figure-eight",
        "Three equal masses chase each other along one closed orbit.",
        seconds=15.0,
        params={"scene_name": "Figure-eight"},
    ),
    Stop(
        BODIES,
        "Massive bodies: a star torn by a black hole",
        "Inside the tidal radius the star comes apart; the pieces orbit or fall in.",
        seconds=25.0,
        params={"scene_name": "Star torn by a black hole"},
    ),
    Stop(
        GALAXIES,
        "Galaxies: a merger",
        "Two disc galaxies in dark-matter halos. Barnes-Hut tree: built on the CPU, walked on"
        " the GPU.",
        seconds=38.0,
        params={"scene_name": "Prograde", "tracers": "6000"},
        time_scale=10.0,
        orbit=0.006,  # nearly face-on throughout: the tidal tails lie in this plane
        zoom=0.8,
    ),
    Stop(
        HOLE,
        "Black hole: light bent round it",
        "One light ray per pixel, traced through curved spacetime to the disc or the sky.",
        seconds=24.0,
        params={"scene_name": "Disc only"},
        orbit=0.05,
        pitch=0.06,
        move=0.6,  # then at rest: the picture is traced at full resolution only when still
    ),
    Stop(
        HOLE,
        "Black hole: spinning, with a body in orbit",
        "Spin 0.9: the shadow is lopsided, and the orbit does not close (it precesses).",
        seconds=22.0,
        params={"scene_name": "Precessing orbit", "spin": 0.9},
        orbit=0.05,
        move=0.5,
    ),
    Stop(
        HOLE,
        "Black hole: a star torn apart",
        "A Sun-like star on a plunging orbit, followed as 2000 particles on exact geodesics.",
        seconds=26.0,
        params={"scene_name": "Star torn apart"},
        orbit=0.0,
        zoom=0.4,  # closes in on the hole as the star falls
    ),
    Stop(
        PARTICLES,
        "Particles: a Higgs boson decays to four leptons",
        "Tracks curl in the magnet; the four measured momenta add up to the Higgs mass.",
        seconds=20.0,
        params={"scene_name": "Higgs -> 4 leptons"},
        start=_fill(500),
    ),
    Stop(
        PARTICLES,
        "Particles: electrons and positrons form a Z",
        "The collision energy is scanned; how often a Z forms traces the resonance curve.",
        seconds=20.0,
        params={"scene_name": "e+ e- -> Z (collider scan)"},
        start=_fill(3000),
    ),
    Stop(
        ATOM,
        "Atom: a hydrogen 3d orbital",
        "The exact wavefunction, drawn as a cloud of points where the electron can be found.",
        seconds=16.0,
        params={"scene_name": "Hydrogen-like orbital"},
        prepare=_orbital,
        orbit=0.12,
    ),
    Stop(
        ATOM,
        "Nucleus: fission of uranium-235",
        "A neutron is absorbed, the nucleus stretches and splits, and new neutrons fly off.",
        seconds=22.0,
        params={"scene_name": "Fission of one nucleus"},
        prepare=_fission,
    ),
)
