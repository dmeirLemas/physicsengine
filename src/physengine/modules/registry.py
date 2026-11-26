from physengine.modules.atommodule import AtomModule
from physengine.modules.blackholemodule import BlackHoleModule
from physengine.modules.galaxymodule import GalaxyModule
from physengine.modules.homemodule import HomeModule
from physengine.modules.nbodymodule import NBodyModule
from physengine.modules.particlemodule import ParticleModule

REGISTRY = {
    "Home": HomeModule,
    "Massive bodies": NBodyModule,
    "Galaxies": GalaxyModule,
    "Black hole": BlackHoleModule,
    "Particles": ParticleModule,
    "Atom": AtomModule,
}
