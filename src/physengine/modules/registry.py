from physengine.modules.blackholemodule import BlackHoleModule
from physengine.modules.galaxymodule import GalaxyModule
from physengine.modules.nbodymodule import NBodyModule

REGISTRY = {
    "NbodyModule": NBodyModule,
    "Galaxies": GalaxyModule,
    "Black hole": BlackHoleModule,
}
