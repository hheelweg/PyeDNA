import sys as _sys

from .const import *
from .fileproc import *
from .geomtools import *
from .quanttools import *
from .structure import Chromophore, CreateDNA, StructureBuilder, StructureConfig
from .trajectory import MDSimulation, Trajectory
from .utils import *
from .pyscf_utils import *
from .plot_utils import *

# TODO: Remove these module aliases after callers have migrated to
# pyedna.structure.amber, pyedna.structure.dye, and pyedna.structure.haddock.
from .structure import amber as _amber_module
from .structure import dye as _dye_module
from .structure import haddock as _haddock_module

_sys.modules[f"{__name__}.amber"] = _amber_module
_sys.modules[f"{__name__}.dye"] = _dye_module
_sys.modules[f"{__name__}.haddock"] = _haddock_module
