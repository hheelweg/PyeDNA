"""Build dye-labeled DNA structures and prepare them for simulation."""

from .amber import AmberSetup
from . import attachments as _attachments_module
from .builder import StructureBuilder
from .attachments import AmberAtomMapping, AttachmentAtom, DyeDefinition, DyeInstance
from .config import (
    AmberConfig,
    DNAConfig,
    DyePlacement,
    HaddockConfig,
    StructureConfig,
    WorkflowConfig,
)
from .haddock import HaddockSetup
from .legacy import Chromophore, cleanPDB

attachments = _attachments_module

__all__ = [
    "AmberAtomMapping",
    "AmberConfig",
    "AmberSetup",
    "AttachmentAtom",
    "attachments",
    "Chromophore",
    "DNAConfig",
    "DyeDefinition",
    "DyeInstance",
    "DyePlacement",
    "HaddockConfig",
    "HaddockSetup",
    "StructureBuilder",
    "StructureConfig",
    "WorkflowConfig",
    "cleanPDB",
]
