"""Build dye-labeled DNA structures and prepare them for simulation."""

import importlib

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

attachments = _attachments_module

__all__ = [
    "AmberAtomMapping",
    "AmberConfig",
    "AmberSetup",
    "AttachmentAtom",
    "attachments",
    "DNAConfig",
    "DyeDefinition",
    "DyeInstance",
    "DyePlacement",
    "HaddockConfig",
    "HaddockSetup",
    "StructureBuilder",
    "StructureConfig",
    "WorkflowConfig",
]


def __getattr__(name):
    if name != "AmberSetup":
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module("pyedna.structure.amber")
    value = getattr(module, name)
    globals()[name] = value
    return value
