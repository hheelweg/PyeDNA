"""Structure attachment definitions and dye-linker assembly."""

from .definitions import (
    AmberAtomMapping,
    AttachmentAtom,
    DyeDefinition,
    DyeInstance,
    create_dye_instances,
    load_dye_definitions,
)
from .dyelink import (
    DyeLinkerConfig,
    forcefield_id,
    resolve_connect_frcmod,
    tleap_source,
)

__all__ = [
    "AmberAtomMapping",
    "AttachmentAtom",
    "DyeDefinition",
    "DyeInstance",
    "DyeLinkerConfig",
    "create_dye_instances",
    "forcefield_id",
    "load_dye_definitions",
    "resolve_connect_frcmod",
    "tleap_source",
]
