"""PyeDNA public package interface.

Package import is intentionally lightweight. Workflow objects are resolved
lazily so analysis, structure, and component entry points can import cleanly.
"""

import importlib

_MODULE_ALIASES = {
    "amber": "pyedna.structure.amber",
    "dye": "pyedna.structure.attachments",
    "haddock": "pyedna.structure.haddock",
}

_LAZY_MODULES = {
    "analysis": "pyedna.analysis",
    "md": "pyedna.md",
    "postproc": "pyedna.postproc",
    "structure": "pyedna.structure",
    "trajectory": "pyedna.trajectory",
    **_MODULE_ALIASES,
}

_LAZY_ATTRS = {
    "MDConfig": "pyedna.md",
    "MDSimulation": "pyedna.md",
    "StructureBuilder": "pyedna.structure",
    "StructureConfig": "pyedna.structure",
    "Trajectory": "pyedna.trajectory",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module

    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
