"""Trajectory snapshot and group-construction helpers."""

import importlib

_LAZY_ATTRS = {
    "Fragment": "pyedna.trajectory.structure",
    "Trajectory": "pyedna.trajectory.snapshot",
    "analyze_trajectory": "pyedna.trajectory.trajectory",
    "build_cap": "pyedna.trajectory.structure",
    "build_group_fragments": "pyedna.trajectory.structure",
    "build_groups": "pyedna.trajectory.structure",
    "combine_molecules": "pyedna.trajectory.structure",
    "get_external_neighbor": "pyedna.trajectory.structure",
    "infer_dye_charge": "pyedna.trajectory.structure",
    "load_analysis_attachments": "pyedna.trajectory.trajectory",
    "load_attach_atoms": "pyedna.trajectory.structure",
    "load_attachment_info": "pyedna.trajectory.structure",
    "load_config": "pyedna.trajectory.trajectory",
    "optimize_cap_geometry": "pyedna.trajectory.structure",
    "unit": "pyedna.trajectory.structure",
    "validate_frame_interval": "pyedna.trajectory.trajectory",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
