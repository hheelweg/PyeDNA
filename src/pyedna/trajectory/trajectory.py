from pathlib import Path

from pyedna.analysis.config import validate_analysis_config

from .snapshot import Trajectory
from .structure import (
    build_cap,
    build_groups,
    combine_molecules,
    get_external_neighbor,
    infer_dye_charge,
    load_attach_atoms,
    load_attachment_info,
    optimize_cap_geometry,
    unit,
)

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

def validate_frame_interval(frame_interval, num_frames):
    if not isinstance(frame_interval, (list, tuple)) or len(frame_interval) != 2:
        raise ValueError("frame_interval must be [initial_frame, final_frame]")

    start, stop = frame_interval

    if not isinstance(start, int) or not isinstance(stop, int):
        raise TypeError("frame_interval values must be integers")
    if start < 0:
        raise ValueError("Initial frame cannot be negative")
    if stop < start:
        raise ValueError("Final frame must be >= initial frame")
    if stop >= num_frames:
        raise ValueError(
            f"Final frame {stop} exceeds trajectory range 0-{num_frames - 1}"
        )

    return start, stop


def load_config(filename):
    if tomllib is None:
        raise ImportError("tomllib/tomli is required")

    with open(filename, "rb") as f:
        return validate_analysis_config(tomllib.load(f)).data

def load_analysis_attachments(config):
    attachments = []

    for item in config.get("attachments", []):
        if "dye" not in item or "residue" not in item:
            raise ValueError("Each [[attachments]] block requires dye and residue")

        cap = item.get("cap", "H").upper()

        if cap not in ("H", "CH3"):
            raise ValueError(
                f"Unsupported cap '{cap}' for residue {item['residue']}; use H or CH3"
            )

        attachments.append({
            "dye": item["dye"],
            "residue": item["residue"],
            "cap": cap
        })

    if not attachments:
        raise ValueError("No [[attachments]] blocks found in traj.toml")

    return attachments

def analyze_trajectory(config_file):
    cfg = load_config(config_file)
    traj_cfg = cfg["trajectory"]
    cwd = Path.cwd()

    traj = Trajectory(
        cwd / traj_cfg["topology_file"],
        cwd / traj_cfg["run_directory"] / traj_cfg["trajectory_file"]
    )

    print(f"Trajectory loaded: {traj.num_frames} frames")
    validate_frame_interval(traj_cfg["frame_interval"], traj.num_frames)

    return traj, cfg
