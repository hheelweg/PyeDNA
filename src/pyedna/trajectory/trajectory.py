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
    if num_frames <= 0:
        raise ValueError("Trajectory contains no frames")

    if isinstance(frame_interval, str):
        if frame_interval.lower() == "all":
            return 0, num_frames - 1
        raise ValueError('frame_interval string value must be "all"')

    if not isinstance(frame_interval, (list, tuple)) or len(frame_interval) != 2:
        raise ValueError('frame_interval must be [initial_frame, final_frame] or "all"')

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


def resolve_analysis_trajectories(config, workdir="."):
    traj_cfg = config["trajectory"]
    workdir = Path(workdir)
    run_directory = _resolve_path(workdir, traj_cfg["run_directory"])

    if "structures" in traj_cfg:
        return _resolve_md_structure_trajectories(
            run_directory,
            traj_cfg["structures"],
        )

    topology_file = _resolve_path(workdir, traj_cfg["topology_file"])
    run_topology_file = run_directory / traj_cfg["topology_file"]
    if not topology_file.exists() and run_topology_file.exists():
        topology_file = run_topology_file

    return [{
        "trajectory_index": 0,
        "structure": None,
        "structure_directory": "",
        "topology_file": topology_file,
        "trajectory_file": run_directory / traj_cfg["trajectory_file"],
        "analysis_directory": None,
    }]


def _resolve_md_structure_trajectories(run_directory, structures):
    manifest = _load_md_manifest(run_directory)
    name = manifest.get("run", {}).get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"MD manifest is missing [run].name: {run_directory / 'manifest.toml'}"
        )

    entries = manifest.get("structures", [])
    by_structure = {
        entry.get("structure"): entry
        for entry in entries
        if isinstance(entry, dict)
    }
    resolved = []
    for trajectory_index, structure in enumerate(structures):
        entry = by_structure.get(structure)
        if entry is None:
            raise ValueError(
                f"Structure {structure} is not listed in {run_directory / 'manifest.toml'}"
            )
        structure_directory = entry.get("directory")
        if not isinstance(structure_directory, str) or not structure_directory:
            raise ValueError(
                f"MD manifest entry for structure {structure} is missing directory"
            )

        structure_dir = run_directory / structure_directory
        topology_file = structure_dir / f"{name}.prmtop"
        trajectory_file = structure_dir / f"{name}.nc"
        missing = [
            path for path in (topology_file, trajectory_file)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing analysis input file(s) for "
                f"structure {structure}: {missing}"
            )

        resolved.append({
            "trajectory_index": trajectory_index,
            "structure": structure,
            "structure_directory": structure_directory,
            "topology_file": topology_file,
            "trajectory_file": trajectory_file,
            "analysis_directory": structure_directory,
        })

    return resolved


def _load_md_manifest(run_directory):
    manifest_file = run_directory / "manifest.toml"
    if not manifest_file.exists():
        raise FileNotFoundError(f"MD manifest not found: {manifest_file}")
    if tomllib is None:
        raise ImportError("tomllib/tomli is required")

    with manifest_file.open("rb") as f:
        return tomllib.load(f)


def _resolve_path(workdir, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return workdir / path


def load_trajectory(topology_file, trajectory_file):
    traj = Trajectory(topology_file, trajectory_file)
    print(f"Trajectory loaded: {traj.num_frames} frames")
    return traj


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
            "cap": cap,
        })

    if not attachments:
        raise ValueError("No [[attachments]] blocks found in traj.toml")

    return attachments


def analyze_trajectory(config_file):
    cfg = load_config(config_file)
    target = resolve_analysis_trajectories(cfg)[0]

    traj = load_trajectory(target["topology_file"], target["trajectory_file"])
    validate_frame_interval(cfg["trajectory"]["frame_interval"], traj.num_frames)

    return traj, cfg
