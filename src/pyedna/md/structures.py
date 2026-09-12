"""Pure helpers for selected MD structure run layout."""

from pathlib import Path


def resolve_structure_runs(config, workdir):
    """Resolve selected finalized structures before starting MD work."""

    workdir = Path(workdir)
    runs = []
    for index, structure in enumerate(config.system.structures):
        structure_path = config.system.structure_path(structure)
        if not structure_path.is_absolute():
            structure_path = workdir / structure_path
        if not structure_path.exists():
            raise FileNotFoundError(
                f"Requested structure not found: {structure_path}"
            )
        runs.append({
            "index": index,
            "structure": structure,
            "structure_path": structure_path,
            "directory": f"structure_{structure:03d}",
        })
    return runs


def write_manifest(path, name, structure_runs, system_config):
    """Write the immutable run-to-structure mapping."""

    lines = [
        "[run]",
        f'name = "{name}"',
        "",
    ]
    for run in structure_runs:
        structure_path = system_config.structure_path(run["structure"])
        lines += [
            "[[structures]]",
            f"index = {run['index']}",
            f"structure = {run['structure']}",
            f'input = "{structure_path.as_posix()}"',
            f'directory = "{run["directory"]}"',
            "",
        ]
    Path(path).write_text("\n".join(lines))


def write_status(structure_dir, structure, state, stage):
    """Write the mutable status file for one structure worker."""

    text = f'structure = {structure}\nstate = "{state}"\nstage = "{stage}"\n'
    (Path(structure_dir) / "status.toml").write_text(text)
