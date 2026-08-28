"""Prepare DNA structures for structure-generation workflows."""

import shutil
import subprocess
import time
from pathlib import Path

from pyedna.config import get_config

from .pdb import set_chain_and_segid


def _copy_library_dna(dna_config, dna_dir, workdir):
    """Copy a configured DNA PDB from the DNA template library."""

    source_pdb = Path(dna_dir) / f"{dna_config.name}.pdb"
    output_pdb = Path(workdir) / f"{dna_config.name}.pdb"

    if not source_pdb.exists():
        raise FileNotFoundError(f"DNA template not found: {source_pdb}")

    shutil.copy2(source_pdb, output_pdb)
    print(f"Copied DNA template: {source_pdb} -> {output_pdb}")
    return output_pdb


def _load_nab_template(dna_type):
    """Load the NAB template for the requested DNA type."""

    if dna_type != "double_helix":
        raise NotImplementedError("Other DNA structures not implemented yet!")

    template_dir = Path(__file__).resolve().parents[1] / "data" / "dna_templates"
    template_file = template_dir / f"{dna_type}.nab"
    if not template_file.exists():
        raise FileNotFoundError(f"NAB template not found: {template_file}")
    return template_file.read_text()


def _write_nab_script(dna_config, workdir):
    """Render and write the NAB script for generated DNA."""

    template = _load_nab_template(dna_config.type)
    nab_script = template.replace("{DNA_SEQUENCE}", dna_config.sequence.lower())
    nab_script = nab_script.replace("{PDB_NAME}", f"{dna_config.name}.pdb")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    nab_file = workdir / f"{dna_config.name}.nab"
    nab_file.write_text(nab_script)
    return nab_file


def _prepend_env_path(env, key, path):
    current = env.get(key)
    env[key] = f"{path}{':' + current if current else ''}"


def _amberclassic_environment(amberclassic_dir, amber_home=None):
    """Return the environment produced by sourcing AmberClassic.sh."""

    amberclassic_dir = Path(amberclassic_dir)
    amber_home = Path(amber_home) if amber_home is not None else None
    setup_script = amberclassic_dir / "AmberClassic.sh"

    if not setup_script.is_file():
        raise FileNotFoundError(
            f"AmberClassic.sh not found in {amberclassic_dir}"
        )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null && env -0',
            "bash",
            str(setup_script),
        ],
        check=True,
        capture_output=True,
    )
    env = dict(
        item.split("=", 1)
        for item in result.stdout.decode().split("\0")
        if item
    )

    lib_dirs = [
        Path(env["CONDA_PREFIX"]) / "lib" if env.get("CONDA_PREFIX") else None,
        (
            Path(env["CONDA_PREFIX"]) / "x86_64-conda-linux-gnu" / "lib"
            if env.get("CONDA_PREFIX")
            else None
        ),
        Path(env["AMBERHOME"]) / "lib" if env.get("AMBERHOME") else None,
        amberclassic_dir / "lib",
    ]
    if amber_home is not None:
        lib_dirs.extend([
            amber_home / "lib",
            amber_home / "miniconda" / "lib",
            amber_home.parent / "lib",
            amber_home.parent / "x86_64-conda-linux-gnu" / "lib",
        ])

    for lib_dir in lib_dirs:
        if lib_dir is not None and lib_dir.is_dir():
            for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
                _prepend_env_path(env, key, lib_dir)

    return env


def _cleanup_nab_files(workdir):
    workdir = Path(workdir)

    for path in [workdir / "a.out", workdir / "tleap.out", *workdir.glob("*.c")]:
        path.unlink(missing_ok=True)


def _run_nab(nab_file, workdir):
    """Compile and run a NAB script using the configured AmberClassic tree."""

    workdir = Path(workdir)
    nab_file = Path(nab_file)

    if nab_file.suffix != ".nab":
        raise ValueError("The NAB input file must have a .nab extension")

    source = workdir / nab_file.name
    if not source.is_file():
        raise FileNotFoundError(
            f"{nab_file.name} not found in the current directory {workdir}"
        )

    config = get_config()
    env = _amberclassic_environment(config.nab.home, amber_home=config.amber.home)

    subprocess.run(
        ["nab", nab_file.name],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        env=env,
    )

    time.sleep(1)

    executable = workdir / "a.out"
    if not executable.is_file():
        raise FileNotFoundError("Compilation failed. a.out not generated")

    subprocess.run(
        [str(executable)],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        env=env,
    )
    _cleanup_nab_files(workdir)


def _generate_dna_with_nab(dna_config, workdir, remove_nab=True):
    """Generate a DNA PDB from sequence using a NAB template."""

    workdir = Path(workdir)
    output_pdb = workdir / f"{dna_config.name}.pdb"
    nab_file = _write_nab_script(dna_config, workdir)

    _run_nab(nab_file, workdir)
    print(
        f"*** Creation of {dna_config.name}.pdb completed: "
        f"DNA type = {dna_config.type}, DNA sequence = {dna_config.sequence}"
    )

    if not output_pdb.exists():
        raise FileNotFoundError(f"Generated DNA PDB not found: {output_pdb}")

    if remove_nab:
        nab_file.unlink(missing_ok=True)

    print(f"Generated DNA structure: {output_pdb}")
    return output_pdb


def _normalize_dna_pdb(pdb_file, chain="A", segid="A"):
    """Normalize chain and segment identifiers in a DNA PDB file in place."""

    pdb_file = Path(pdb_file)
    lines = []

    for line in pdb_file.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            line = set_chain_and_segid(line, chain=chain, segid=segid)
        lines.append(line)

    pdb_file.write_text("\n".join(lines) + "\n")


def prepare_dna(structure_config, dna_dir, workdir="."):
    """Create the configured DNA PDB from NAB or a DNA library template."""

    workdir = Path(workdir)
    dna_config = structure_config.dna

    if dna_config.source == "library":
        output_pdb = _copy_library_dna(dna_config, dna_dir, workdir)
    elif dna_config.source == "generate":
        output_pdb = _generate_dna_with_nab(dna_config, workdir)
    else:
        raise ValueError(
            f"Unknown dna_source {dna_config.source!r}; "
            "expected 'library' or 'generate'"
        )

    _normalize_dna_pdb(output_pdb, chain="A", segid="A")
    return output_pdb
