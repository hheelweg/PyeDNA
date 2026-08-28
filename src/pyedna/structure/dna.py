"""Prepare DNA structures for structure-generation workflows."""

try:
    from importlib.resources import as_file, files
except ImportError:
    try:
        from importlib_resources import as_file, files
    except ImportError:
        as_file = files = None

from contextlib import nullcontext
import shutil
import subprocess
from pathlib import Path

from pyedna.config import get_config

from .pdb import set_chain_and_segid


def _resource_path(*parts):
    if files is None or as_file is None:
        return nullcontext(Path(__file__).resolve().parents[1].joinpath(*parts))

    resource = files("pyedna")
    for part in parts:
        resource = resource / part
    return as_file(resource)


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


def _run_nab(nab_file, workdir):
    """Run NAB through the project shell wrapper."""

    with _resource_path("data", "nab_scripts", "create_dna.sh") as run_nab_script:
        subprocess.run(
            [
                "bash",
                str(run_nab_script),
                Path(nab_file).name,
                str(get_config().nab.home),
            ],
            cwd=workdir,
            check=True,
            stdout=subprocess.DEVNULL,
        )


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
