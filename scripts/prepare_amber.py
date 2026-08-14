import argparse
from pathlib import Path

from pyedna.structure import StructureBuilder


def main(config_file="structure.toml"):
    """Prepare an Amber-ready structure and tleap input from a finalized model."""

    workdir = Path.cwd()
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = workdir / config_path
    print(f"Structure configuration: {config_path}")
    builder = StructureBuilder.from_file(config_path, workdir=workdir)
    setup = builder.prepare_amber()

    print(f"Input structure : {setup.input_pdb}")
    print(f"Prepared PDB    : {setup.amber_pdb}")
    print(f"Bond file       : {setup.bond_file}")
    print(f"tleap input     : {setup.tleap_file}")
    print(f"Amber topology  : {setup.prmtop_file}")
    print(f"Amber restart   : {setup.rst7_file}")
    print(f"Solvated PDB    : {setup.solvated_pdb}")
    print(f"Dyes            : {', '.join(setup.dye_definitions)}")
    print(f"Bonds           : {len(setup.bonds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Amber MD input files")
    parser.add_argument(
        "--config",
        default="structure.toml",
        help="Structure TOML file (default: structure.toml)",
    )
    args = parser.parse_args()
    main(config_file=args.config)
