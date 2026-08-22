import argparse
from pathlib import Path

from pyedna.structure import StructureBuilder


def print_amber_summary(setup):
    """Print the Amber files produced from a finalized structure."""

    print(f"Input structure : {setup.input_pdb}")
    print(f"Prepared PDB    : {setup.amber_pdb}")
    print(f"Bond file       : {setup.bond_file}")
    print(f"tleap input     : {setup.tleap_file}")
    print(f"Amber topology  : {setup.prmtop_file}")
    print(f"Amber restart   : {setup.rst7_file}")
    print(f"Solvated PDB    : {setup.solvated_pdb}")
    print(f"Dyes            : {', '.join(setup.dye_definitions)}")
    print(f"Bonds           : {len(setup.bonds)}")


def main(stage, config_file="structure.toml"):
    """Prepare, finalize, or Amber-prepare the configured DNA–dye structure."""

    workdir = Path.cwd()
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = workdir / config_path
    print(f"Structure configuration: {config_path}")
    builder = StructureBuilder.from_file(
        config_path,
        workdir=workdir,
    )

    if stage == "prepare":
        setup = builder.prepare()
        print(f"DNA structure: {setup.dna_pdb}")
        print(f"HADDOCK configuration: {setup.docking_config}")
    elif stage == "finalize":
        setup = builder.finalize()
        print(f"Final structures: {setup.structure_dir}")
        if builder.config.workflow.prepare_amber:
            amber_setup = builder.prepare_amber()
            print_amber_summary(amber_setup)
    else:
        setup = builder.prepare_amber()
        print_amber_summary(setup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a DNA–dye structure")
    parser.add_argument("stage", choices=["prepare", "finalize", "amber"])
    parser.add_argument(
        "--config",
        default="structure.toml",
        help="Structure TOML file (default: structure.toml)",
    )
    args = parser.parse_args()
    main(args.stage, config_file=args.config)
