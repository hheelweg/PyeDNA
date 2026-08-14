import argparse
from pathlib import Path

from pyedna.structure import StructureBuilder


def main(stage, config_file="structure.toml"):
    """Prepare or finalize the configured DNA–dye structure."""

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
    else:
        setup = builder.finalize()
        print(f"Final structures: {setup.structure_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a DNA–dye structure")
    parser.add_argument("stage", choices=["prepare", "finalize"])
    parser.add_argument(
        "--config",
        default="structure.toml",
        help="Structure TOML file (default: structure.toml)",
    )
    args = parser.parse_args()
    main(args.stage, config_file=args.config)
