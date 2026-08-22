import argparse
from pathlib import Path

from pyedna.structure.attachments import DyeLinkerConfig


def main(config_file="dyelnk.toml"):
    """Assemble dye/linker templates and generate the final linked MOL2."""
    workdir = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = workdir / config_path

    dyelnk = DyeLinkerConfig.from_file(config_path)

    output = dyelnk.build_linked_mol2(workdir)

    print(f"Generated dye-linker MOL2: {output}")
    print(f"Generated dye-linker FRCMOD: {output.with_suffix('.frcmod')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a dye-linker structure"
    )

    parser.add_argument(
        "--config",
        default="dyelnk.toml",
        help="Dye-linker TOML file (default: dyelnk.toml)",
    )

    args = parser.parse_args()

    main(config_file=args.config)
