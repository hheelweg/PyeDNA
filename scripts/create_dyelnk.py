import argparse
from pathlib import Path

from pyedna.structure.dyelnk import DyeLinkerConfig


def main(config_file="dyelnk.toml"):
    """Assemble dye/linker templates and generate the final linked MOL2."""
    workdir = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = workdir / config_path

    dyelnk = DyeLinkerConfig.from_file(config_path)

    assembled_pdb = dyelnk.assemble(
        workdir / f"{dyelnk.dye}_{dyelnk.linker}_assembled.pdb",
        n_conformers=20,
    )   

    tleap_input = workdir / "tleap_dyelnk.in"
    mol2_output = workdir / f"{dyelnk.dye}_{dyelnk.linker}_linked.mol2"

    dyelnk.write_tleap_input(
        assembled_pdb,
        output_file=tleap_input,
        mol2_output=mol2_output,
    )

    output = dyelnk.run_tleap(
        tleap_input,
        mol2_output,
        assembled_pdb=assembled_pdb,
        workdir=workdir,
    )

    print(f"Generated dye-linker MOL2: {output}")


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