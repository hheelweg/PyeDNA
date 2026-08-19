import argparse
from pathlib import Path
import numpy as np

from pyedna.components import LinkerDefinition


def main(config_file="linker.toml"):

    workdir = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = workdir / config_path

    print(f"Linker configuration: {config_path}")

    linker = LinkerDefinition.from_file(config_path)
    linker.validate()


    linker.generate_conformer(
        workdir / f"{linker.name}.sdf"
    )

    linker.optimize_geometry(
        workdir / "qm_opt"
    )

    linker.compute_resp_esp(
        workdir/"qm_opt"/f"{linker.name}_opt.xyz",
        workdir/"qm_opt"/f"{linker.name}.esp"
    )

    mol2 = linker.generate_charges(workdir)

    print(f"Generated AMBER mol2: {mol2}")

    l03 = linker.extract_residue_mol2(
        mol2,
        "three_prime",
        workdir / "L03.mol2"
    )

    l05 = linker.extract_residue_mol2(
        mol2,
        "five_prime",
        workdir / "L05.mol2"
    )

    print(f"Generated L03 mol2: {l03}")
    print(f"Generated L05 mol2: {l05}")

    l03_frcmod = linker.generate_frcmod(
        l03,
        workdir / "L03.frcmod"
    )

    l05_frcmod = linker.generate_frcmod(
        l05,
        workdir / "L05.frcmod"
    )

    linker.print_partition_atoms()

    linker.print_partition_charges(mol2)

    linker.compare_boundary_difference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a linker")
    parser.add_argument(
        "--config",
        default="linker.toml",
        help="Linker TOML file (default: linker.toml)",
    )
    args = parser.parse_args()
    main(config_file=args.config)