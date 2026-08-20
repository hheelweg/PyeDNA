import argparse
from pathlib import Path

from pyedna.components import LinkerDefinition


def main(config_file="linker.toml"):

    workdir = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = workdir / config_path

    print(f"Linker configuration: {config_path}")

    linker = LinkerDefinition.from_file(config_path)
    linker.validate()

    linker.generate_conformer(workdir / f"{linker.name}.sdf")

    linker.optimize_geometry(workdir / "qm_opt")

    linker.compute_resp_esp(
        workdir / "qm_opt" / f"{linker.name}_opt.xyz",
        workdir / "qm_opt" / f"{linker.name}.esp",
    )

    mol2 = linker.generate_charges(workdir)

    print(f"Generated AMBER mol2: {mol2}")

    residue_templates = linker.generate_residue_templates(mol2, workdir)

    for name, paths in residue_templates.items():
        print(f"Generated {name} mol2: {paths['mol2']}")
        print(f"Generated {name} frcmod: {paths['frcmod']}")
        print(f"{name} mol2 charge sum: {paths['charge']: .6f}")

    removed = linker.cleanup_outputs(workdir)
    print(f"Cleanup mode: {linker.output.cleanup}")
    print(f"Removed temporary files: {len(removed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a linker")
    parser.add_argument(
        "--config",
        default="linker.toml",
        help="Linker TOML file (default: linker.toml)",
    )
    args = parser.parse_args()
    main(config_file=args.config)
