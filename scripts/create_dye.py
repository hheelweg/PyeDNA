import argparse
from pathlib import Path

from pyedna.components import DyeDefinition


def main(config_file="dye.toml"):
    cwd = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = cwd / config_path

    print(f"Dye configuration: {config_path}")

    dye = DyeDefinition.from_file(config_path)
    workdir = dye.output_directory(cwd)
    print(f"Dye output directory: {workdir}")
    dye.validate()

    dye.generate_conformer(workdir / f"{dye.name}.sdf")

    print(f"Generated dye structure: {dye.name}.sdf")
    print(f"Generated dye structure: {dye.name}.pdb")

    dye.optimize_geometry(workdir / "qm_opt")

    dye.compute_resp_esp(
        workdir / "qm_opt" / f"{dye.name}_opt.xyz",
        workdir / "qm_opt" / f"{dye.name}.esp",
    )

    mol2 = dye.generate_charges(workdir)

    print(f"Generated capped dye mol2: {mol2}")

    residue_template = dye.generate_residue_template(mol2, workdir)

    print(f"Generated {dye.residue_name} mol2: {residue_template['mol2']}")
    print(f"Generated {dye.residue_name} frcmod: {residue_template['frcmod']}")
    print(f"{dye.residue_name} mol2 charge sum: {residue_template['charge']: .6f}")

    removed = dye.cleanup_outputs(workdir)
    print(f"Cleanup mode: {dye.output.cleanup}")
    print(f"Removed temporary files: {len(removed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dye")

    parser.add_argument(
        "--config",
        default="dye.toml",
        help="Dye TOML file (default: dye.toml)",
    )
    args = parser.parse_args()

    main(config_file=args.config)
