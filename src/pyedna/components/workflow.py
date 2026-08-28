from pathlib import Path

from pyedna.components import DyeDefinition, LinkerDefinition
from pyedna.structure.attachments import DyeLinkerConfig


def run_create_dye(config_file="dye.toml"):
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


def run_create_linker(config_file="linker.toml"):
    cwd = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = cwd / config_path

    print(f"Linker configuration: {config_path}")

    linker = LinkerDefinition.from_file(config_path)
    workdir = linker.output_directory(cwd)

    print(f"Linker output directory: {workdir}")

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


def run_create_dyelnk(config_file="dyelnk.toml"):
    workdir = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = workdir / config_path

    dyelnk = DyeLinkerConfig.from_file(config_path)
    output = dyelnk.build_linked_mol2(workdir)

    print(f"Generated dye-linker MOL2: {output}")
    print(f"Generated dye-linker FRCMOD: {output.with_suffix('.frcmod')}")