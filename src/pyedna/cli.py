import argparse
from pathlib import Path

from pyedna.structure import StructureBuilder

from pyedna.components import DyeDefinition, LinkerDefinition
from pyedna.structure.attachments import DyeLinkerConfig

from pyedna.md import MDSimulation

from pyedna.analysis.workflow import run_trajectory_analysis



def run_structure(stage, config_file):
    config_path = Path(config_file).resolve()

    builder = StructureBuilder.from_file(
        config_path,
        workdir=Path.cwd(),
    )

    if stage == "prepare":
        builder.prepare()
    elif stage == "finalize":
        builder.finalize()
    elif stage == "amber":
        builder.prepare_amber()
    elif stage == "dock":
        import subprocess

        config = Path("docking_config.cfg")
        if not config.is_file():
            raise FileNotFoundError("docking_config.cfg not found")

        run_dir = Path("haddock/run")
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)

        subprocess.run(
            ["haddock3", str(config)],
            check=True,
        )
    else:
        raise ValueError(f"Unknown structure stage: {stage}")


def run_create_dye(config_file):
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


def run_create_linker(config_file):
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


def run_create_dyelnk(config_file):
    workdir = Path.cwd()
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = workdir / config_path

    dyelnk = DyeLinkerConfig.from_file(config_path)
    output = dyelnk.build_linked_mol2(workdir)

    print(f"Generated dye-linker MOL2: {output}")
    print(f"Generated dye-linker FRCMOD: {output.with_suffix('.frcmod')}")


def run_md(config_file):
    md = MDSimulation.from_file(config_file)
    print(f"MD run directory: {md.output_dir}", flush=True)
    md.run()

def run_analysis_trajectory(config_file):
    return run_trajectory_analysis(config_file)

def main():
    parser = argparse.ArgumentParser(
        prog="pyedna",
        description="PyeDNA scientific workflow tools",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    structure = commands.add_parser(
        "structure",
        help="Create and prepare DNA-dye structures",
    )

    structure.add_argument(
        "stage",
        choices=["prepare", "dock", "finalize", "amber"],
    )

    structure.add_argument(
        "config",
        nargs="?",
        default="structure.toml",
    )

    components = commands.add_parser(
        "components",
        help="Create molecular components",
    )

    component_commands = components.add_subparsers(
        dest="component_command",
        required=True,
    )

    create_dye = component_commands.add_parser(
        "create-dye",
        help="Create and parameterize a dye",
    )

    create_dye.add_argument(
        "config",
        nargs="?",
        default="dye.toml",
    )

    create_linker = component_commands.add_parser(
        "create-linker",
        help="Create and parameterize a linker",
    )

    create_linker.add_argument(
        "config",
        nargs="?",
        default="linker.toml",
    )

    create_dyelnk = component_commands.add_parser(
        "create-dyelnk",
        help="Create a linked dye-linker component",
    )

    create_dyelnk.add_argument(
        "config",
        nargs="?",
        default="dyelnk.toml",
    )

    md = commands.add_parser(
        "md",
        help="Run molecular dynamics workflows",
    )

    md_commands = md.add_subparsers(
        dest="md_command",
        required=True,
    )

    md_run = md_commands.add_parser(
        "run",
        help="Run an Amber MD simulation",
    )

    md_run.add_argument(
        "config",
        nargs="?",
        default="md.toml",
    )

    analysis = commands.add_parser(
        "analysis",
        help="Run trajectory and post-processing workflows",
    )

    analysis_commands = analysis.add_subparsers(
        dest="analysis_command",
        required=True,
    )

    trajectory = analysis_commands.add_parser(
        "trajectory",
        help="Analyze an MD trajectory",
    )

    trajectory.add_argument(
        "config",
        nargs="?",
        default="traj.toml",
    )

    args = parser.parse_args()

    if args.command == "structure":
        run_structure(args.stage, args.config)
    elif args.command == "components":
        if args.component_command == "create-dye":
            run_create_dye(args.config)
        elif args.component_command == "create-linker":
            run_create_linker(args.config)
        elif args.component_command == "create-dyelnk":
            run_create_dyelnk(args.config)
    elif args.command == "md":
        if args.md_command == "run":
            run_md(args.config)
    elif args.command == "analysis":
        if args.analysis_command == "trajectory":
            run_analysis_trajectory(args.config)


if __name__ == "__main__":
    main()