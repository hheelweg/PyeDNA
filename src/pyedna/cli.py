import argparse

from pyedna.config_workflow import run_config



def main():
    parser = argparse.ArgumentParser(
        prog="pyedna",
        description="PyeDNA scientific workflow tools",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser(
        "config",
        help="Manage PyeDNA runtime configuration",
    )

    config_commands = config.add_subparsers(
        dest="config_command",
        required=True,
    )

    config_commands.add_parser(
        "init",
        help="Create a template runtime configuration",
    )

    config_commands.add_parser(
        "show",
        help="Show the active runtime configuration",
    )

    config_commands.add_parser(
        "check",
        help="Validate the runtime configuration",
    )

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
        from pyedna.structure.workflow import run_structure

        run_structure(args.stage, args.config)
    elif args.command == "config":
        run_config(args.config_command)
    elif args.command == "components":
        from pyedna.components.workflow import (
            run_create_dye,
            run_create_linker,
            run_create_dyelnk,
        )

        if args.component_command == "create-dye":
            run_create_dye(args.config)
        elif args.component_command == "create-linker":
            run_create_linker(args.config)
        elif args.component_command == "create-dyelnk":
            run_create_dyelnk(args.config)
    elif args.command == "md":
        from pyedna.md.workflow import run_md

        if args.md_command == "run":
            run_md(args.config)
    elif args.command == "analysis":
        from pyedna.analysis.workflow import run_trajectory_analysis

        if args.analysis_command == "trajectory":
            run_trajectory_analysis(args.config)


if __name__ == "__main__":
    main()
