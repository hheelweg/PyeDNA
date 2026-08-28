# PyeDNA Documentation

PyeDNA prepares, simulates, and analyzes dye-labeled DNA systems. The user-facing workflow is organized around the scientific stages rather than the Python package layout:

```text
component creation
    -> pyedna components create-dye / create-linker
    -> pyedna components create-dyelnk
    -> libraries.dye_dir / libraries.linker_dir
    -> pyedna structure prepare
    -> DNA preparation
    -> pyedna structure dock
    -> model selection / finalization
    -> pyedna structure finalize
    -> pyedna structure amber
    -> prmtop + rst7
    -> pyedna md run
    -> Amber trajectory
    -> pyedna analysis trajectory
```

Install PyeDNA as a Python package, configure local external-software paths in `~/.config/pyedna/config.toml`, and run workflows through the `pyedna` CLI.

## Start Here

- [Installation](getting_started/installation.md)
- [Configuration](getting_started/configuration.md)
- [Workflow overview](getting_started/workflow_overview.md)

## Workflows

- [Create components](create_components/README.md)
- [Create a dye](create_components/create_dye.md)
- [Create a linker](create_components/create_linker.md)
- [Create a dye-linker component](create_components/create_dyelnk.md)
- [Create a structure](create_structure/create_structure.md)
- [Amber setup](create_structure/amber_setup.md)
- [Run MD](run_md/do_md.md)
- [Analyze a trajectory](analyze_trajectory/analyze_traj.md)

## Concepts

- [HADDOCK3 in PyeDNA](concepts/haddock3.md)
- [Force fields](concepts/force_fields.md)
- [Molecular files](concepts/molecular_files.md)

## Examples

Representative TOML inputs are in [examples](examples/). Values such as molecule names, residue indices, mapped atoms, and trajectory filenames are system-specific.
