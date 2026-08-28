# Create Structure

The structure workflow builds DNA-dye systems from reusable components and prepares Amber MD inputs.

## Pages

- [create_structure](create_structure.md) describes the multi-stage structure workflow and `structure.toml`.
- [amber_setup](amber_setup.md) describes the transition from a selected docked model to `prmtop` and `rst7`.

## Command Stages

```text
prepare  -> prepare DNA, dye-linker components, HADDOCK topologies, restraints, and docking_config.cfg
dock     -> run HADDOCK3 using docking_config.cfg
finalize -> select HADDOCK models and reconstruct final PDB structures
amber    -> prepare a selected final PDB with tleap
```

The canonical CLI commands are:

```bash
pyedna structure prepare structure.toml
pyedna structure dock structure.toml
pyedna structure finalize structure.toml
pyedna structure amber structure.toml
```

Scheduler scripts may wrap these commands on HPC systems, but the `jobs/` directory is not a required installation dependency.
