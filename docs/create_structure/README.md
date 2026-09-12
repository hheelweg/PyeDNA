# Create Structure

The structure workflow builds DNA-dye systems from reusable components and writes ranked, finalized, unsolvated PDB models.

## Pages

- [create_structure](create_structure.md) describes the multi-stage structure workflow and `structure.toml`.
- [amber_setup](amber_setup.md) notes that Amber/tleap preparation now belongs to the MD workflow.

## Command Stages

```text
prepare  -> prepare DNA, dye-linker components, HADDOCK topologies, restraints, and docking_config.cfg
dock     -> run HADDOCK3 using docking_config.cfg
finalize -> select HADDOCK models and reconstruct final PDB structures
```

The canonical CLI commands are:

```bash
pyedna structure prepare structure.toml
pyedna structure dock structure.toml
pyedna structure finalize structure.toml
```

Scheduler scripts may wrap these commands on HPC systems, but the `jobs/` directory is not a required installation dependency.
