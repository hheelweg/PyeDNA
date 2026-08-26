# Create Structure

The structure workflow builds DNA-dye systems from reusable components and prepares Amber MD inputs.

## Pages

- [create_structure](create_structure.md) describes the multi-stage structure workflow and `structure.toml`.
- [amber_setup](amber_setup.md) describes the transition from a selected docked model to `prmtop` and `rst7`.

## Command Stages

```text
prepare  -> prepare DNA, dye-linker components, HADDOCK topologies, restraints, and docking_config.cfg
finalize -> select HADDOCK models and reconstruct final PDB structures
amber    -> prepare a selected final PDB with tleap
```

The job wrapper `jobs/structure/create_structure.sh` runs `prepare`, then submits `jobs/structure/run_haddock.sh`. The HADDOCK job runs HADDOCK3 and then calls `finalize`.
