# Workflow Overview

PyeDNA is organized as installed-package CLI workflows. Scheduler `.sh` files may wrap these commands on HPC systems, but the `pyedna` CLI is the PyeDNA entry point.

> **Important**
>
> Make sure `pyedna_env` environment (as mentioned in [installation.md](installation.md)) is active and runtime configuration `config.toml` is properly set up with correct paths. 

## 1. Create Reusable Components

Use [create_dye](../create_components/create_dye.md) to parameterize a capped dye core and write a reusable dye residue template. Use [create_linker](../create_components/create_linker.md) to parameterize linker residue templates for 3' and 5' contexts. Use [create_dyelnk](../create_components/create_dyelnk.md) to assemble existing dye and linker templates into a linked dye-linker component used by structure generation.

```bash
pyedna components create-dye dye.toml
pyedna components create-linker linker.toml
pyedna components create-dyelnk dyelnk.toml
```

## 2. Create a DNA-Dye Structure

Use [create_structure](../create_structure/create_structure.md) to prepare DNA, place dye-linker components at configured DNA residues, write HADDOCK3 inputs, run docking, process selected HADDOCK models, and prepare final Amber inputs.

```bash
pyedna structure prepare structure.toml
pyedna structure dock structure.toml
pyedna structure finalize structure.toml
pyedna structure amber structure.toml
```

The command stages are:

```text
prepare  -> write DNA/HADDOCK inputs
dock     -> run HADDOCK3 from docking_config.cfg
finalize -> select and reconstruct docked models
amber    -> run tleap on a selected finalized model
```

## 3. Prepare Amber Inputs

[Amber setup](../create_structure/amber_setup.md) starts from a selected finalized PDB and a generated bond table. PyeDNA writes a `tleap` input file, loads DNA and dye/linker force-field data, adds covalent bonds, solvates, neutralizes, and writes `prmtop`, `rst7`, and solvated PDB files.

## 4. Run Molecular Dynamics

Use [do_md](../run_md/do_md.md) with `md.toml`. The MD workflow selects one pmemd engine from the resources visible to the process: serial CPU jobs use `pmemd`, CPU MPI jobs use `pmemd.MPI`, and GPU jobs with a visible CUDA device use `pmemd.cuda`.

```bash
pyedna md run md.toml
```

## 5. Analyze Trajectories

Use [analyze_traj](../analyze_trajectory/analyze_traj.md) with `traj.toml`. Analysis starts from an Amber topology and NetCDF trajectory, builds capped dye snapshots at configured attachments, groups attachments into scientific units, and runs requested classical or quantum calculations.

```bash
pyedna analysis trajectory traj.toml
```

The current quantum trajectory workflow requires GPU4PySCF and does not implement a CPU-only fallback.
