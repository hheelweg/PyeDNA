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

Use [create_structure](../create_structure/create_structure.md) to prepare DNA, place dye-linker components at configured DNA residues, write HADDOCK3 inputs, run docking, and process selected HADDOCK models into ranked unsolvated structures.

```bash
pyedna structure prepare structure.toml
pyedna structure dock structure.toml
pyedna structure finalize structure.toml
```

The command stages are:

```text
prepare  -> write DNA/HADDOCK inputs
dock     -> run HADDOCK3 from docking_config.cfg
finalize -> select and reconstruct docked models
```

## 3. Run Molecular Dynamics

Use [do_md](../run_md/do_md.md) with `md.toml`. The MD workflow selects one or more finalized structures with `[system].structures`, prepares Amber inputs with `tleap`, and then selects one pmemd engine from the resources visible to the process: serial CPU jobs use `pmemd`, CPU MPI jobs use `pmemd.MPI`, and GPU jobs with visible CUDA devices use `pmemd.cuda`.

```bash
pyedna md run md.toml
```

One `sbatch jobs/md/do_md.sh md.toml` submission can run multiple selected structures; allocated GPUs determine how many structure workers run at once.

## 4. Analyze Trajectories

Use [analyze_traj](../analyze_trajectory/analyze_traj.md) with `traj.toml`. Analysis starts from one or more Amber topologies and NetCDF trajectories, typically selected from an MD run with `[trajectory].structures`, builds capped dye snapshots at configured attachments, groups attachments into scientific units, and runs requested classical or quantum calculations.

```bash
pyedna analysis trajectory traj.toml
```

Quantum trajectory analysis uses the same scientific TOML on CPU and GPU jobs. PyeDNA runs PySCF on CPU when no CUDA GPU is visible, and uses GPU4PySCF automatically when a GPU allocation and the validated GPU stack are available.
