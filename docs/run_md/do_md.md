# Run Amber MD Simulations (`do_md`)

## Purpose

`do_md` runs Amber molecular dynamics for a prepared DNA/dye system.

## What the Workflow Does

PyeDNA creates a timestamped directory under `output.directory`, copies the input `md.toml`, symlinks or copies `prmtop` and `rst7`, writes Amber input files for each internal stage, selects one Amber pmemd engine from the resources visible to the process, runs all requested MD stages with that engine, verifies expected files, and cleans intermediate files according to `output.cleanup`.

### Stage Summary

| User stage | Internal stage | What happens |
| --- | --- | --- |
| `minimize` | `min1` | The first minimization typically relaxes solvent and ions around the prepared DNA-dye structure while keeping the DNA/dye coordinates fixed. This removes unfavorable solvent contacts before the molecular structure itself is allowed to move. |
|  | `min2` | The second minimization continues from the `min1` coordinates and also relaxes the DNA/dye structure, giving the full system a lower-energy starting point before heating. |
| `equilibrate` | `eq1` | The heating stage starts from the minimized structure, initializes MD from the configured initial temperature, and raises the system toward the target simulation temperature. |
|  | `eq2` | The NPT equilibration stage continues from the heated structure, turns on pressure coupling, and lets the solvated system settle at the configured temperature and pressure before production. |
| `production` | `prod` | The production stage continues from the equilibrated structure and writes the trajectory used for downstream analysis. |

## Prerequisites

- Amber `prmtop` and `rst7` files from structure Amber setup.
- PyeDNA runtime configuration with `amber.pmemd_home` providing `pmemd`, plus `pmemd.MPI` for CPU MPI jobs or `pmemd.cuda` for GPU jobs.
- For GPU MD, a scheduler allocation that exposes a CUDA device to the job process.

## User Input Required

**Required:** system name and the prepared Amber input files.

> **MD Setup Files**
>
> The `do_md` workflow most seamlessly works for structures created with the `create_structure` workflow, provided that the `prepare_amber` step has not thrown an issue. 

Minimization and Equilibration happen in two steps and under specific restraints one can specify.

> **Restraint Usage**
>
> The default for every stage is `target = "none"`, meaning no positional restraints are applied. `target = "structure"` applies restraints to all non-solvent, non-ion residues, i.e. the DNA and dye/linker structure. `target = "terminal"` applies restraints only to terminal DNA residues; fixing terminal nucleotides can improve simulation stability by reducing end fraying or large terminal motions during MD.


## Minimal Configuration Example For `md.toml`

```toml
[system]
name = "example_system"
prmtop = "example_system.prmtop"
rst7 = "example_system.rst7"

[workflow]
stages = ["minimize", "equilibrate", "production"]

[simulation]
temperature = 300.0
pressure = 1.0
timestep = 0.002
cutoff = 8.0

[minimization]
max_steps = 1000
steepest_descent_steps = 500

[minimization.restraints.stage1]
target = "structure"
strength = 10.0

[minimization.restraints.stage2]
target = "none"

[equilibration]
heating_steps = 10000
npt_steps = 50000
ntpr = 5000
ntwx = 5000
ntwr = 5000

[production]
steps = 1000000
log_interval = 5000
trajectory_interval = 5000
restart_interval = 50000
force_interval = 0

[output]
directory = "md"
cleanup = "standard"
```

## Configuration Reference

### `[system]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `name` | required | none | System basename. |
| `prmtop` | optional | `<name>.prmtop` | Amber topology file. Relative paths resolve from the working directory. |
| `rst7` | optional | `<name>.rst7` | Amber restart/coordinate file. Relative paths resolve from the working directory. |

### `[workflow]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `stages` | optional | `["minimize", "equilibrate", "production"]` | Ordered list containing any of `"minimize"`, `"equilibrate"`, and `"production"`. Later stages require outputs from earlier stages. |

### `[simulation]`

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `temperature` | optional | `300.0` | Target temperature in K. |
| `pressure` | optional | `1.0` | Target pressure for NPT stages. |
| `timestep` | optional | `0.002` | Amber timestep, in ps. |
| `cutoff` | optional | `8.0` | Nonbonded cutoff. |
| `initial_temperature` | optional | `0.0` | Starting temperature for heating. |
| `iwrap` | optional | `1` | Amber coordinate wrapping control. |
| `ntb` | optional | `1` | Amber periodic boundary setting used in minimization/heating. |
| `ntc` | optional | `2` | Amber SHAKE control. |
| `ntf` | optional | `2` | Amber force evaluation control. |
| `ntp` | optional | `2` | Amber pressure scaling mode for NPT stages. |
| `ioutfm` | optional | `1` | Amber trajectory output format control; `1` writes NetCDF. |

### `[minimization]`

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `max_steps` | optional | `1000` | Total minimization cycles for each minimization substage. |
| `steepest_descent_steps` | optional | `500` | Steepest-descent cycles before switching minimizer. |

Minimization has two internal substages, `min1` and `min2`, configured by `[minimization.restraints.stage1]` and `[minimization.restraints.stage2]`.

### `[equilibration]`

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `heating_steps` | optional | `10000` | Steps for `eq1`, the heating stage. |
| `npt_steps` | optional | `50000` | Steps for `eq2`, the NPT equilibration stage. |
| `ntpr` | optional | `5000` | Equilibration log/energy output interval. |
| `ntwx` | optional | `5000` | Equilibration trajectory output interval. |
| `ntwr` | optional | `5000` | Equilibration restart output interval. |

Equilibration has two internal substages, `eq1` and `eq2`, configured by `[equilibration.restraints.stage1]` and `[equilibration.restraints.stage2]`.

### `[production]`

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `steps` | optional | `1000000` | Number of production MD steps. |
| `log_interval` | optional | `5000` | Production log/energy output interval; maps to Amber `ntpr`. |
| `trajectory_interval` | optional | `5000` | Production trajectory output interval; maps to Amber `ntwx`. |
| `restart_interval` | optional | `50000` | Production restart output interval; maps to Amber `ntwr`. |
| `force_interval` | optional | `0` | Production force output interval; maps to Amber `ntwf`. |

Legacy production names `ntpr`, `ntwx`, `ntwr`, and `ntwf` are accepted if the clearer names are not also present.

### Restraints

Each restraint table has:

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `target` | optional | `"none"` | One of `"none"`, `"terminal"`, `"structure"`, or `"custom"`. `custom` validates but is not implemented at runtime. |
| `strength` | required unless target is `"none"` | none | Amber positional restraint weight. |

`terminal` selects terminal DNA residues. `structure` selects all non-solvent, non-ion residues from the topology. Solvent and ions are inferred from topology molecule and bonding data.

### `[thermostat]`, `[barostat]`, and `[output]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `[thermostat].type` | optional | `"langevin"` | Only `"langevin"` is currently supported. |
| `[thermostat].gamma` | optional | `5.0` | Langevin collision frequency. |
| `[thermostat].seed` | optional | `-1` | Amber random seed. |
| `[barostat].tau` | optional | `2.0` | Pressure relaxation time. |
| `[output].directory` | optional | `"md"` | Root output directory. |
| `[output].cleanup` | optional | `"standard"` | One of `"minimal"`, `"standard"`, `"restart"`, or `"all"`. |

## Generated Outputs

The output directory is:

```text
<output.directory>/run_YYYY_MM_DD_HH_MM/
```

Important files include stage `.in` and `.out` files, minimization/equilibration restart files, equilibration NetCDF files, final `<name>.ncrst`, and final `<name>.nc`.

## How To Run The Workflow

```bash
pyedna md run md.toml
```

If the config filename is omitted, PyeDNA uses `md.toml` in the current directory:

```bash
pyedna md run
```

On HPC systems, use the sample scheduler wrapper:

```bash
sbatch jobs/md/do_md.sh md.toml
```

Edit the `#SBATCH` resource lines in `jobs/md/do_md.sh` for the resources you want to allocate on your cluster. Keep `md.toml` unchanged.

Serial Amber stages run through `srun`; CPU MPI stages run through `mpirun -np $SLURM_NTASKS`.

## CPU/GPU Resource Selection

`md.toml` stores scientific MD settings only. It does not contain a CPU/GPU backend field. The example [jobs/md/do_md.sh](../../jobs/md/do_md.sh) script requests SLURM resources, then runs the same command:

```bash
pyedna md run "$@"
```

Choose serial CPU, CPU MPI, or GPU execution by changing the script's `#SBATCH` resource lines:

```bash
# Serial CPU
#SBATCH --ntasks=1
```

```bash
# CPU MPI
#SBATCH --ntasks=<N>
#SBATCH --cpus-per-task=1
```

```bash
# GPU
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
```

At runtime, PyeDNA checks scheduler-provided environment variables and selects one Amber executable for the whole workflow:

| Runtime signal | Amber executable |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` exposes at least one CUDA device | `pmemd.cuda` |
| no visible CUDA device and `SLURM_NTASKS > 1` | `pmemd.MPI` |
| no visible CUDA device and `SLURM_NTASKS` is unset or `1` | `pmemd` |

If `CUDA_VISIBLE_DEVICES` exposes a CUDA device, PyeDNA selects:

```text
MD backend: GPU
Amber engine: pmemd.cuda
```

If no CUDA device is visible and `SLURM_NTASKS > 1`, PyeDNA selects:

```text
MD backend: CPU MPI
Amber engine: pmemd.MPI
```

Otherwise, PyeDNA selects serial CPU:

```text
MD backend: CPU
Amber engine: pmemd
```

The selected engine is resolved once per workflow run and reused for minimization, equilibration, and production. PyeDNA does not select GPU mode merely because `pmemd.cuda` is installed.

## Common Modifications Or Advanced Options

Use `workflow.stages` to rerun a subset only when required input restart files already exist in the new run directory or are otherwise available. Adjust `output.cleanup` to retain more or fewer runtime files.

## Limitations / Troubleshooting

The GPU backend uses serial `pmemd.cuda`; one GPU is expected for the standard wrapper examples. CPU scaling uses `pmemd.MPI` only when `SLURM_NTASKS > 1`; allocating many CPU cores with `--cpus-per-task` but only one task should not be expected to provide efficient CPU scaling. `custom` restraint targets are not implemented. `equilibrate` requires `min_<name>.ncrst`; `production` requires `eq2_<name>.ncrst`.
