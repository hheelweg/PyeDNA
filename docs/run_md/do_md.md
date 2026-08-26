# do_md

## Purpose

`do_md` runs Amber molecular dynamics for a prepared PyeDNA system. The workflow is:

```text
minimization
    -> equilibration / heating
    -> production
```

## What the Workflow Does

PyeDNA creates a timestamped directory under `output.directory`, copies the input `md.toml`, symlinks or copies `prmtop` and `rst7`, writes Amber input files for each internal stage, runs `sander` for minimization, runs `pmemd.cuda` for equilibration and production, verifies expected files, and cleans intermediate files according to `output.cleanup`.

## Prerequisites

- Amber `prmtop` and `rst7` files from structure Amber setup.
- GPU allocation for the current script path. `scripts/do_md.py` checks that at least one CUDA GPU is visible.
- Amber executables available, including `sander` and `pmemd.cuda`.

## User Input Required

**Required:** system name and the prepared Amber input files.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should choose MD length, timestep, output intervals, restraints, and cleanup level for different scientific goals.

## Minimal Configuration Example

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
sbatch "$PYEDNA_HOME/jobs/md/do_md_gpu.sh" md.toml
```

The direct entry point is:

```bash
python "$PYEDNA_HOME/scripts/do_md.py" md.toml
```

## Common Modifications Or Advanced Options

Use `workflow.stages` to rerun a subset only when required input restart files already exist in the new run directory or are otherwise available. Adjust `output.cleanup` to retain more or fewer runtime files.

## Limitations / Troubleshooting

The current direct script requires a visible CUDA GPU before starting. `custom` restraint targets are not implemented. `equilibrate` requires `min_<name>.ncrst`; `production` requires `eq2_<name>.ncrst`.
