# Run Amber MD Simulations (`do_md`)

## Purpose

`do_md` prepares and runs Amber molecular dynamics for one or more finalized DNA/dye structural models.

## What the Workflow Does

PyeDNA validates all requested finalized structures before starting expensive work, creates one timestamped directory under `output.directory`, copies the input `md.toml`, writes an immutable `manifest.toml`, creates one `structure_<NNN>/` directory per selected ranked structure, prepares Amber topology/coordinate inputs with `tleap` inside each structure directory, writes Amber input files for each internal MD stage, selects Amber engines from the minimization configuration and visible runtime resources, and cleans intermediate files according to `output.cleanup`.

### Stage Summary

| User stage | Internal stage | What happens |
| --- | --- | --- |
| `prepare` | `tleap` | Reads `structures/<system>_<structure>.pdb` and `structures/bonds.csv`, applies component atom-name/connectivity metadata, solvates, neutralizes if requested, and writes `.prmtop`, `.rst7`, and a solvated PDB. |
| `minimize` | `min1` | The first minimization typically relaxes solvent and ions around the prepared DNA-dye structure while keeping the DNA/dye coordinates fixed. |
|  | `min2` | The second minimization continues from the `min1` coordinates and also relaxes the DNA/dye structure before heating. |
| `equilibrate` | `eq1` | The heating stage starts from the minimized structure and raises the system toward the target simulation temperature. |
|  | `eq2` | The NPT equilibration stage continues from the heated structure and lets the solvated system settle at the configured temperature and pressure. |
| `production` | `prod` | The production stage continues from the equilibrated structure and writes the trajectory used for downstream analysis. |

## Prerequisites

- Finalized structures from `create_structure finalize`, for example `structures/<system>_1.pdb`, `structures/bonds.csv`, and any generated dye-linker metadata under `structures/amber/`.
- PyeDNA runtime configuration with `amber.ambertools_home` providing `tleap` and `amber.pmemd_home` providing `pmemd`, plus `pmemd.MPI` for CPU MPI jobs or `pmemd.cuda` for GPU jobs.
- For GPU MD, a scheduler allocation that exposes one or more CUDA devices to the job process.

## User Input Required

**Required:** system name, structure directory, and one or more ranked structure numbers.

> **Structure Selection**
>
> `structures = [1]` is the normal single-structure workflow. Use `structures = [1, 2, 3]` to prepare and simulate multiple finalized structures in one submission. PyeDNA preserves the order in `structures`, rejects duplicates, and fails before preparation if any requested PDB is missing.

> **Restraint Usage**
>
> The default for every stage is `target = "none"`, meaning no positional restraints are applied. `target = "structure"` applies restraints to all non-solvent, non-ion residues. `target = "terminal"` applies restraints only to terminal DNA residues.

## Minimal Configuration Example For `md.toml`

```toml
[system]
name = "dna_CY3_CY5"
structure_directory = "structures"
structures = [1]

[amber]
dna_forcefield = "OL15"
dye_forcefield = "gaff2"
water_forcefield = "tip3p"
solvent_padding = 20.0
positive_ion = "Na+"
negative_ion = "Cl-"
neutralize = true

[workflow]
stages = ["prepare", "minimize", "equilibrate", "production"]

[simulation]
temperature = 300.0
pressure = 1.0
timestep = 0.002
cutoff = 8.0

[minimization]
engine = "pmemd"
cutoff = 8.0
ntmin = 1
max_steps = 1000
steepest_descent_steps = 500

[minimization.restraints.stage1]
target = "structure"
strength = 10.0

[minimization.restraints.stage2]
target = "none"

[equilibration]
engine = "auto"
cutoff = 8.0
heating_steps = 10000
npt_steps = 50000
npt_chunks = 1
npt_cleanup = "previous"
ntpr = 5000
ntwx = 5000
ntwr = 5000

[production]
engine = "auto"
cutoff = 8.0
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
| `name` | required | none | System basename used to resolve finalized structures and name Amber outputs. |
| `structure_directory` | optional | `"structures"` | Directory containing finalized unsolvated structures and `bonds.csv`. Relative paths resolve from the working directory. |
| `structures` | optional | `[1]` | Ranked finalized structure numbers to run. `1` resolves to `<structure_directory>/<name>_1.pdb`. Values must be unique positive integers and order is preserved. |
| `prmtop` | optional | none | Path to a ready Amber topology. When set, the `prepare` stage must be absent from `workflow.stages`; the file is copied into the run directory as `<name>.prmtop` and restraint masks are resolved from it. Use for systems built outside the structure workflow. Requires `rst7`. |
| `rst7` | optional | none | Coordinates matching `prmtop`; copied as `<name>.rst7`. Requires `prmtop`. The placeholder `<structure_directory>/<name>_<structure>.pdb` must still exist. |

### `[amber]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `dna_forcefield` | optional | `"OL15"` | DNA force-field identifier passed through PyeDNA's tleap source mapping. |
| `dye_forcefield` | optional | `"gaff2"` | Dye/linker force-field identifier; must match generated/library component files. |
| `water_forcefield` | optional | `"tip3p"` | Water model/force-field selector. `"tip3p"` sources `leaprc.water.tip3p` and uses `TIP3PBOX` for `solvateBox`. |
| `water_model` | optional | none | Legacy compatibility override for the solvent box selector. Prefer `water_forcefield`. |
| `solvent_padding` | optional | `20.0` | Padding passed to `solvateBox`. |
| `positive_ion` | optional | `"Na+"` | Positive ion name passed to `addIons`. |
| `negative_ion` | optional | `"Cl-"` | Negative ion name passed to `addIons`. |
| `neutralize` | optional | `true` | If true, `addIons mol <ion> 0` is called for both positive and negative ions. |

### `[workflow]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `stages` | optional | `["prepare", "minimize", "equilibrate", "production"]` | Ordered list containing any of `"prepare"`, `"minimize"`, `"equilibrate"`, and `"production"`. Later stages require outputs from earlier stages. |

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
| `engine` | optional | `"pmemd"` | Amber executable used for `min1` and `min2`. One of `"auto"`, `"pmemd"`, `"pmemd.MPI"`, `"pmemd.cuda"`, or `"sander"`. `"auto"` uses the runtime-selected workflow engine. |
| `cutoff` | optional | `[simulation].cutoff` | Nonbonded cutoff for `min1` and `min2`. |
| `ntmin` | optional | `1` | Amber minimization method control written to `min1` and `min2`. `1` uses the standard Amber minimizer; advanced users may set other Amber-supported values such as `3`. |
| `max_steps` | optional | `1000` | Total minimization cycles for each minimization substage. |
| `steepest_descent_steps` | optional | `500` | Steepest-descent cycles before switching minimizer. |

Minimization has two internal substages, `min1` and `min2`, configured by `[minimization.restraints.stage1]` and `[minimization.restraints.stage2]`.

### `[equilibration]`

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `engine` | optional | `"auto"` | Amber executable used for `eq1` and `eq2`. One of `"auto"`, `"pmemd"`, `"pmemd.MPI"`, `"pmemd.cuda"`, or `"sander"`. `"auto"` uses resource-dependent runtime selection. |
| `cutoff` | optional | `[simulation].cutoff` | Nonbonded cutoff for both equilibration substages unless `heating_cutoff` or `npt_cutoff` is set. |
| `heating_cutoff` | optional | `cutoff` or `[simulation].cutoff` | Nonbonded cutoff for `eq1`, the heating stage. |
| `npt_cutoff` | optional | `cutoff` or `[simulation].cutoff` | Nonbonded cutoff for `eq2`, the NPT equilibration stage. |
| `heating_steps` | optional | `10000` | Steps for `eq1`, the heating stage. |
| `npt_steps` | optional | `50000` | Steps for `eq2`, the NPT equilibration stage. |
| `npt_chunks` | optional | `1` | Number of restart chunks used to run the total `npt_steps`. Must evenly divide `npt_steps`. Values greater than 1 rebuild Amber GPU grid cells between chunks, which is useful when `pmemd.cuda` stops because periodic box dimensions changed too much during NPT density relaxation. |
| `npt_cleanup` | optional | `"previous"` | Cleanup policy for temporary `eq2_###_<name>` chunk files. `"previous"` removes a completed chunk only after the following chunk succeeds, keeping storage low while preserving the restart needed if the current chunk fails. `"all"` removes chunk files after the final canonical `eq2_<name>.ncrst` and `eq2_<name>.out` are written. `"none"` keeps chunk files during the run and on failures; completed runs are still subject to `[output].cleanup`. |
| `ntpr` | optional | `5000` | Equilibration log/energy output interval. |
| `ntwx` | optional | `5000` | Equilibration trajectory output interval. |
| `ntwr` | optional | `5000` | Equilibration restart output interval. |

### `[production]`

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `engine` | optional | `"auto"` | Amber executable used for production. One of `"auto"`, `"pmemd"`, `"pmemd.MPI"`, `"pmemd.cuda"`, or `"sander"`. `"auto"` uses resource-dependent runtime selection. |
| `cutoff` | optional | `[simulation].cutoff` | Nonbonded cutoff for production. |
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
    md.toml
    manifest.toml
    structure_001/
        status.toml
        <system>.pdb
        <system>.prmtop
        <system>.rst7
        <system>_solvated.pdb
        tleap_amber.log
        min1_<system>.in
        prod_<system>.out
```

Each selected structure receives its own `structure_<NNN>/` directory, where `<NNN>` is the ranked structure number with zero padding. `structure_003/` therefore always means ranked structure 3, even if it appears first in `structures = [3, 1]`.

`manifest.toml` records the immutable mapping from input order to ranked structure number, input structure, and structure directory. Each worker updates only its own `status.toml`, whose state is one of `pending`, `running`, `completed`, or `failed`, and whose stage is one of `prepare`, `minimize`, `equilibrate`, or `production` while work is running.

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

Edit the `#SBATCH` resource lines in `jobs/md/do_md.sh` for the resources you want to allocate on your cluster. Keep structure selection in `md.toml`.

Serial Amber stages run through `srun`; CPU MPI stages run through `mpirun -np $SLURM_NTASKS`.

## CPU/GPU Resource Selection

The example [jobs/md/do_md.sh](../../jobs/md/do_md.sh) script requests SLURM resources, then runs the same command:

```bash
pyedna md run "$MD_CONFIG"
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

At runtime, PyeDNA checks scheduler-provided environment variables and selects the resource-dependent Amber executable used whenever a stage-group `engine` is set to `"auto"`:

| Runtime signal | Amber executable |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` exposes at least one CUDA device | `pmemd.cuda` |
| no visible CUDA device and `SLURM_NTASKS > 1` | `pmemd.MPI` |
| no visible CUDA device and `SLURM_NTASKS` is unset or `1` | `pmemd` |

PyeDNA runs selected structures from `[system].structures` one at a time. When the GPU backend is selected and multiple CUDA devices are visible, each structure is still run serially using the first visible GPU assigned to the process. CPU serial and CPU MPI modes also run selected structures one at a time; CPU MPI uses the allocated task count for each structure.

Each major MD block can override the executable:

| Config field | Default | Stages controlled | Meaning |
| --- | --- | --- | --- |
| `[minimization].engine` | `"pmemd"` | `min1`, `min2` | Serial CPU `pmemd` by default, even inside a GPU allocation, to keep initial relaxation conservative. |
| `[equilibration].engine` | `"auto"` | `eq1`, `eq2` | Resource-dependent executable from the runtime table above. |
| `[production].engine` | `"auto"` | `prod` | Resource-dependent executable from the runtime table above. |

Accepted engine values are `"auto"`, `"pmemd"`, `"pmemd.MPI"`, `"pmemd.cuda"`, and `"sander"`. `pmemd`, `pmemd.MPI`, and `pmemd.cuda` are resolved below `pmemd_home`; `sander` is resolved below `ambertools_home`. If a stage explicitly requests `pmemd.cuda` but no CUDA device is visible, PyeDNA prints a warning and stops before launching Amber so the job does not fail later with `cudaGetDeviceCount failed`. If `pmemd.MPI` is requested with only one visible SLURM task, PyeDNA warns and runs one MPI task.

Stage-specific cutoff fields override `[simulation].cutoff`:

| Config field | Applies to | Fallback |
| --- | --- | --- |
| `[minimization].cutoff` | `min1`, `min2` | `[simulation].cutoff` |
| `[equilibration].cutoff` | `eq1`, `eq2` | `[simulation].cutoff` |
| `[equilibration].heating_cutoff` | `eq1` | `[equilibration].cutoff`, then `[simulation].cutoff` |
| `[equilibration].npt_cutoff` | `eq2` | `[equilibration].cutoff`, then `[simulation].cutoff` |
| `[production].cutoff` | `prod` | `[simulation].cutoff` |

## Common Modifications Or Advanced Options

Set `[system].structures = [1]` for the single-structure case, or list multiple ranked structures for one submission. Use `workflow.stages` to rerun a subset only when required prepared inputs and restart files already exist in each structure directory or are otherwise available. Adjust `output.cleanup` to retain more or fewer runtime files.

For large solvated systems whose density changes substantially at the start of NPT equilibration, `pmemd.cuda` may stop with a message that periodic box dimensions changed too much and that the calculation should be restarted from the previous restart file. Use `[equilibration].npt_chunks` to make PyeDNA do that restart pattern automatically while keeping the same total `npt_steps`, restraints, cutoff, pressure target, and barostat `tau`. For example, `npt_steps = 100000` and `npt_chunks = 10` runs ten 10000-step GPU NPT chunks and writes the final restart as `eq2_<name>.ncrst` for production. A practical starting point for rapidly compressing systems is 10000 to 20000 steps per chunk. The default `npt_cleanup = "previous"` deletes old chunk files as soon as they are no longer needed for a safe restart; set it to `"none"` when debugging chunk-specific failures.

## Limitations / Troubleshooting

The GPU backend uses `pmemd.cuda`; selected structures are run serially even if multiple GPUs are visible. Minimization defaults to serial CPU `pmemd`, while equilibration and production default to `"auto"`. CPU scaling uses `pmemd.MPI` only when `SLURM_NTASKS > 1` through `"auto"` or when a stage explicitly sets `engine = "pmemd.MPI"`; allocating many CPU cores with `--cpus-per-task` but only one task should not be expected to provide efficient CPU scaling. `custom` restraint targets are not implemented. `equilibrate` requires `min_<name>.ncrst`; `production` requires `eq2_<name>.ncrst`.
