# Analyze Amber Trajectory (`analyze_traj`)

## Purpose

`analyze_traj` analyzes an Amber trajectory using the hierarchy in order to post-process especially geneated ensembles of dye molecules relative to one another classically as well as quantum-mechanically.

## What the Workflow Does

PyeDNA loads one or more Amber topology/trajectory pairs, validates the requested frame interval for each trajectory, creates an output run directory, copies the config, and writes a manifest. For each frame, it extracts capped dye snapshots for each attachment, builds configured groups, runs classical jobs, runs quantum jobs, computes quantum and classical interactions, and appends compact JSONL rows described by schemas in `manifest.json`.

## Prerequisites

- An MD run directory from `pyedna md run`, including `manifest.toml` and one `structure_<NNN>/` directory per selected structure.
- Alternatively, an explicit Amber topology file and Amber NetCDF trajectory file for legacy single-trajectory analysis.
- `libraries.dye_dir` set so analysis can read dye MOL2 charge data and `.attach` metadata.
- `resid_mapping.json` available in the working directory when attachment residues need to map back to Amber dye residues. This file is produced by the `finalize` stage of `create_structure` as `./resid_mapping.json`.
- PySCF when quantum jobs are requested with the PySCF backend.
- Optional: CuPy and GPU4PySCF for GPU-accelerated PySCF execution when a CUDA GPU is visible to the job.

## User Input Required

**Required:** an MD run directory and selected structures, or explicit trajectory files; a frame interval; dye attachments; group definitions; and requested calculations.

We first need to make sure `analyze_traj` reads in the `[[attachments]]` properly that have been done initially when creating the DNA/dye structure from some `structure.toml`.

> **Loading Attachment Information**
>
> In order to load information about dyes attached to the DNA structure, one needs to mirror the same structure of `[[attachments]]` as used for `structure.toml` in the `create_structure` workflow, i.e. chose the `libraries.dye_dir` codename for the `dye` and **importantly**, as `residue` the DNA residue from the `structure.toml` that we have used to attach the dye in the original (raw) DNA `.pdb` file (e.g. from `libraries.dna_dir`).
> **Note**: The residue IDs of the initial (raw) DNA `.pdb` and the Amber MD input `.pdb` differ by the simple reason that one nucleotide is replaces by one dye and two linker residues, i.e. each `[[attachment]]` in `structure.toml` increases the number of total residues by 2. 
> The mapping file that handles this confidently is `./resid_mapping.json`, so the user should **not delete** this file.

In order to perform analysis and computation on multiple dyes at once (most prominently if we want to do computations on a dye-dimer of neighboring dye molecules that are very close in space) one can group different residues, again specified by their residue IDs in `[[attachments]]`, to `[[groups]]`

> **Defining Groups**
>
> If one wants to perform computations on individual dyes molecules, one can define a group with `attachements = [resid]`.
> If one wants the group to include multiple dye molecules, one can specify e.g. `attachements = [resid1, resid2]`.
> The group `name` is important to reference with computation/analysis is supposed to be performed on the groups

Calculations are typically being performed **only** on pre-defined `[[groups]]`. Once such groups have been computed, one can decide which types of analyses one wants to perform on this group.

> **Specifying Computations**
>
> Analyses on `[[groups]]` can be either of classical (see blocks `[[classical]]`) or quantum (see blocks `[[quantum]]`) nature.
> Refer to the below specified keywords in order to see what exactly this can entail. Importantly, one needs to specify the correct group `name`. 

If one wants to back out quantities that emerge from computation outcomes between different `[[groups]]` one can use `[[interactions]]`.
The term interaction does *not* refer to some actual physcial interaction between groups but more to quantities that only make sense between different groups, e.g. electronic coupling and/or center-of-mass/geometry distance. 

> **Getting Group-to-Group Quantities (`[[interactions]]`)**
>
> There are both group-to-group computations for classical (see blocks `[[classical_interactions]]`) and quantum quantities (see blocks `[[quantum_interactions]]`) implemented. See documentation in table below for more details.
> One needs to specify `groups = [group_name1, group_name2]` in order to define which groups to consider. 



## Minimal Configuration Example For `traj.toml`:

```toml
[trajectory]
run_directory = "md/run_2026_01_01_12_00"
structures = [1, 2]
frame_interval = "all"
frame_stride = 1
optimize_caps = false

[[attachments]]
dye = "CY3"
residue = 10
cap = "H"

[[attachments]]
dye = "CY5"
residue = 11
cap = "H"

[[groups]]
name = "donor"
attachments = [10]

[[groups]]
name = "acceptor"
attachments = [11]

[[classical]]
group = "donor"
outputs = ["center_of_geometry", "plane_deviation"]

[qm_defaults]
method = "tddft"
basis = "sto-3g"
nstates = 1
outputs = ["excited_state_energies", "oscillator_strengths", "strongest_state"]
verbosity = 0

[[quantum]]
group = "donor"

[[quantum]]
group = "acceptor"

[[quantum_interactions]]
type = "coupling"
groups = ["donor", "acceptor"]
method = "tdm"
coupling_type = "electronic"
state_pairs = [["strongest", "strongest"], [0, 0]]

[[classical_interactions]]
type = "distance"
groups = ["donor", "acceptor"]
method = "center_of_geometry"

[[classical_interactions]]
type = "axis_angle"
groups = ["donor", "acceptor"]

[[classical_interactions]]
type = "orientation_factor"
groups = ["donor", "acceptor"]
method = "center_of_geometry"

[[classical_interactions]]
type = "plane_angle"
groups = ["donor", "acceptor"]

[analysis]
output_root = "analysis"

[analysis.units]
energy = "eV"
coupling = "eV"
distance = "angstrom"

[quantum_scheduler]
device = "auto"
parallel = true
```

## Configuration Reference

### `[trajectory]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `run_directory` | required | none | MD run directory, resolved relative to the current working directory. In structure-selection mode it must contain the MD `manifest.toml`. |
| `structures` | conditionally required | none | Ranked finalized structure numbers to analyze from the MD run. Use this instead of `topology_file` and `trajectory_file`. Values must be unique positive integers. Each value resolves through `manifest.toml` to `structure_<NNN>/<system>.prmtop` and `structure_<NNN>/<system>.nc`. |
| `topology_file` | conditionally required | none | Legacy single-trajectory mode only. Amber topology file, resolved relative to the current working directory; if absent there, PyeDNA also checks inside `run_directory`. |
| `trajectory_file` | conditionally required | none | Legacy single-trajectory mode only. Trajectory file within `run_directory`. |
| `frame_interval` | required | none | Either `"all"` or two integers `[initial_frame, final_frame]`, inclusive. `"all"` analyzes frames `0` through `num_frames - 1` for each selected trajectory. Explicit starts must be non-negative and explicit final frames must be within each selected trajectory. The same selection is applied independently to every structure trajectory. |
| `frame_stride` | optional | `1` | Positive integer stride for frame iteration within `frame_interval`. `1` analyzes every selected frame; `10` analyzes every tenth frame such as `0, 10, 20, ...` for `frame_interval = "all"`. |
| `optimize_caps` | optional | `[qm_defaults].optimize_caps`, else `false` | If true, cap atoms appended to extracted dye snapshots are optimized with constrained DFT. |
| `basis` | optional | `[qm_defaults].basis`, else `"6-31g"` | Basis used when building PySCF molecules for caps/groups. |

The preferred MD-run mode is:

```toml
[trajectory]
run_directory = "md/run_2026_01_01_12_00"
structures = [1, 2]
frame_interval = "all"
frame_stride = 1
```

To analyze every tenth available frame:

```toml
[trajectory]
run_directory = "md/run_2026_01_01_12_00"
structures = [1, 2]
frame_interval = "all"
frame_stride = 10
```

To analyze every fifth frame between explicit inclusive endpoints:

```toml
[trajectory]
run_directory = "md/run_2026_01_01_12_00"
structures = [1, 2]
frame_interval = [20, 100]
frame_stride = 5
```

Legacy explicit-file mode remains supported:

```toml
[trajectory]
run_directory = "md/run_2026_01_01_12_00"
topology_file = "dna_CY3_CY5.prmtop"
trajectory_file = "dna_CY3_CY5.nc"
frame_interval = "all"
frame_stride = 1
```

### `[[attachments]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `dye` | required | none | Dye name used to load `<libraries.dye_dir>/<dye>/gaff2/<dye>.mol2` and `<libraries.dye_dir>/<dye>/<dye>.attach`. |
| `residue` | required | none | Unique attachment residue identifier. Groups reference attachments by this integer. |
| `cap` | optional | `"H"` | Cap used when cutting the dye from the trajectory. Supported values are `"H"` and `"CH3"`; normalized to uppercase. |

Duplicate attachment residues are not allowed.

When the trajectory comes from the structure and MD workflows, the `pyedna structure finalize` stage writes `resid_mapping.json` in the working directory. That file records how the `residue` values specified in `structure.toml` under `[[attachments]]` map onto the final Amber residue numbering after each DNA residue is replaced by dye/linker residues. Use this mapping when deciding which attachment residues to list in `traj.toml`.

### `[[groups]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `name` | required | none | Unique group name. |
| `attachments` | required | none | Non-empty list of attachment residue IDs defined in `[[attachments]]`. |

Groups are built by combining the capped snapshot molecules for the listed attachments.

### `[[classical]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `group` | required | none | Existing group name. |
| `outputs` | optional | none | Supported values are `center_of_geometry`, `center_of_mass`, `radius_of_gyration`, and `plane_deviation`. `plane_deviation` requires the group to contain exactly one attachment and writes `plane_rmsd`. |

### `[[quantum]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `group` | required | none | Existing group name. |
| `method` | required | none | `"dft"` or `"tddft"`. |
| `backend` | optional | backend-specific default | `"pyscf"` or `"orca"`. |
| `basis` | optional | backend-specific default | Quantum basis string. |
| `xc` | optional | backend-specific default | DFT exchange-correlation functional. |
| `charge` | optional | inferred from built group unless backend/job overrides | Integer molecular charge. |
| `spin` | optional | backend default | Integer spin. |
| `nstates` | optional | backend default | Positive integer number of states. |
| `state_ids` | optional | backend default | Contiguous zero-based state IDs such as `[0, 1, 2]`. |
| `outputs` | optional | `[]` | Supported outputs include `energies`, `excited_state_energies`, `excitation_energies`, `oscillator_strengths`, `transition_dipoles`, `transition_quadrupoles`, `transition_density_matrices`, `tdm`, `strongest_state`, `mulliken`, `mulliken_populations`, `mulliken_charges`, `opa`, and `orbital_participation`. |
| `gpu`, `density_fit`, `tda`, `singlet` | optional | backend-specific defaults | Boolean quantum settings. `gpu` is deprecated; CPU/GPU execution is selected from visible runtime resources. |
| `scf_cycles`, `verbosity` | optional | backend-specific defaults | Integer quantum settings. |

Values in `[qm_defaults]` are copied into each `[[quantum]]` job unless that job sets the field directly. `optimize_caps` is also accepted here as a trajectory construction default, but is not copied into individual `[[quantum]]` jobs. Legacy `[quantum_defaults]` is still accepted as an alias, but do not define both tables in one config.

### `[qm_defaults]`

Use `[qm_defaults]` for quantum settings that should apply to all `[[quantum]]` jobs, such as `backend`, `basis`, `xc`, `density_fit`, `tda`, `singlet`, `nstates`, `scf_cycles`, or `verbosity`. Job-specific values in an individual `[[quantum]]` block override the defaults. `basis` and `optimize_caps` can also provide defaults for capped molecule and group construction unless `[trajectory]` sets them explicitly.

Do not set `gpu` in `[qm_defaults]`. CPU/GPU execution is controlled by `[quantum_scheduler].device`.

```toml
[qm_defaults]
backend = "pyscf"
basis = "6-31g"
xc = "b3lyp"
density_fit = true
optimize_caps = false
```

### Interactions

Interactions are quantities computed between two groups, such as a distance between two dye groups or an electronic coupling between two quantum-calculated groups. In this context, "interaction" does not mean a force-field nonbonded interaction term; it means a requested group-to-group analysis result.

`[[interactions]]` is accepted as a generic legacy-style table and is normalized into quantum or classical interactions based on `type`.

| Table | Type | Required fields | Optional fields |
| --- | --- | --- | --- |
| `[[classical_interactions]]` | `"distance"` | exactly two `groups` or at least two `attachments` | `method = "center_of_geometry"` or `"center_of_mass"` |
| `[[classical_interactions]]` | `"axis_angle"` | exactly two `groups`; each group must contain exactly one attachment | `method = "axis"` |
| `[[classical_interactions]]` | `"orientation_factor"` | exactly two `groups`; each group must contain exactly one attachment | `method = "center_of_geometry"` or `"center_of_mass"` |
| `[[classical_interactions]]` | `"plane_angle"` | exactly two `groups`; each group must contain exactly one attachment | `method = "plane"` |
| `[[quantum_interactions]]` | `"coupling"` | exactly two `groups` or at least two `attachments` | `method = "tdm"`, `state_pairs`, `coupling_type = "electronic"`, `"cJ"`, or `"cK"` |

Interactions must define exactly one of `groups` or `attachments`. `axis_angle`, `orientation_factor`, and `plane_angle` currently support `groups` only, and each referenced group must contain exactly one dye attachment. `plane_deviation` has the same one-attachment group restriction under `[[classical]]`. Coupling interactions can request state pairs containing non-negative integers or `"strongest"`.

For geometry-dependent analyses such as `axis_angle`, `orientation_factor`, `plane_deviation`, and `plane_angle`, each dye-library entry must manually define optional geometry metadata when the analysis is requested:

```toml
# <libraries.dye_dir>/<DYE>/geometry.toml
[axis]
atoms = ["ATOM1", "ATOM2"]

[plane]
atoms = ["ATOM1", "ATOM2", "ATOM3"]
```

`[axis].atoms` must contain exactly two atom names. `[plane].atoms`, when present, must contain at least three atom names. Three non-collinear atoms define a plane exactly, so `plane_deviation` gives `plane_rmsd` approximately zero for a three-atom plane up to floating-point error. The `axis_angle` value is an undirected molecular-axis angle in degrees, computed from `acos(abs(dot(axis1, axis2)))`, so antiparallel axes give `0` degrees. The `plane_angle` value is also undirected and uses `acos(abs(dot(normal1, normal2)))`, so it lies between `0` and `90` degrees.

For `orientation_factor`, PyeDNA treats the user-specified dye `[axis]` as a classical proxy for the dye transition-dipole direction. This is a geometry-based approximation unless that molecular axis has independently been shown to correspond to the actual optical transition dipole. For donor and acceptor axes `mu_D` and `mu_A`, and the donor-to-acceptor unit vector `R`, PyeDNA computes `kappa = mu_D dot mu_A - 3 (mu_D dot R)(mu_A dot R)` and writes both signed `kappa` and `kappa_squared`. The center vector is defined using `method = "center_of_geometry"` by default, or `method = "center_of_mass"` when requested. If the donor and acceptor centers coincide, the calculation fails clearly.

For `plane_deviation`, PyeDNA selects the `[plane].atoms` for the single dye in the group, fits a best-fit plane using SVD, and writes `plane_rmsd = sigma_3 / sqrt(N)` in Angstrom to `classical.jsonl`. For `plane_angle`, PyeDNA fits each group's plane normal from `[plane].atoms` and writes the undirected angle in degrees to `classical_interactions.jsonl`.

### `[analysis]`, `[analysis.units]`, `[analysis.save]`, `[output]`, and `[quantum_scheduler]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `[analysis].output_root` | optional | `"analysis"` | Root output directory. |
| `[analysis].name` | optional | `"auto"` | Output run directory name; `"auto"` creates a timestamped name. |
| `[analysis.units].energy` | optional | `"eV"` | One of `"hartree"`, `"au"`, `"e_h"`, `"eV"`, or `"cm-1"`; validation is case-insensitive for supported units. |
| `[analysis.units].coupling` | optional | `"cm-1"` | Same supported values as energy. |
| `[analysis.units].distance` | optional | `"angstrom"` | One of `"angstrom"`, `"a"`, `"bohr"`, or `"nm"`. |
| `[analysis.save].save_intermediates` | optional | backend-specific behavior | Must be boolean if present. |
| `[output].quantum_file` | optional | `"quantum.jsonl"` | Quantum results filename, used only when `[[quantum]]` requests at least one output column. |
| `[output].classical_file` | optional | `"classical.jsonl"` | Classical results filename, used only when `[[classical]]` requests at least one output column. |
| `[output].quantum_interactions_file` | optional | `"quantum_interactions.jsonl"` | Quantum interaction results filename, used only when `[[quantum_interactions]]` requests at least one output column. |
| `[output].classical_interactions_file` | optional | `"classical_interactions.jsonl"` | Classical interaction results filename, used only when `[[classical_interactions]]` requests at least one output column. |
| `[output].interaction_file` | optional | quantum interactions legacy alias | Used when `quantum_interactions_file` is absent. |
| `[quantum_scheduler].device` | optional | `"auto"` | `"auto"` uses GPU4PySCF for quantum jobs when CUDA GPUs are visible and CPU PySCF otherwise. `"cpu"` forces CPU PySCF even inside a GPU allocation. `"gpu"` requires at least one visible CUDA GPU. |
| `[quantum_scheduler].parallel` | optional | `false` | If true, independent quantum jobs within one frame may run concurrently. |
| `[quantum_scheduler].gpu_ids` | optional | inferred from visible GPUs | Deprecated advanced override. Non-empty list of visible GPU IDs. |
| `[quantum_scheduler].max_workers` | optional | GPU: visible GPU count; CPU: `1` | Positive integer advanced override for concurrent quantum worker processes. |

## How To Run The Workflow

Run the workflow directly with:

```bash
pyedna analysis trajectory traj.toml
```

If the config filename is omitted, PyeDNA uses `traj.toml` in the current directory:

```bash
pyedna analysis trajectory
```

On HPC systems, use the sample scheduler wrapper:

```bash
sbatch jobs/analysis/analyze_traj.sh traj.toml
```

Edit the `#SBATCH` resource lines in `jobs/analysis/analyze_traj.sh` for the resources you want to allocate on your cluster. Keep `traj.toml` focused on trajectory, group, classical, and quantum-analysis settings.

## CPU/GPU Resource Selection

`traj.toml` stores scientific analysis settings and can optionally pin the quantum device. It should not need legacy `[[quantum]].gpu = true` or `[quantum_scheduler].gpu_ids = [...]` just to mirror the SLURM allocation. The example [jobs/analysis/analyze_traj.sh](../../jobs/analysis/analyze_traj.sh) script requests SLURM resources, then runs the same command:

```bash
pyedna analysis trajectory "$@"
```

Choose CPU or GPU execution by changing the script's `#SBATCH` resource lines:

```bash
# CPU
#SBATCH --cpus-per-task=16
```

```bash
# GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
```

At runtime, PyeDNA inspects `CUDA_VISIBLE_DEVICES` and SLURM variables such as `SLURM_CPUS_PER_TASK`, `SLURM_JOB_GPUS`, `SLURM_GPUS`, and `SLURM_GPUS_ON_NODE`. If no CUDA GPU is visible, DFT/TDDFT quantum jobs run with plain CPU PySCF. If one or more GPUs are visible, the default `device = "auto"` uses GPU4PySCF for PySCF-backed DFT/TDDFT.

Use explicit CPU mode for debugging or for systems where GPU4PySCF is unavailable or numerically unstable:

```toml
[quantum_scheduler]
device = "cpu"
parallel = false
```

Use one GPU for the most conservative GPU run:

```toml
[quantum_scheduler]
device = "gpu"
parallel = false
```

For quantum analysis, `[quantum_scheduler].parallel = true` allows independent quantum jobs within one frame to run concurrently. In GPU mode, the default is one spawned quantum worker per visible GPU, with each worker receiving one `CUDA_VISIBLE_DEVICES` token before GPU4PySCF is imported. In CPU mode, the default is one quantum worker so each PySCF calculation can use the allocated CPU threads without accidental oversubscription. Use `[quantum_scheduler].max_workers` only as an advanced override.

Two-GPU quantum-job parallelism can be requested with:

```toml
[quantum_scheduler]
device = "gpu"
parallel = true
max_workers = 2
```

## Generated Outputs

The default output directory is:

```text
analysis/analysis_YYYY_MM_DD_HH_MM/
```

For an MD run with `structures = [1, 2]`, outputs are grouped by structure. Only requested result-family files are generated; this example shows all four families present:

```text
analysis/analysis_YYYY_MM_DD_HH_MM/
    traj.toml
    manifest.json
    structure_001/
        classical.jsonl
        quantum.jsonl
        quantum_interactions.jsonl
        classical_interactions.jsonl
    structure_002/
        classical.jsonl
        quantum.jsonl
        quantum_interactions.jsonl
        classical_interactions.jsonl
```

Legacy explicit-file analysis writes the requested JSONL files directly in the analysis run directory.

Outputs include:

- `traj.toml`: copy of the analysis configuration used for the run.
- `manifest.json`: authoritative run metadata, trajectory input paths, selected structure mappings, units, output file paths for active result families, requested calculations, static calculation metadata, `analysis_format_version`, and ordered schemas for each result family. Families with only a `frame` schema column are inactive and do not create JSONL files.
- `classical.jsonl`: one JSON array per analyzed frame. The array columns are defined by `manifest.json` and aggregate all requested `[[classical]]` group outputs for that frame, such as `donor.center_of_geometry` or `donor.plane_rmsd`.
- `quantum.jsonl`: one JSON array per analyzed frame. The schema can include scalar, array, or object columns such as `donor.excited_state_energies`, `donor.transition_dipoles`, or structured fragment summaries. Static quantum request metadata is stored in `manifest.json`; runtime-static molecule metadata such as `atom_count`, `charge`, and `spin` is added to manifest runtime metadata when available.
- `classical_interactions.jsonl`: one JSON array per analyzed frame. The array aggregates all requested classical interaction outputs, such as `donor.acceptor.distance`, `donor.acceptor.axis_angle`, `donor.acceptor.plane_angle`, `donor.acceptor.kappa`, and `donor.acceptor.kappa_squared`.
- `quantum_interactions.jsonl`: one JSON array per analyzed frame. Coupling columns are keyed by groups and state pair, for example `donor.acceptor.state_0_0.coupling`.

The `.jsonl` files use JSON Lines format: each line is an independently readable JSON array. Each row has the same number and order of cells as the corresponding schema in `manifest.json`. Cell values may be scalars, nested arrays, structured objects, or `null` when a value is absent for that frame.

In MD-run structure mode, JSONL rows do not repeat `trajectory_index`, `structure`, or `structure_directory`. The root `manifest.json` records those values once for each `structure_<NNN>/` directory. When loading results with PyeDNA, those fields are reconstructed as in-memory columns so dataframes can still be filtered or grouped by structure. `trajectory_index` is the zero-based position in `[trajectory].structures`, so `structures = [2, 1]` loads structure 2 with `trajectory_index = 0` and structure 1 with `trajectory_index = 1`.

As a rough shape check, the number of rows is normally one row per analyzed frame in each generated result-family file. If a family is not requested, its JSONL file is omitted:

| File | Expected number of rows |
| --- | --- |
| `classical.jsonl` | number of analyzed frames; all `[[classical]]` outputs are columns in each row |
| `quantum.jsonl` | number of analyzed frames; all `[[quantum]]` outputs are columns in each row |
| `classical_interactions.jsonl` | number of analyzed frames; all `[[classical_interactions]]` outputs are columns in each row |
| `quantum_interactions.jsonl` | number of analyzed frames; requested state-pair couplings are columns in each row |

Nested arrays and dictionaries are kept as JSON values in the raw files. When loaded through PyeDNA's helper functions, schema columns become dataframe columns directly. See [Loading Analysis Results](loading_results.md).

> **Important**
>
> GPU trajectory quantum analysis requires the validated CUDA/CuPy/GPU4PySCF stack. CPU trajectory quantum analysis requires PySCF only.

The ORCA backend name is accepted by configuration validation, but the current ORCA backend raises `NotImplementedError` at runtime.

## Common Modifications Or Advanced Options

Use `[qm_defaults]` to avoid repeating backend, basis, functional, or TDDFT settings across quantum jobs. Use interactions to request derived distances or couplings between groups.

## Limitations / Troubleshooting

Attachment/group definitions are strict: groups reference attachment residues directly, and duplicate attachment residues are rejected. The analysis code currently loads dye charge and attach metadata from the `gaff2` dye library path.

## Migration Note

The current supported analysis entry point is TOML-driven `analyze_traj`. Older analysis scripts and ad hoc parameter formats have been removed in favor of `pyedna.analysis` and `pyedna.trajectory`.
