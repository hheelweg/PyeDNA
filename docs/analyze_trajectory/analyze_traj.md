# Analyze Amber Trajectory (`analyze_traj`)

## Purpose

`analyze_traj` analyzes an Amber trajectory using the hierarchy in order to post-process especially geneated ensembles of dye molecules relative to one another classically as well as quantum-mechanically.

## What the Workflow Does

PyeDNA loads the topology and trajectory, validates the requested frame interval, creates an output run directory, copies the config, and writes a manifest. For each frame, it extracts capped dye snapshots for each attachment, builds configured groups, runs classical jobs, runs quantum jobs, computes quantum and classical interactions, and appends JSONL output records.

## Prerequisites

- Amber topology file.
- Amber NetCDF trajectory file.
- `DYE_DIR` set so analysis can read dye MOL2 charge data and `.attach` metadata.
- `resid_mapping.json` available in the working directory when attachment residues need to map back to Amber dye residues. This file is produced by the `finalize` stage of `create_structure` as `./resid_mapping.json`.
- PySCF/GPU4PySCF or ORCA backend resources when quantum jobs are requested.

## User Input Required

**Required:** trajectory files, frame interval, dye attachments, group definitions, and requested calculations.

We first need to make sure `analyze_traj` reads in the `[[attachments]]` properly that have been done initially when creating the DNA/dye structure from some `structure.toml`.

> **Loading Attachment Information**
>
> In order to load information about dyes attached to the DNA structure, one needs to mirror the same structure of `[[attachments]]` as used for `structure.toml` in the `create_structure` workflow, i.e. chose the `DYE_DIR` codename for the `dye` and **importantly**, as `residue` the DNA residue from the `structure.toml` that we have used to attach the dye in the original (raw) DNA `.pdb` file (e.g. from `DNA_DIR`).
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
topology_file = "dna_CY3_CY5.prmtop"
trajectory_file = "dna_CY3_CY5.nc"
frame_interval = [0, 10]
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
outputs = ["center_of_geometry"]

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

[analysis]
output_root = "analysis"

[analysis.units]
energy = "eV"
coupling = "eV"
distance = "angstrom"
```

## Configuration Reference

### `[trajectory]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `run_directory` | required | none | Directory containing the trajectory file, resolved relative to the current working directory. |
| `topology_file` | required | none | Amber topology file, resolved relative to the current working directory. |
| `trajectory_file` | required | none | Trajectory file within `run_directory`. |
| `frame_interval` | required | none | Two integers `[initial_frame, final_frame]`, inclusive. Start must be non-negative and final must be within the trajectory. |
| `optimize_caps` | optional | `false` | If true, cap atoms appended to extracted dye snapshots are optimized with constrained DFT. |
| `basis` | optional | `"6-31g"` | Basis used when building PySCF molecules for caps/groups. |

### `[[attachments]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `dye` | required | none | Dye name used to load `DYE_DIR/<dye>/gaff2/<dye>.mol2` and `DYE_DIR/<dye>/<dye>.attach`. |
| `residue` | required | none | Unique attachment residue identifier. Groups reference attachments by this integer. |
| `cap` | optional | `"H"` | Cap used when cutting the dye from the trajectory. Supported values are `"H"` and `"CH3"`; normalized to uppercase. |

Duplicate attachment residues are not allowed.

When the trajectory comes from the `create_structure` -> `do_md` workflow, the `create_structure finalize` stage writes `structures/resid_mapping.json`. That file records how the `residue` values specified in `structure.toml` under `[[attachments]]` map onto the final Amber residue numbering after each DNA residue is replaced by dye/linker residues. Use this mapping when deciding which attachment residues to list in `traj.toml`.

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
| `outputs` | optional | none | Supported values are `axis_angle`, `center_of_geometry`, `center_of_mass`, and `radius_of_gyration`. |

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
| `gpu`, `density_fit`, `tda`, `singlet` | optional | backend-specific defaults | Boolean quantum settings. |
| `scf_cycles`, `verbosity` | optional | backend-specific defaults | Integer quantum settings. |

Values in `[quantum_defaults]` are copied into each `[[quantum]]` job unless that job sets the field directly.

### `[quantum_defaults]`

Use `[quantum_defaults]` for quantum settings that should apply to all `[[quantum]]` jobs, such as `backend`, `basis`, `xc`, `gpu`, `density_fit`, `tda`, `singlet`, `nstates`, `scf_cycles`, or `verbosity`. Job-specific values in an individual `[[quantum]]` block override the defaults.

```toml
[quantum_defaults]
backend = "pyscf"
basis = "6-31g"
xc = "b3lyp"
gpu = true
density_fit = true
```

### Interactions

Interactions are quantities computed between two groups, such as a distance between two dye groups or an electronic coupling between two quantum-calculated groups. In this context, "interaction" does not mean a force-field nonbonded interaction term; it means a requested group-to-group analysis result.

`[[interactions]]` is accepted as a generic legacy-style table and is normalized into quantum or classical interactions based on `type`.

| Table | Type | Required fields | Optional fields |
| --- | --- | --- | --- |
| `[[classical_interactions]]` | `"distance"` | exactly two `groups` or at least two `attachments` | `method = "center_of_geometry"` or `"center_of_mass"` |
| `[[quantum_interactions]]` | `"coupling"` | exactly two `groups` or at least two `attachments` | `method = "tdm"`, `state_pairs`, `coupling_type = "electronic"`, `"cJ"`, or `"cK"` |

Interactions must define exactly one of `groups` or `attachments`. Coupling interactions can request state pairs containing non-negative integers or `"strongest"`.

### `[analysis]`, `[analysis.units]`, `[analysis.save]`, `[output]`, and `[quantum_scheduler]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `[analysis].output_root` | optional | `"analysis"` | Root output directory. |
| `[analysis].name` | optional | `"auto"` | Output run directory name; `"auto"` creates a timestamped name. |
| `[analysis.units].energy` | optional | `"eV"` | One of `"hartree"`, `"au"`, `"e_h"`, `"eV"`, or `"cm-1"`; validation is case-insensitive for supported units. |
| `[analysis.units].coupling` | optional | `"cm-1"` | Same supported values as energy. |
| `[analysis.units].distance` | optional | `"angstrom"` | One of `"angstrom"`, `"a"`, `"bohr"`, or `"nm"`. |
| `[analysis.save].save_intermediates` | optional | backend-specific behavior | Must be boolean if present. |
| `[output].quantum_file` | optional | `"quantum.jsonl"` | Quantum results filename. |
| `[output].classical_file` | optional | `"classical.jsonl"` | Classical results filename. |
| `[output].quantum_interactions_file` | optional | `"quantum_interactions.jsonl"` | Quantum interaction results filename. |
| `[output].classical_interactions_file` | optional | `"classical_interactions.jsonl"` | Classical interaction results filename. |
| `[output].interaction_file` | optional | quantum interactions legacy alias | Used when `quantum_interactions_file` is absent. |
| `[quantum_scheduler].parallel` | optional | `false` | If true, defaults `gpu_ids` to `[0]` and `max_workers` to number of GPU IDs. |
| `[quantum_scheduler].gpu_ids` | optional | `[0]` when parallel | Non-empty list of integers. |
| `[quantum_scheduler].max_workers` | optional | `len(gpu_ids)` when parallel | Positive integer. |

## Generated Outputs

The default output directory is:

```text
analysis/analysis_YYYY_MM_DD_HH_MM/
```

Outputs include:

- `traj.toml`: copy of the analysis configuration used for the run.
- `manifest.json`: run metadata, trajectory input paths, units, output file paths, requested interactions, and quantum job summaries.
- `classical.jsonl`: one JSON object per classical result, with `frame`, `group`, and a `values` object containing requested quantities such as `center_of_geometry`, `center_of_mass`, or `radius_of_gyration`.
- `quantum.jsonl`: one JSON object per quantum result, with `frame`, `group`, `method`, `atom_count`, `charge`, `spin`, and, for TDDFT jobs, a nested `tddft` object containing requested outputs such as excited-state energies, oscillator strengths, transition dipoles, transition density matrices, or strongest-state information.
- `classical_interactions.jsonl`: one JSON object per classical group-to-group interaction result, with `frame`, `type`, `method`, `groups`, and a `values` object such as `distance`.
- `quantum_interactions.jsonl`: one JSON object per quantum group-to-group interaction result, with `frame`, `type`, `method`, `groups`, `state_pair`, and a `values` object containing coupling quantities.

The `.jsonl` files use JSON Lines format: each line is an independent JSON object. This makes the files easy to append during long analyses and straightforward to load into tabular tools later.

As a rough shape check, the number of records is normally:

| File | Expected number of records |
| --- | --- |
| `classical.jsonl` | number of analyzed frames x number of `[[classical]]` jobs |
| `quantum.jsonl` | number of analyzed frames x number of `[[quantum]]` jobs |
| `classical_interactions.jsonl` | number of analyzed frames x number of `[[classical_interactions]]` jobs |
| `quantum_interactions.jsonl` | number of analyzed frames x number of `[[quantum_interactions]]` jobs x number of requested `state_pairs`; one `[0, 0]` pair is used when `state_pairs` is omitted |

Nested arrays and dictionaries are kept as JSON values in the raw files. When loaded through PyeDNA's helper functions, scalar lists are flattened into numbered dataframe columns. See [Loading Analysis Results](loading_results.md).

## How To Run The Workflow

```bash
python "$PYEDNA_HOME/scripts/analyze_traj.py" traj.toml
```

The script also accepts:

```bash
python "$PYEDNA_HOME/scripts/analyze_traj.py" --config traj.toml
```

## Common Modifications Or Advanced Options

Use `[quantum_defaults]` to avoid repeating backend, basis, functional, GPU, or TDDFT settings across quantum jobs. Use interactions to request derived distances or couplings between groups.

## Limitations / Troubleshooting

Attachment/group definitions are strict: groups reference attachment residues directly, and duplicate attachment residues are rejected. The analysis code currently loads dye charge and attach metadata from the `gaff2` dye library path.

## Migration Note

The current supported analysis entry point is TOML-driven `analyze_traj`. Older analysis scripts and ad hoc parameter formats have been removed in favor of `pyedna.analysis` and `pyedna.trajectory`.
