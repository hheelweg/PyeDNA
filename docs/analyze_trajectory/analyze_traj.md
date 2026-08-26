# analyze_traj

## Purpose

`analyze_traj` analyzes an Amber trajectory using the hierarchy:

```text
trajectory
    -> attachments
    -> groups
    -> classical and/or quantum calculations
```

## Prerequisites

- Amber topology file.
- Amber NetCDF trajectory file.
- `DYE_DIR` set so analysis can read dye MOL2 charge data and `.attach` metadata.
- `resid_mapping.json` available in the working directory when attachment residues need to map back to Amber dye residues.
- PySCF/GPU4PySCF or ORCA backend resources when quantum jobs are requested.

## User Input Required

**Required:** trajectory files, frame interval, dye attachments, group definitions, and requested calculations.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should choose `[[attachments]]` residues, especially how `dna_residue` values from `structures/resid_mapping.json` relate to trajectory residue numbering.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should group attachments into donors, acceptors, dimers, or other scientifically meaningful units for classical and quantum calculations.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should choose quantum method, backend, charge, spin, states, outputs, and coupling state pairs for their scientific question.

## Minimal Configuration Example

```toml
[trajectory]
run_directory = "md/run_2026_01_01_12_00"
topology_file = "example_system.prmtop"
trajectory_file = "example_system.nc"
frame_interval = [0, 10]
optimize_caps = false
basis = "6-31g"

[[attachments]]
dye = "EXD"
residue = 4
cap = "H"

[[groups]]
name = "donor"
attachments = [4]

[[classical]]
group = "donor"
outputs = ["center_of_geometry", "radius_of_gyration"]

[analysis]
output_root = "analysis"
name = "example_analysis"

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

### Interactions

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

## What The Workflow Does

PyeDNA loads the topology and trajectory, validates the requested frame interval, creates an output run directory, copies the config, and writes a manifest. For each frame, it extracts capped dye snapshots for each attachment, builds configured groups, runs classical jobs, runs quantum jobs, computes quantum and classical interactions, and appends JSONL output records.

## Generated Outputs

The default output directory is:

```text
analysis/analysis_YYYY_MM_DD_HH_MM/
```

Outputs include:

- `traj.toml`
- `manifest.json`
- `quantum.jsonl`
- `classical.jsonl`
- `quantum_interactions.jsonl`
- `classical_interactions.jsonl`

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
