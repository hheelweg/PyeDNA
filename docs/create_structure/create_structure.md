# create_structure

## Purpose

`create_structure` prepares dye-labeled DNA structures. It can generate or copy DNA, assemble dye-linker components from libraries, prepare HADDOCK3 docking inputs, process completed HADDOCK output, and prepare a selected model for Amber.

## Prerequisites

- `PYEDNA_HOME` set, because PyeDNA reads HADDOCK templates and helper scripts from the repository.
- `DYE_DIR` and `LNK_DIR` set for `[[attachments]]` workflows.
- `DNA_DIR` set when `dna.source = "library"`.
- NAB available through the configured environment when `dna.source = "generate"`.
- HADDOCK3 available in the `haddock` Conda environment used by `run_haddock.sh`.
- AmberTools available for internal dye-linker assembly and final Amber setup.

## User Input Required

**Required:** system name, DNA source, DNA name, and one or more dye placements or attachments.

For the current main workflow, use `[[attachments]]` with an existing dye, existing linker, and the DNA residue to replace.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should choose DNA attachment residues, including residue numbering assumptions and how many DNA residues a dye-linker component replaces.

> **AUTHOR INPUT REQUIRED**
>
> Explain when users should generate a simple NAB DNA structure versus provide an existing PDB from `DNA_DIR`, and what preparation is expected for library DNA structures.

> **AUTHOR INPUT REQUIRED**
>
> Explain the scientific assumptions behind HADDOCK attachment restraints and how users should judge whether the generated restraints match the intended covalent connectivity.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should choose among HADDOCK models beyond the current implementation's geometry-score ranking.

## Minimal Configuration Example

```toml
[system]
name = "example_system"

[dna]
source = "generate"
name = "example_dna"
sequence = "ATCGATCG"
type = "double_helix"

[[attachments]]
dye = "EXD"
linker = "EL"
residue = 4

[docking]
engine = "haddock3"
top_models = 5

[amber]
model = 1
output_name = "example_system"
dna_forcefield = "OL15"
dye_forcefield = "gaff2"
water_forcefield = "leaprc.water.tip3p"
water_model = "TIP3P"
solvent_padding = 20.0
neutralize = true
```

## Configuration Reference

### `[system]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `name` | required | none | Base name for selected structures and final Amber outputs. Legacy `[structure].name` is also accepted. |

### `[dna]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `source` | required | none | Must be `"generate"` or `"library"`. |
| `name` | required | none | Output DNA basename. For library input, PyeDNA copies `DNA_DIR/<name>.pdb`. |
| `sequence` | required for `source = "generate"` | none | DNA sequence inserted into the NAB template. |
| `type` | required for `source = "generate"` | none | Current generated-DNA implementation supports `"double_helix"`. |

The DNA-input portion is functional but may be streamlined further in the future. Generated DNA uses the NAB template in `data/dna_templates/double_helix.nab`; library DNA is copied from `DNA_DIR` and then normalized to chain/segment ID `A`.

### `[[attachments]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `dye` | required | none | Existing dye code/name in `DYE_DIR`. |
| `linker` | required | none | Existing linker code/name in `LNK_DIR`. |
| `residue` | required | none | DNA residue index to replace with the dye-linker component. |

PyeDNA converts each attachment into an internal dye placement named `<dye>_<linker>` with one site. Do not mix `[[attachments]]` with legacy `[[dyes]]`.

### Legacy `[[dyes]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `name` | required | none | Existing library dye name, or generated dye-linker name. |
| `sites` | required | none | Consecutive DNA residue indices occupied by the dye. Sites must not overlap across dyes. |

This shape remains accepted for existing workflows, but `[[attachments]]` is the current user-facing route for dye/linker systems.

### `[docking]` / `[haddock]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `engine` | optional | `"haddock3"` | Only `"haddock3"` is supported. |
| `top_models` | optional | `5` | Number of selected HADDOCK models to copy and reformat; must be at least 1. |
| `[docking.overrides.<section>]` | optional | none | HADDOCK template overrides. Accepted sections are `general`, `topoaa`, `rigidbody`, `seletop`, `flexref`, and `caprieval`; keys must match known template parameters. |

`[haddock]` and `[docking]` are merged, with `[docking]` values overriding `[haddock]` values.

### `[amber]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `model` | optional | `1` | Selected model number from `structures/<system.name>_<model>.pdb`; must be at least 1 and cannot exceed `docking.top_models`. |
| `output_name` | optional | `system.name` | Basename for final Amber files. |
| `dna_forcefield` | optional | `"OL15"` | DNA force-field identifier. |
| `dye_forcefield` | optional | `"gaff2"` | Dye/linker force-field identifier. |
| `water_forcefield` | optional | `"leaprc.water.tip3p"` | `tleap` water force-field source. |
| `water_model` | optional | `"TIP3P"` | Solvent box model. Current implementation supports `"TIP3P"`. |
| `solvent_padding` | optional | `20.0` | Padding passed to `solvateBox`. |
| `positive_ion` | optional | `"Na+"` | Positive ion name passed to `addIons`. |
| `negative_ion` | optional | `"Cl-"` | Negative ion name passed to `addIons`. |
| `neutralize` | optional | `true` | If true, `addIons mol <ion> 0` is called for both positive and negative ions. |

Legacy `[forcefield]` accepts `dna`, `attachments`, and `water`, which are mapped onto the corresponding Amber fields.

### `[workflow]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `prepare_amber` | optional | `false` | If true, `finalize` immediately runs Amber preparation after processing HADDOCK results. Must be boolean. |

## What the Workflow Does

`prepare` prepares DNA, assembles requested dye-linker intermediates, creates HADDOCK dye instances with unique segment IDs, writes ligand topology/parameter files, removes DNA residues occupied by dyes from the HADDOCK DNA input, writes distance restraints for intended covalent connections, and renders `docking_config.cfg`.

`finalize` reads completed HADDOCK output, ranks models using the sum of selected CAPRI geometry columns (`vdw`, `elec`, `bonds`, `angles`, `dihe`, and `improper`), copies the top models into `structures/`, restores original atom/residue names, reinserts dye residues into the DNA template order, and writes final bond and residue-mapping metadata.

`amber` prepares one selected finalized structure for Amber using `tleap`; see [Amber setup](amber_setup.md).

## Generated Outputs

Important outputs include:

- `docking_config.cfg`
- `haddock/<dna>_haddock.pdb`
- `haddock/bond_restraint.tbl`
- `haddock/bonds.csv`
- `haddock/run/` from HADDOCK3
- `structures/<system>_<n>.pdb`
- `structures/bonds.csv`
- `structures/resid_mapping.json`
- final Amber outputs from the `amber` stage

## How To Run The Workflow

Prepare inputs and submit HADDOCK:

```bash
bash "$PYEDNA_HOME/jobs/structure/create_structure.sh" structure.toml
```

Run an individual stage directly:

```bash
python "$PYEDNA_HOME/scripts/create_structure.py" prepare --config structure.toml
python "$PYEDNA_HOME/scripts/create_structure.py" finalize --config structure.toml
python "$PYEDNA_HOME/scripts/create_structure.py" amber --config structure.toml
```

## Common Modifications Or Advanced Options

Use `[docking.overrides.*]` only for HADDOCK parameters that are present in the PyeDNA template defaults. Use `[workflow].prepare_amber = true` when finalization should immediately generate Amber inputs for the selected model.

## Limitations / Troubleshooting

Generated DNA currently supports `double_helix` only. `[[attachments]]` requires matching dye/linker library entries and a manually curated DNA-linker compatibility FRCMOD. HADDOCK finalization requires `haddock/run/4_caprieval/capri_ss.tsv` and selected flexref model PDB files.
