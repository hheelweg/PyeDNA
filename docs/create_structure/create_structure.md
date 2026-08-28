# Creating MD Input Structure (`create_structure`)

## Purpose

`create_structure` prepares dye-labeled DNA structures. It can generate or copy DNA, assemble dye-linker components from libraries, prepare HADDOCK3 docking inputs, process completed HADDOCK output, and prepare a selected model for Amber MD.

## What the Workflow Does

`prepare` prepares DNA, assembles requested dye-linker intermediates, creates HADDOCK dye instances with unique segment IDs, writes ligand topology/parameter files, removes DNA residues occupied by dyes from the HADDOCK DNA input, writes distance restraints for intended covalent connections, and renders `docking_config.cfg`.

`dock` runs HADDOCK3 with the generated `docking_config.cfg` and writes HADDOCK output under `haddock/run/`.

`finalize` reads completed HADDOCK output, ranks models using the sum of selected CAPRI geometry columns (`vdw`, `elec`, `bonds`, `angles`, `dihe`, and `improper`), copies the top models into `structures/`, restores original atom/residue names, reinserts dye residues into the DNA template order, and writes final bond and residue-mapping metadata.

`amber` prepares one selected finalized structure for Amber using `tleap`; see [Amber setup](amber_setup.md).

## Prerequisites

- PyeDNA installed in the active Python environment.
- `libraries.dye_dir` and `libraries.linker_dir` set for `[[attachments]]` workflows.
- `libraries.dna_dir` set when `dna.source = "library"`.
- NAB available through the runtime config when `dna.source = "generate"`.
- HADDOCK3 available in the active environment when running `pyedna structure dock`.
- ACPYPE available for preparing HADDOCK/CNS topology and parameter files.
- PyeDNA runtime configuration with `amber.ambertools_home` pointing to AmberTools for internal dye-linker assembly and final Amber setup.

## User Input Required

**Required:** system name, DNA source, DNA name, and one or more dye placements or attachments.

For the current main workflow, use `[[attachments]]` with an existing dye, existing linker, and the DNA residue to replace.

> **Attachment and DNA Residue Replacment**
>
> Each `[[attachment]]` contains one `dye`, loaded as a template from `libraries.dye_dir`, and one `linker` specification, loaded as templates from `libraries.linker_dir`.
> Note that each linker comes with a 3' and 5' end, i.e. when respcifying in `[[attachment]]` the `residue` to replace in the DNA structure is effectively getting replaced by *three* formal residues (one dye, two linkers), which will affect the residue indexing in the final `.pdb` structure we generate here.
> **Important**: We can only load dyes and linkers whose name is existent in `libraries.dye_dir` or `libraries.linker_dir`, repsectively. Also be careful adjusting `[forcefield].attachments` and `[forcefield].dna` accordingly.

The DNA can currently be loaded as a template from `libraries.dna_dir` OR actually be generated with the a simple run of the Nucleid Acid Builder ([NAB](https://github.com/Amber-MD/AmberClassic.git)).

> **Generating DNA structures and `libraries.dna_dir`**
>
> PyeDNA currently can only generate very simple DNA structures directly via NAB.
> Therefore, we also supply access to a manually constructed `libraries.dna_dir` with raw DNA `.pdb` files that can be used.
> - [ ] TODO : We want to streamline and automatize this workflow in the future.

The core engine for building reasonable input structures for the DNA/dye attachements, we use [HADDOCK](https://github.com/haddocking/haddock3). 
More information on the precise implementation if Haddock into PyeDNA and HADDOCK-specific parameters can be found [here](../concepts/haddock3.md). 

> **Haddock3**
>
> - [ ] TODO : Explain the scientific assumptions behind HADDOCK attachment restraints and how users should judge whether the generated restraints match the intended covalent connectivity.
> - [ ] TODO : Explain how users should choose among HADDOCK models beyond the current implementation's geometry-score ranking.
> - [ ] TODO : Explain how users can manually change supported `[docking.overrides.*]` parameters and, for developer workflows, modify the packaged HADDOCK template.


## Minimal Configuration Example For `structure.toml`

```toml
[system]
name = "dna_CY3_CY5"

[dna]
source = "generate"
name = "dna"
sequence = "TGCACTCTCGATTTATGACCGAGCT"
type = "double_helix"

[[attachments]]
dye = "CY3"
linker = "PP"
residue = 10

[[attachments]]
dye = "CY5"
linker = "DE"
residue = 11

[forcefield]
dna = "OL15"
attachments = "gaff2"
water = "tip3p"

[docking]
engine = "haddock3"
top_models = 5

[amber]
model = 1
solvent_padding = 20.0
positive_ion = "Na+"
negative_ion = "Cl-"
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
| `name` | required | none | Output DNA basename. For library input, PyeDNA copies `<libraries.dna_dir>/<name>.pdb`. |
| `sequence` | required for `source = "generate"` | none | DNA sequence inserted into the NAB template. |
| `type` | required for `source = "generate"` | none | Current generated-DNA implementation supports `"double_helix"`. |

The DNA-input portion is functional but may be streamlined further in the future. Generated DNA uses the NAB template in `data/dna_templates/double_helix.nab`; library DNA is copied from `libraries.dna_dir` and then normalized to chain/segment ID `A`.

### `[[attachments]]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `dye` | required | none | Existing dye code/name in `libraries.dye_dir`. |
| `linker` | required | none | Existing linker code/name in `libraries.linker_dir`. |
| `residue` | required | none | DNA residue index to replace with the dye-linker component. |

PyeDNA converts each attachment into an internal dye placement named `<dye>_<linker>` with one site.


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

### `[forcefield]`

This is the preferred user-facing place to select force fields for `create_structure`.

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `dna` | optional | `"OL15"` | DNA force-field identifier. Compact values such as `"OL15"` are expanded internally to the corresponding `tleap` source. |
| `attachments` | optional | `"gaff2"` | Dye/linker force-field identifier. This must match the library layout under `libraries.dye_dir` and `libraries.linker_dir`. |
| `water` | optional | `"leaprc.water.tip3p"` | Water force-field source or compact water identifier. `"tip3p"` is expanded internally to `leaprc.water.tip3p`. |

Do not use `forcefield.components`; the parser will reject it and ask for `forcefield.attachments`.

### `[amber]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `model` | optional | `1` | Selected model number from `structures/<system.name>_<model>.pdb`; must be at least 1 and cannot exceed `docking.top_models`. |
| `output_name` | optional | `system.name` | Basename for final Amber files. |
| `water_model` | optional | `"TIP3P"` | Solvent box model. Current implementation supports `"TIP3P"`. |
| `solvent_padding` | optional | `20.0` | Padding passed to `solvateBox`. |
| `positive_ion` | optional | `"Na+"` | Positive ion name passed to `addIons`. |
| `negative_ion` | optional | `"Cl-"` | Negative ion name passed to `addIons`. |
| `neutralize` | optional | `true` | If true, `addIons mol <ion> 0` is called for both positive and negative ions. |

Advanced/internal aliases `amber.dna_forcefield`, `amber.dye_forcefield`, and `amber.water_forcefield` are also accepted by the current parser because `[forcefield]` values are mapped onto those internal settings. Prefer `[forcefield]` in user-written `structure.toml` files.

### `[workflow]`

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `prepare_amber` | optional | `false` | If true, `finalize` immediately runs Amber preparation after processing HADDOCK results. Must be boolean. |

## Generated Outputs

Important outputs include:

- `docking_config.cfg`
- `haddock/<dna>_haddock.pdb`
- `haddock/bond_restraint.tbl`
- `haddock/bonds.csv`
- `haddock/run/` from `pyedna structure dock`
- `structures/<system>_<n>.pdb`
- `structures/bonds.csv`
- `resid_mapping.json`
- final Amber outputs from the `amber` stage

## How To Run The Workflow

Run the structure stages:

```bash
pyedna structure prepare structure.toml
pyedna structure dock structure.toml
pyedna structure finalize structure.toml
pyedna structure amber structure.toml
```

If the config filename is omitted, PyeDNA uses `structure.toml` in the current directory:

```bash
pyedna structure prepare
```

On HPC systems, scheduler scripts may wrap these CLI commands, for example to submit HADDOCK or MD work to the appropriate queue.

## Common Modifications Or Advanced Options

Use `[docking.overrides.*]` only for HADDOCK parameters that are present in the PyeDNA template defaults. Use `[workflow].prepare_amber = true` when finalization should immediately generate Amber inputs for the selected model.

## Limitations / Troubleshooting

Generated DNA currently supports `double_helix` only. `[[attachments]]` requires matching dye/linker library entries and a manually curated DNA-linker compatibility FRCMOD. HADDOCK finalization requires `haddock/run/4_caprieval/capri_ss.tsv` and selected flexref model PDB files.
