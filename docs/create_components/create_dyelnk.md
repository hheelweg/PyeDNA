# Create Dye-Linker Composite (`create_dyelnk`)

## Purpose

`create_dyelnk` combines an existing dye definition with existing linker definitions into a reusable dye-linker attachment component. It does not parameterize a dye or linker from scratch; it loads component-library outputs and assembles them.

## What the Workflow Does

PyeDNA loads the dye MOL2 and attachment metadata, loads both linker variants and their `3CONNECT`/`5CONNECT` metadata, validates all referenced atom names, generates a low-clash assembled PDB using RDKit conformers for the linkers, writes a `tleap` input file, runs `tleap` to create a linked MOL2, and runs `parmchk2` to create a linked FRCMOD.

The molecular order in the assembled component is 5' linker, dye, then 3' linker.

## Prerequisites

- `DYE_DIR` containing the selected dye MOL2/FRCMOD/attach files.
- `LNK_DIR` containing the selected linker 3' and 5' MOL2/FRCMOD/attach files.
- A DNA-linker compatibility parameter file at `LNK_DIR/connect/<dye_forcefield>/<dna_forcefield>/connectparams.frcmod` or legacy `connectparms.frcmod`.
- AmberTools `tleap` and `parmchk2`.

## User Input Required

**Required:** existing dye name, existing linker name, and the force-field identifiers used to resolve library files.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should choose compatible dye/linker combinations and how to recognize whether a linker was parameterized for the intended dye and DNA force fields.

## Minimal Configuration Example for `dyelnk.toml`

```toml
[dyelnk]
dye = "CY3"
linker = "DE"
dye_forcefield = "gaff2"
dna_forcefield = "OL15"
```

This example uses the `CY3` dye and `DE` linker names from the sample `create_dye` and `create_linker` configurations. The corresponding component files must already exist in `DYE_DIR` and `LNK_DIR`.

## Configuration Reference

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `[dyelnk].dye` | required | none | Name/code of an existing dye in `DYE_DIR`. |
| `[dyelnk].linker` | required | none | Name/code of an existing linker in `LNK_DIR`. |
| `[dyelnk].dye_forcefield` | optional | `"gaff2"` | Library subdirectory identifier for the dye/linker force field. `leaprc.*` prefixes are normalized. |
| `[dyelnk].dna_forcefield` | optional | `"OL15"` | Library subdirectory identifier for the DNA force field. |


## Generated Outputs

The default direct workflow writes:

- `<dye>_<linker>_assembled.pdb`
- `tleap_dyelnk.in`
- `<dye>_<linker>_linked.mol2`
- `<dye>_<linker>_linked.frcmod`
- `<dye>_<linker>_linked.parmchk2.log`

When called internally by `create_structure`, files are named with the attachment name and may be removed after Amber setup.

## How To Run

```bash
bash "$PYEDNA_HOME/jobs/components/create_dyelnk.sh" dyelnk.toml
```

The direct entry point is:

```bash
python "$PYEDNA_HOME/scripts/create_dyelnk.py" --config dyelnk.toml
```

## Common Modifications Or Advanced Options

The public TOML interface currently exposes only component names and force-field identifiers. Linker conformer count is an internal default in the called Python method.

## Limitations / Troubleshooting

Missing MOL2, FRCMOD, `.attach`, or compatibility FRCMOD files are reported before assembly. `parmchk2` output is written to a log file when linked FRCMOD generation fails.
