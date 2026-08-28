# Amber Setup

## Purpose

Amber setup converts a selected finalized DNA-dye PDB structure into Amber MD inputs: `prmtop`, `rst7`, and a solvated PDB.

## What the Workflow Does

PyeDNA starts from `structures/<system.name>_<amber.model>.pdb`. It reads `structures/bonds.csv`, loads dye/linker residue templates and FRCMOD files, prepares an Amber-facing PDB with atom-name changes required by component metadata, writes a `tleap` input file, and optionally runs `tleap`.

The generated `tleap` input:

1. sources the DNA force field, dye/linker force field, and water force field;
2. loads each dye/linker MOL2 template;
3. loads dye/linker FRCMOD files and compatibility FRCMOD files;
4. applies any Amber atom-name/type mappings found in dye `.attach` metadata;
5. loads the prepared PDB;
6. adds covalent bonds from `structures/bonds.csv`, including DNA-dye, dye-dye, and internal composite-dye bonds;
7. checks the molecule;
8. solvates with `solvateBox`;
9. neutralizes with configured ions when `neutralize = true`;
10. writes `prmtop`, `rst7`, and a solvated PDB.

Conceptually:

```text
HADDOCK3 -> determines candidate 3D arrangements
tleap    -> establishes the Amber topology/force-field representation
```

## Prerequisites

- A finalized structure at `structures/<system.name>_<amber.model>.pdb`.
- A final bond table at `structures/bonds.csv`.
- Dye/linker MOL2, FRCMOD, and attachment metadata in `libraries.dye_dir` and `libraries.linker_dir`, or generated linked intermediates from the structure workflow.
- PyeDNA runtime configuration with `amber.ambertools_home` pointing to an AmberTools installation that provides `tleap` and Amber force-field data.

## User Input Required

**Required:** selected model number and force-field/solvation settings in `structure.toml`.

> **AUTHOR INPUT REQUIRED**
>
> Explain how users should decide whether the selected docked model is chemically and structurally suitable before running `tleap`.

> **AUTHOR INPUT REQUIRED**
>
> Explain when users must manually add DNA-linker bond, angle, or dihedral parameters to `connectparams.frcmod`.

## Minimal Configuration Example

```toml
[forcefield]
dna = "OL15"
attachments = "gaff2"
water = "tip3p"

[amber]
model = 1
output_name = "example_system"
water_model = "TIP3P"
solvent_padding = 20.0
positive_ion = "Na+"
negative_ion = "Cl-"
neutralize = true
```

## Configuration Reference

See the `[forcefield]` and `[amber]` tables in [create_structure](create_structure.md). User-facing force-field choices belong in `[forcefield]`; Amber setup maps those choices onto the internal `tleap` settings.

## Generated Outputs

For `output_name = "example_system"`, Amber setup writes:

- `example_system.pdb`: Amber-prepared, unsolvated PDB
- `example_system_tleap.in`: generated `tleap` input, removed after successful full setup
- `tleap_amber.log`: combined `tleap` stdout/stderr and `leap.log`
- `example_system.prmtop`: Amber topology
- `example_system.rst7`: Amber restart/coordinate file
- `example_system_solvated.pdb`: solvated structure

After successful `tleap`, generated linked dye-linker intermediates and the `tleap` input are removed by the current implementation.

## How To Run The Workflow

```bash
pyedna structure amber structure.toml
```

If `[workflow].prepare_amber = true`, the `finalize` stage also runs Amber setup after processing HADDOCK results.

## Common Modifications Or Advanced Options

Adjust `amber.model` to select a different finalized HADDOCK model. Adjust `[forcefield]` for DNA, attachment, or water force-field choices. Adjust `solvent_padding`, ions, and neutralization according to the intended MD system.

## Limitations / Troubleshooting

Current water-model handling maps only `water_model = "TIP3P"` to `TIP3PBOX`. If `tleap` fails because bond, angle, or dihedral parameters are missing, inspect `tleap_amber.log` and the reported compatibility FRCMOD path.
