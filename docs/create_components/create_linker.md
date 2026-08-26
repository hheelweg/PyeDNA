# create_linker

## Purpose

`create_linker` creates reusable linker residue templates for 3' and 5' DNA attachment contexts and writes them into `LNK_DIR` when library output is requested.

## Prerequisites

- `PYEDNA_HOME`, `AMBERHOME`, and AmberTools executables available.
- RDKit and PySCF/GPU4PySCF available.
- `LNK_DIR` set when `output.directory = "library"`.
- `AMBERHOME` set so OL15 reference charges can be read from `dat/leap/lib/DNA.OL15.lib`.

## User Input Required

**Required:** linker name/code, mapped SMILES fragments, residue partition boundaries for both variants, and RESP charge-restraint force field.

> **AUTHOR INPUT REQUIRED**
>
> Explain the atom-map convention for linker `dye_cap`, `core`, and `dna_cap`, including how users should preserve chemically meaningful attachment atoms across fragments.

> **AUTHOR INPUT REQUIRED**
>
> Explain how to choose `three_prime` and `five_prime` boundary bonds and how those variants correspond to the physical orientation of a linker on DNA.

## Minimal Configuration Example

```toml
[component]
type = "linker"
name = "ExampleLinker"
code = "EL"

[structure]
dye_cap = "[H:1]"
core = "[C:2]([H:3])([H:4])[O:5]"
dna_cap = "[P:6](=[O:7])([O-:8])([O:9][C:10]([H:11])([H:12])[H:13])"

[structure.boundaries.three_prime]
dye = [1, 2]
dna = [5, 6]

[structure.boundaries.five_prime]
dye = [1, 2]
dna = [5, 6]

[parameterization]
charge_method = "resp"
forcefield = "gaff2"

[parameterization.restraints]
forcefield = "OL15"
```

This is a syntax example only. The mapped atoms and boundaries must match the actual linker chemistry.

## Configuration Reference

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `[component].type` | optional | `"linker"` | Must be `"linker"` if supplied. |
| `[component].name` | required | none | Base name for generated files. |
| `[component].code` | optional for current-directory output; required for library output | `L03`/`L05` residue names when omitted | Library identifier. When supplied, final residue names are `<code>3` and `<code>5`. |
| `[structure].dye_cap` | required | none | Mapped SMILES fragment representing the dye-side temporary cap. |
| `[structure].core` | required | none | Mapped SMILES fragment for the retained linker residue. |
| `[structure].dna_cap` | required | none | Mapped SMILES fragment representing the DNA-side cap. Current charge-restraint inference expects an OL15-like phosphate topology. |
| `[structure.boundaries.<variant>].dye` | required | none | Two mapped atom IDs defining the bond between the dye cap and retained residue. The atoms must exist and be bonded. |
| `[structure.boundaries.<variant>].dna` | required | none | Two mapped atom IDs defining the bond between the retained residue and DNA cap. The atoms must exist and be bonded. |
| `[parameterization].charge_method` | optional | `"resp"` | Only `"resp"` is supported. |
| `[parameterization].forcefield` | optional | `"gaff2"` | Amber/GAFF atom typing passed to AmberTools. |
| `[parameterization.restraints].forcefield` | required | none | Reference force field for fixed RESP restraints. Current implementation supports `"OL15"`. |
| `[qm].basis` or `[qm.geometry].basis` | optional | `"6-31g(d)"` | Basis for geometry optimization. |
| `[qm].maxsteps` or `[qm.geometry].maxsteps` | optional | `100` | Maximum geometry optimization steps. |
| `[qm].classical_preopt` | optional | `false` | If true, RDKit MMFF/UFF pre-optimization is run. |
| `[qm].classical_conformers` | optional | `20` | Must be at least 1. |
| `[output].directory` | optional | `"cwd"` | `"cwd"` writes locally; `"library"` writes into `LNK_DIR/<code>/<forcefield>/<restraint_forcefield>/`. |
| `[output].work_subdir` | optional | `"resp_fit"` | RESP intermediate directory. |
| `[output].cleanup` | optional | `"scratch"` | One of `"none"`, `"scratch"`, `"minimal"`, or `"library"`. |

Legacy `[linker]`, `[smiles]`, `[boundaries]`, `[charges]`, and `[amber]` shapes are partly accepted by the parser, but the structure above is preferred.

## What the Workflow Does

PyeDNA combines the mapped SMILES fragments, validates that every atom has a unique map ID, embeds and optimizes the full capped linker, computes an electrostatic potential, and performs RESP fitting. It fixes DNA-cap charges from OL15 reference charges inferred from the DNA-cap topology, writes one group charge restraint for the retained linker residue, and extracts separate 3' and 5' residue templates.

## Generated Outputs

The workflow writes `<code>3.mol2`, `<code>3.frcmod`, `<code>3.attach`, `<code>5.mol2`, `<code>5.frcmod`, and `<code>5.attach`. The `.attach` files contain `3CONNECT` and `5CONNECT` atom names for later bonding.

## How To Run

```bash
sbatch "$PYEDNA_HOME/jobs/components/create_linker.sh" linker.toml
```

## Common Modifications Or Advanced Options

Use `output.cleanup = "none"` while developing a new linker. Use library output only after verifying the final charge sums and attachment metadata.

## Limitations / Troubleshooting

Only RESP fitting is supported. Charge-restraint loading currently supports OL15. `custom` DNA-cap topologies that cannot be mapped to OL15 atom names will fail validation during charge loading.
