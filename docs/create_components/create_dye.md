# create_dye

## Purpose

`create_dye` creates and parameterizes a reusable dye component. It writes molecular and force-field files for an uncapped dye residue, optionally into the dye library configured by `DYE_DIR`.

## What the Workflow Does

PyeDNA attaches temporary cap atoms to the mapped dye core, embeds a 3D conformer, writes SDF/PDB structures, optimizes the capped geometry, computes an electrostatic potential, and performs two-stage RESP charge fitting. It then extracts the uncapped core atoms into a final residue MOL2, creates missing GAFF parameters with `parmchk2`, writes a `.attach` file containing the final linker attachment atom names, and checks that the MOL2 charge is close to the target formal charge of the dye.


## Prerequisites

- `PYEDNA_HOME`, `AMBERHOME`, and AmberTools executables available through the environment.
- PySCF/GPU4PySCF (for geometry optimization) and RDKit available in the Python environment
- `DYE_DIR` set when `output.directory = "library"`.

## User Input Required

**Required:** dye name, dye code or residue name, mapped dye core SMILES, cap SMILES, and cap target atom-map IDs.

> **Important Input: `core_smiles` and `cap_targets`**
>
> Note that `PyeDNA` requires feeding a `core_smiles` string that descibes the chemical structure of the *uncapped* dye molecule that is getting atatched to the DNA, even if that means the structure is not chemically sound, i.e. at the attachment points of the linker molecule a bond is missing, see in the attached example structure, where the attachment points are the two `N` atoms.
> **Important**: Note that we index these atoms with `1` and `2` in `core_smiles`, which is important because these indices appear in `cap_targets`, giving the required information which atoms need to be capped for the geometry optimization and charge fitting procedure.


> **Important Input: `cap_smiles`**
>
> The `cap_smiles` strings (e.g. `[H]` or `C` as SMILES strings) are required to chemically saturate `core_smiles` dye structure and actually mimic the electronic environment if attached linker chain. 

**Example (`core_smiles`)**
```smiles
CC1(C)C2=CC=CC=C2[N:1]/C1=C\C=C\C(C3(C)C)=[N+:2]C4=C3C=CC=C4
```



## Minimal Configuration Example

```toml
[component]
type = "dye"
name = "Cy3"
code = "CY3"

[structure]
core_smiles = 'CC1(C)C2=CC=CC=C2[N:1]/C1=C\C=C\C(C3(C)C)=[N+:2]C4=C3C=CC=C4'
cap_smiles =  'C'
cap_targets = [1, 2]
formal_charge = 1

[parameterization]
charge_method = "resp"
forcefield = "gaff2"

[output]
directory = "cwd"
cleanup = "scratch"
```

The SMILES above is only a syntax example. The mapped atoms and chemistry must be chosen for the actual dye.

## Configuration Reference

| Field | Required | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `[component].type` | optional | `"dye"` | Must be `"dye"` if supplied. |
| `[component].name` | required | none | Base name for generated files and intermediate capped molecule. |
| `[component].code` | optional for current-directory output; required for library output | `name` is used as residue name when omitted | Final residue name and library directory identifier. |
| `[structure].core_smiles` | required | none | RDKit-readable dye core SMILES. At least one atom-map ID must be present. |
| `[structure].cap_smiles` | required | none | RDKit-readable cap fragment. Current validation supports exactly one cap atom. |
| `[structure].cap_targets` | required | none | Unique atom-map IDs in `core_smiles`; one cap is attached to each target. `core_targets` is accepted as a legacy alias. |
| `[structure].formal_charge` | optional | formal charge inferred from core SMILES | Target charge for the uncapped dye RESP group and final residue charge check. |
| `[parameterization].charge_method` | optional | `"resp"` | Only `"resp"` is supported. |
| `[parameterization].forcefield` | optional | `"gaff2"` | Amber/GAFF atom typing passed to AmberTools. |
| `[parameterization].amber_forcefield` | optional | `"gaff2"` | Legacy alias for force-field selection. |
| `[amber].forcefield` | optional | `"gaff2"` | Also accepted; values in `[parameterization]` override `[amber]`. |
| `[qm].basis` or `[qm.geometry].basis` | optional | `"6-31g(d)"` | Basis for PySCF geometry optimization. |
| `[qm].maxsteps` or `[qm.geometry].maxsteps` | optional | `100` | Maximum geometry optimization steps. |
| `[qm].classical_preopt` | optional | `false` | If true, RDKit MMFF/UFF conformer pre-optimization is run before QM optimization. |
| `[qm].classical_conformers` | optional | `20` | Number of RDKit conformers for classical pre-optimization; must be at least 1. |
| `[output].directory` | optional | `"cwd"` | `"cwd"` writes into the current working directory; `"library"` writes into `DYE_DIR/<code>/<forcefield>/`. |
| `[output].work_subdir` | optional | `"resp_fit"` | Subdirectory for RESP fitting intermediates. |
| `[output].cleanup` | optional | `"scratch"` | One of `"none"`, `"scratch"`, `"minimal"`, or `"library"`. `"library"` requires `output.directory = "library"`. |

Legacy `[dye]`, `[core]`, `[caps]`, `[charge]`, and `[amber]` shapes are still partly accepted by the parser, but new examples should use `[component]`, `[structure]`, `[parameterization]`, `[qm]`, and `[output]`.


## Generated Outputs

Final outputs include `<code>.mol2`, `<code>.frcmod`, and `<code>.attach`. Intermediate outputs can include `<name>.sdf`, `<name>.pdb`, `qm_opt/`, and the RESP working directory, depending on `[output].cleanup` settings. 

## How To Run

```bash
sbatch "$PYEDNA_HOME/jobs/components/create_dye.sh" dye.toml
```

The direct Python entry point is:

```bash
python "$PYEDNA_HOME/scripts/create_dye.py" --config dye.toml
```

## Common Modifications Or Advanced Options

Use `output.cleanup = "none"` when debugging parameterization. Use `output.directory = "library"` only after confirming the generated files should become reusable library inputs.

## Limitations / Troubleshooting

Only RESP charge fitting is supported. The cap fragment must currently contain one atom. Existing library output directories are not overwritten.
