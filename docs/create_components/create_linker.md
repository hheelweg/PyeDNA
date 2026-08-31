# Create Linker Molecule (`create_linker`)

## Purpose

`create_linker` creates reusable linker residue templates for 3' and 5' DNA attachment contexts and writes them into `libraries.linker_dir` when library output is requested.

## What the Workflow Does

PyeDNA combines the mapped SMILES fragments, validates that every atom has a unique map ID, embeds and optimizes the full capped linker, computes an electrostatic potential, and performs RESP fitting. It fixes DNA-cap charges from OL15 reference charges inferred from the DNA-cap topology, writes one group charge restraint for the retained linker residue, and extracts separate 3' and 5' residue templates.

## Prerequisites

- PyeDNA runtime configuration with `amber.ambertools_home` pointing to an AmberTools installation that provides `antechamber`, `respgen`, `resp`, and `parmchk2`.
- RDKit and PySCF available in the Python environment.
- Optional: CuPy and GPU4PySCF available in the Python environment for GPU-accelerated PySCF steps when a CUDA device is visible to the process.
- `libraries.linker_dir` set when `output.directory = "library"`.
- `amber.ambertools_home` set so OL15 reference charges can be read from `dat/leap/lib/DNA.OL15.lib`.

## User Input Required

**Required:** linker name/code, mapped SMILES fragments, residue partition boundaries for both variants, and RESP charge-restraint force field.

> **Important Input: `core`, `dye_cap`, `dna_cap`**
>
> Note that each linker molecule contains a chemical core (`core` SMILES string) as well as two caps, `dye_cap` and `dna_cap` (both SMILES strings), which describe the chemistry of attachments to the dye and dna side, respectively. 
> Note that the `dna_cap` is typically always the same as it is supposed to mimic the phospate group (-$\mathrm{OPOO^-OCH_3}$) of the DNA, and `dye_cap` is suppsoed to mimic the electronic withdrawing character of (most) dyes, e.g. something like an acetyl group (-$\mathrm{COCH_3}$).
> **Note**: Heavy (i.e. non-hydrogen) atoms need to be named/indexed in SMILES string *consecutively* in the connectivity direction from `dye_cap` to `core` to `dna_cap` for internal handling of atom indices.

**Example (`core`)**
```smiles
[CH2:4][CH2:5][O:6][CH2:7][CH2:8]
```

**Example (`dye_cap`)**
```smiles
[CH3:1][C:2](=[O:3])
```

**Example (`dna_cap`)**
```smiles
[O:9][P:10](=[O:11])([O-:12])[O:13][CH3:14]
```

This gives the full (chemically saturated) linker molecule, which will be used for the geometry optimization and charge fitting of the linker:
 ```smiles
 [CH3:1][C:2](=[O:3])[CH2:4][CH2:5][O:6][CH2:7][CH2:8][O:9][P:10](=[O:11])([O-:12])[O:13][CH3:14]
 ```

 For the RESP charge fitting the charges for `dna_cap` will be constrained to match the DNA `forcefield` in `[parameterization.restraints]`, e.g. OL15. 
 This is designed so as to match the physics of the DNA forcefield around the DNA-like atoms of the linker molecules in close porximity to the DNA. 

 Note the each dye has **two** attachment via a linker to the DNA, so even though both linkers have identical core chemistry, they differ in the way we need to handle the attachment to DNA because of the differences in 3' and 5' residue atoms in typical DNA `.pdb` files, i.e. their individual dna_cap will differ
This will technically give rise to two different sets of linker files (one with suffix `3` and the other one with suffix `5`) that need to be handled separately. 


> **Important Input: `three_prime` and `five_prime`**
>
> `dye` and `dna` will require a list of (heavy) atom indices as specified in `[structure]` SMILES strings `[atom_idx1, atom_idx2]`, denoting that the bond between `atom_idx1` and `atom_idx2` will be cleaved to separate the linker into `dye_cap`, `core`, and it's 3' or 5' `dna_cap`.
> The linker to the 3' end of the DNA should end with a chemical -$\mathrm{OPOO^{-}}$ group, while the 5' linker has to end with a chemical -$\mathrm{O}$ group, wich consequently leads to different `[structure.boundaries.three_prime].dna` and `[structure.boundaries.five_prime].dna`
> **Note**: Pay close attention to heavy atom indices specified in SMILES strings for `[structure.boundaries]` in order to set up this. 

## Minimal Configuration Example for `linker.toml`

```toml
[component]
type = "linker"
name = "diethyl_ether"
code = "DE"

[structure]
dye_cap = "[CH3:1][C:2](=[O:3])"
core = "[CH2:4][CH2:5][O:6][CH2:7][CH2:8]"
dna_cap = "[O:9][P:10](=[O:11])([O-:12])[O:13][CH3:14]"

[structure.boundaries.three_prime]
dye = [2, 4]
dna = [10, 13]

[structure.boundaries.five_prime]
dye = [2, 4]
dna = [9, 10]

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
| `[output].directory` | optional | `"cwd"` | `"cwd"` writes locally; `"library"` writes into `<libraries.linker_dir>/<code>/<forcefield>/<restraint_forcefield>/`. |
| `[output].work_subdir` | optional | `"resp_fit"` | RESP intermediate directory. |
| `[output].cleanup` | optional | `"scratch"` | One of `"none"`, `"scratch"`, `"minimal"`, or `"library"`. |

Legacy `[linker]`, `[smiles]`, `[boundaries]`, `[charges]`, and `[amber]` shapes are partly accepted by the parser, but the structure above is preferred.


## Generated Outputs

The workflow writes `<code>3.mol2`, `<code>3.frcmod`, `<code>3.attach`, `<code>5.mol2`, `<code>5.frcmod`, and `<code>5.attach`. The `.attach` files contain `3CONNECT` and `5CONNECT` atom names for later bonding.

## How To Run

```bash
pyedna components create-linker linker.toml
```

If the config filename is omitted, PyeDNA uses `linker.toml` in the current directory. On HPC systems, scheduler scripts may wrap this CLI command.

The scientific TOML file and CLI command are the same for CPU and GPU execution. Submit the job without GPU resources for CPU PySCF execution, or request a CUDA GPU through the scheduler to let PyeDNA use GPU4PySCF automatically when the validated GPU stack is installed.

## Common Modifications Or Advanced Options

Use `output.cleanup = "none"` while developing a new linker. Use library output only after verifying the final charge sums and attachment metadata.

## Limitations / Troubleshooting

Only RESP fitting is supported. Charge-restraint loading currently supports OL15. `custom` DNA-cap topologies that cannot be mapped to OL15 atom names will fail validation during charge loading.
