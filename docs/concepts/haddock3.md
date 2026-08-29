# HADDOCK3 in PyeDNA

PyeDNA uses **HADDOCK3** to generate plausible three-dimensional arrangements of dye-linker components relative to DNA before the system is converted into an Amber topology.

HADDOCK3 is an information-driven docking framework: it can use known structural information as restraints while sampling molecular arrangements. In PyeDNA, the known information is the intended DNA-linker connectivity.

The key distinction is:

```text
HADDOCK3 -> determines candidate molecular geometry
tleap    -> establishes the final Amber bonding/topology representation
```

HADDOCK3 does not build the final Amber force field. It produces candidate coordinates that PyeDNA later finalizes and passes to `tleap`.

## Workflow Role

The structure-generation path is:

```text
DNA structure
    +
dye-linker structures
    +
known attachment atoms
    ->
HADDOCK3 restrained docking
    ->
candidate docked models
    ->
PyeDNA finalization
    ->
tleap / Amber setup
```

PyeDNA supplies HADDOCK3 with:

- the DNA PDB prepared from NAB generation or `libraries.dna_dir`;
- one PDB plus CNS topology/parameter information for each dye-linker component;
- atom-to-atom distance restraints for the intended DNA-linker attachments;
- a rendered `docking_config.cfg` based on `templates/haddock_templates/docking_config.cfg`.

## Molecular Information Passed to HADDOCK

HADDOCK/CNS needs more than coordinates for non-standard residues such as dyes and linkers. PyeDNA therefore converts each dye-linker MOL2 template into HADDOCK-compatible files before docking:

| Input to HADDOCK | Format | Role |
| --- | --- | --- |
| DNA coordinates | PDB | Standard nucleic-acid structure supplied as the first molecule. |
| Dye-linker coordinates | PDB | One separate HADDOCK molecule per dye-linker component. |
| Dye-linker topology | CNS `.top` | Residue, atom, bond, and connectivity definitions for non-standard dye/linker residues. |
| Dye-linker parameters | CNS `.par` | Force-field parameters and charges used by CNS during docking/refinement. |
| Attachment restraints | CNS restraint table | Distance restraints encoding the intended DNA-linker bonds. |

In the current implementation, PyeDNA starts from the dye-linker MOL2 files produced by the component workflow. It then uses ACPYPE to create CNS topology and parameter files with GAFF-style atom typing and user-provided charges. PyeDNA combines the unique dye-linker files into:

```text
haddock/dyes_haddock.top
haddock/dyes_haddock.par
```

These files are passed to HADDOCK through `ligand_top_fname` and `ligand_param_fname`.

This means HADDOCK is **not** using the final Amber `prmtop`/`rst7` representation. However, it is still using a force-field-based molecular model: CNS-compatible topology, bonded terms, nonbonded parameters, and charges are available during docking and refinement.

## Why Restraints Are Needed

For an attached dye, PyeDNA already knows which atoms must later become connected. What it does not know is the dye/linker orientation that satisfies those connections without severe clashes.

Conceptually, each attachment restraint keeps two atoms within an allowed distance range:

$$
d_{\mathrm{low}} \le d_{ij} \le d_{\mathrm{high}}
$$

where

$$
d_{ij} = \lVert \mathbf{r}_i - \mathbf{r}_j \rVert .
$$

Outside the allowed range, HADDOCK/CNS applies a penalty. A useful schematic picture is:

$$
E_{\mathrm{rest}} \sim k_{\mathrm{rest}}(\Delta d)^2
$$

where $\Delta d$ is the distance violation and $k_{\mathrm{rest}}$ controls the restraint strength. This is a physical picture of the restraint, not a complete CNS implementation detail.

For PyeDNA, these restraints represent known future covalent connectivity, not uncertain experimental contacts. They should therefore guide the docking strongly.

## Restraints Are Not Bonds

During docking, the future DNA-linker connection is still a restraint:

```text
DNA atom  ...  linker atom
          restraint
```

After docking and PyeDNA finalization, the final Amber topology contains an explicit bond:

```text
DNA atom  -  linker atom
           bond
```

This separation lets HADDOCK search for a suitable geometry before the final Amber topology exists.

## HADDOCK Stages Used

PyeDNA renders a fixed HADDOCK3 workflow template with the following main modules:

| Stage | Role in PyeDNA |
| --- | --- |
| `[topoaa]` | Builds the CNS representation. PyeDNA supplies additional topology and parameter files for non-standard dye/linker residues. |
| `[rigidbody]` | Samples relative translations and rotations of DNA and dye-linker molecules while enforcing attachment restraints. |
| `[seletop]` | Selects the best rigid-body candidates for more expensive refinement. |
| `[flexref]` | Refines selected models with local flexibility, especially useful for linker torsions and dye/linker accommodation. |
| `[caprieval]` | Evaluates final HADDOCK models so PyeDNA can select and reformat top candidates. |

The rigid-body stage searches over approximate molecular placement. For molecule $m$, the main variables can be thought of as translation and orientation:

$$
\mathbf{R}_m,\quad \boldsymbol{\Omega}_m .
$$

The flexible-refinement stage then allows local adaptation around promising placements.

This is more than a geometric overlap check. HADDOCK performs restrained energy minimization and refinement with respect to both the molecular force-field terms and the user-specified restraints. For PyeDNA, that means candidate structures are selected from a physically informed landscape rather than from simple atom-distance filtering.

## Scoring and Model Selection

HADDOCK ranks models using weighted energetic and geometric terms. A simplified scoring expression is:

```math
S_{\mathrm{HADDOCK}}
=
w_{\mathrm{vdW}}E_{\mathrm{vdW}}
+
w_{\mathrm{elec}}E_{\mathrm{elec}}
+
w_{\mathrm{desolv}}E_{\mathrm{desolv}}
+
w_{\mathrm{AIR}}E_{\mathrm{AIR}}
+
w_{\mathrm{BSA}}\mathrm{BSA}
+ \cdots
```

In this expression:

- $E_{\mathrm{vdW}}$ reflects short-range packing and steric clashes;
- $E_{\mathrm{elec}}$ reflects electrostatic interactions;
- $E_{\mathrm{desolv}}$ is an empirical desolvation contribution;
- $E_{\mathrm{AIR}}$ measures restraint violations;
- $\mathrm{BSA}$ is buried surface area.

The HADDOCK score is a model-ranking function, not an ab initio energy or a binding free energy.

PyeDNA's `finalize` stage ranks completed HADDOCK models using selected CAPRI geometry terms and copies the configured number of top models into `structures/`.

## Physical Picture

The docking problem in PyeDNA is not:

> Where would a free dye bind to DNA?

It is:

> Given known attachment sites, what plausible 3D structures satisfy that connectivity?

A compact way to view the effective objective is:

```math
E_{\mathrm{effective}}(\mathbf{R})
=
E_{\mathrm{molecular}}(\mathbf{R})
+
E_{\mathrm{attachment}}(\mathbf{R})
+
E_{\mathrm{DNA-restraints}}(\mathbf{R}) .
```

Here, $E_{\mathrm{molecular}}$ includes terms such as van der Waals and electrostatics, $E_{\mathrm{attachment}}$ penalizes violation of the intended DNA-linker geometry, and $E_{\mathrm{DNA-restraints}}$ helps preserve nucleic-acid structure during refinement.

## After HADDOCK

A HADDOCK model is not a thermally equilibrated MD snapshot. It is a restrained structural candidate.

After HADDOCK, PyeDNA:

1. selects top models;
2. restores PyeDNA atom and residue naming;
3. reconstructs final DNA-dye residue ordering;
4. writes bond and residue-mapping metadata;
5. prepares a selected structure for Amber with `tleap`.

Amber MD then performs solvation, minimization, equilibration, and production sampling.

## What Users Should Inspect

Before accepting a model for Amber preparation, inspect whether:

- intended DNA-linker attachment sites are geometrically satisfied;
- linker paths are physically sensible;
- dyes do not penetrate DNA bases or backbone;
- dye-DNA or dye-dye clashes are absent or mild;
- the DNA duplex remains structurally intact;
- dye orientation is reasonable for the intended construct;
- multiple high-ranking models represent different plausible starting structures.

A favorable HADDOCK score alone does not guarantee a usable structure.

> **AUTHOR INPUT REQUIRED**
>
> - [ ] TODO : Add project-specific structural acceptance criteria for choosing a HADDOCK model before Amber preparation, such as typical acceptable attachment distances, visual checks, or cases where a lower-ranked model should be preferred.

## PyeDNA Configuration

Users normally configure HADDOCK through `structure.toml`, not by editing the complete HADDOCK configuration directly. The relevant user-facing fields are documented in [create_structure](../create_structure/create_structure.md).

Advanced HADDOCK settings can be overridden where supported, but changes should be made only when their effect on sampling, restraint strength, or scoring is understood. For general HADDOCK3 details, use the HADDOCK3 user manual.
