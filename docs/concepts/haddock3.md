# HADDOCK3 In PyeDNA

HADDOCK3 is used by PyeDNA to sample candidate three-dimensional arrangements of dye-linker components relative to DNA before Amber topology generation.

## Why Docking Is Needed

Dye-labeled DNA systems require more than residue templates and covalent connectivity. The dye-linker component must also have plausible 3D coordinates relative to the DNA backbone. PyeDNA uses HADDOCK3 to generate candidate docked structures constrained by attachment information.

## What PyeDNA Supplies

PyeDNA supplies HADDOCK3 with:

- a DNA PDB from NAB generation or `DNA_DIR`;
- dye or dye-linker PDB/topology/parameter files prepared for HADDOCK;
- distance restraints for the intended covalent attachment points;
- a rendered `docking_config.cfg`.

For attachment workflows, PyeDNA removes the DNA residue occupied by a dye-linker component from the HADDOCK DNA input and writes restraints between neighboring DNA atoms and dye-linker attachment atoms.

## What HADDOCK Samples

HADDOCK3 samples candidate spatial arrangements that satisfy the provided restraints and scoring settings. In PyeDNA, those models are not final Amber systems. They are candidate coordinate sets that still need PyeDNA finalization and Amber preparation.

## What Candidate Models Represent

Each candidate model represents one possible docked geometry for the DNA-dye system under the current restraints and HADDOCK settings. PyeDNA ranks completed HADDOCK output using selected CAPRI geometry terms and copies the top configured number of models into `structures/`.

> **AUTHOR INPUT REQUIRED**
>
> Explain what structural checks users should perform on selected HADDOCK models before accepting them for Amber preparation.

## How PyeDNA Uses HADDOCK Results

After HADDOCK finishes, PyeDNA restores original atom and residue names, reconstructs the final DNA-dye PDB order, writes a final bond table, and records attachment residue mapping for later analysis.

The distinction between HADDOCK3 and Amber preparation is:

```text
HADDOCK3 -> determines candidate 3D arrangements
tleap    -> establishes the Amber topology/force-field representation
```

PyeDNA-specific TOML fields for docking are documented in [create_structure](../create_structure/create_structure.md).
