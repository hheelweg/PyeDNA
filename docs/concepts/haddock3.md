# HADDOCK3 in PyeDNA

PyeDNA uses **HADDOCK3** to generate physically reasonable three-dimensional arrangements of dye–linker components relative to DNA before the system is converted into an Amber topology.

HADDOCK3 is an information-driven molecular docking framework. Rather than searching only for structures with favorable non-bonded interaction energies, HADDOCK can incorporate prior structural information as **restraints** that actively guide the docking calculation.

This feature is central to the PyeDNA workflow.

For a dye attached to DNA, PyeDNA already knows **which atoms must ultimately become covalently connected**. What is generally unknown is the three-dimensional orientation of the dye and linker relative to the DNA that simultaneously

- places the attachment atoms close enough to form the intended bonds,
- avoids severe steric overlap,
- preserves a reasonable DNA structure,
- and allows the flexible linker and dye to adopt a compatible geometry.

PyeDNA therefore uses HADDOCK3 as a **restrained conformational search and refinement step**.

The general HADDOCK3 documentation is available in the [HADDOCK3 user manual](https://www.bonvinlab.org/haddock3-user-manual/).

---

## Role of HADDOCK3 in the PyeDNA workflow

The relevant part of the PyeDNA structure workflow can be summarized as

```text
DNA structure
+
parameterized dye–linker structures
+
known DNA ↔ linker attachment atoms
                │
                ▼
       construct HADDOCK inputs
                │
                ▼
             HADDOCK3
                │
      restrained docking search
                │
                ▼
       candidate 3D structures
                │
                ▼
       select / finalize model
                │
                ▼
              tleap
                │
                ▼
       Amber topology + restart
```

The distinction between the last two stages is important:

```text
HADDOCK3 → determines candidate molecular geometry
tleap     → establishes the final Amber bonding/topology representation
```

HADDOCK is therefore **not used to construct the final Amber force field**. Its role is to find suitable coordinates from which the intended covalent DNA–linker bonds can subsequently be created.

---

# Why restraints are necessary

Suppose a linker is intended to replace one DNA residue and connect two neighboring DNA residues.

PyeDNA knows that specific atoms on the linker must eventually connect to specific atoms on the neighboring DNA residues. Merely placing the dye somewhere close to DNA does not enforce this topology.

The attachment information is therefore translated into **unambiguous HADDOCK distance restraints**.

Conceptually, a restraint specifies that two selected atoms should satisfy a target distance

\[
d_{\mathrm{low}} \leq d_{ij} \leq d_{\mathrm{high}},
\]

where

\[
d_{ij}=|\mathbf r_i-\mathbf r_j|
\]

is the instantaneous separation of the restrained atoms.

Inside the permitted region the restraint contributes little or no energetic penalty. Outside that region, HADDOCK/CNS applies a restoring penalty that can be represented schematically as

\[
E_{\mathrm{rest}} \sim
k_{\mathrm{rest}}
\left(\Delta d\right)^2,
\]

where \(\Delta d\) measures the violation of the allowed distance range and \(k_{\mathrm{rest}}\) controls how strongly the calculation is driven back toward the desired geometry.

This expression is useful as a **physical picture** of the restraint rather than as the complete CNS implementation.

The important consequence is that the restraints modify the energy landscape explored during docking:

```text
unrestrained search
    ↓
many geometrically irrelevant DNA–dye orientations

attachment restraints
    ↓
search concentrated around structures compatible
with the intended DNA–linker connectivity
```

For PyeDNA these restraints represent **known chemical connectivity**, not uncertain experimental interaction data. They are consequently treated as unambiguous restraints that should remain active throughout docking and refinement.

---

## Restraints are not covalent bonds

A HADDOCK distance restraint should not be confused with an actual force-field bond.

During docking,

```text
DNA atom  ·······  linker atom
          restraint
```

is still represented through a distance constraint/penalty.

After docking and PyeDNA finalization,

```text
DNA atom — linker atom
           bond
```

is explicitly introduced into the molecular topology used by Amber.

This separation allows HADDOCK to search for a geometry in which the future bond can be formed without requiring the final Amber topology to exist beforehand.

---

# The PyeDNA HADDOCK3 workflow

PyeDNA does not construct an arbitrary HADDOCK workflow for every calculation.

It currently uses a specific template:

```text
data/haddock_templates/docking_config.cfg
```

which PyeDNA renders for each DNA–dye system.

The major HADDOCK modules are

```text
[topoaa]
    ↓
[rigidbody]
    ↓
[seletop]
    ↓
[flexref]
    ↓
[caprieval]
```

Each stage has a distinct purpose.

---

# Global configuration and molecular inputs

The beginning of the generated configuration defines the HADDOCK run and its molecular inputs:

```text
run_dir = "..."
mode = "..."
ncores = ...

clean = ...
postprocess = ...

molecules = [
    ...
]
```

## `run_dir`

Directory in which HADDOCK creates the docking calculation.

## `mode`

Controls the HADDOCK3 execution mode used by the workflow.

## `ncores`

Number of CPU cores available to HADDOCK.

This controls computational parallelism rather than the physical docking model.

## `clean` and `postprocess`

PyeDNA currently exposes these through its template so that intermediate HADDOCK/CNS files can be retained and automatic post-processing behavior can be controlled.

Keeping intermediate files is particularly useful for diagnosing failed topology generation, CNS errors, restraint violations, or unusual docking geometries.

---

## Molecular ordering

PyeDNA provides the input molecules as

```text
molecules = [
    DNA,
    dye_1,
    dye_2,
    ...
]
```

with **DNA always supplied first**.

Every dye–linker component is provided as a separate HADDOCK molecule.

For a DNA system containing two attached dyes, the conceptual input is therefore

```text
molecule 1 → DNA
molecule 2 → dye/linker A
molecule 3 → dye/linker B
```

This is important because HADDOCK initially treats these as separate molecular bodies whose relative positions can be sampled.

---

# 1. Topology generation: `[topoaa]`

The first HADDOCK stage is

```text
[topoaa]

ligand_top_fname = "..."
ligand_param_fname = "..."

delenph = ...
autohis = ...
```

`topoaa` converts the supplied molecular coordinates into the CNS representation required by the following HADDOCK modules.

DNA residues are understood by the standard HADDOCK/CNS topology machinery, but dye and linker residues are non-standard molecules. PyeDNA therefore supplies additional CNS topology and parameter information.

## `ligand_top_fname`

Contains CNS residue/topology definitions for the non-standard dye/linker residues.

These definitions describe information such as

- atoms,
- atom types,
- residue organization,
- and molecular connectivity required by CNS.

## `ligand_param_fname`

Contains the associated molecular mechanics parameters needed by CNS during docking and refinement.

Together,

```text
ligand topology
+
ligand parameters
```

allow HADDOCK to treat the PyeDNA dye/linker residues as actual molecular species during its calculations.

The same files are explicitly supplied again to the later CNS-based docking and refinement modules.

---

## `delenph`

Controls removal of non-polar hydrogen atoms.

The PyeDNA template explicitly specifies this behavior rather than relying implicitly on generic HADDOCK defaults.

## `autohis`

Controls HADDOCK's automatic histidine handling.

This is primarily a protein-oriented feature and is not relevant to the DNA–dye systems being constructed here, so PyeDNA explicitly controls/disables it in the rendered configuration.

---

# 2. Rigid-body docking: `[rigidbody]`

The first actual docking search occurs in

```text
[rigidbody]
```

This stage explores the relative arrangement of DNA and the individual dye molecules while treating each input molecule primarily as a rigid body.

The important degrees of freedom are therefore approximately

\[
\mathbf R_m,\qquad \mathbf\Omega_m,
\]

where \(\mathbf R_m\) is the translation and \(\mathbf\Omega_m\) the orientation of molecular body \(m\).

Internal bond lengths, bond angles, and torsional coordinates are not the principal search variables at this stage.

Conceptually:

```text
DNA                  DNA
│                    │
│        →           │--- dye
│                    │
dye

different translations and rotations
```

are explored without yet requiring extensive internal structural rearrangement.

---

## Attachment restraints

PyeDNA supplies

```text
unambig_fname = "..."
```

to this module.

This file contains the atom-to-atom distance restraints encoding the desired DNA–linker attachments.

These restraints are especially important during rigid-body docking because they turn an otherwise enormous translational and rotational search into a chemically informed search.

Without them, a dye could dock almost anywhere on the DNA surface.

With them, its position must remain compatible with its intended attachment sites.

---

## `sampling`

```text
sampling = ...
```

sets the number of rigid-body docking models that HADDOCK attempts to generate.

Higher sampling explores more candidate molecular arrangements at increased computational cost.

The amount of sampling required is strongly related to the amount of structural information supplied by the restraints:

```text
weak / ambiguous constraints → larger search space → more sampling

strong attachment constraints → smaller search space → less sampling required
```

PyeDNA operates much closer to the second case because the intended attachment atoms are known.

---

## `ntrials`

```text
ntrials = ...
```

controls the number of internal attempts HADDOCK can make when generating each requested rigid-body solution.

This should not be confused with `sampling`: one controls the requested number of models, while the other controls how hard HADDOCK may try internally to produce those models successfully.

---

# Restraint strength during rigid-body docking

PyeDNA explicitly controls

```text
randremoval = ...
unambig_scale = ...
```

## `randremoval`

HADDOCK supports workflows in which some restraints are randomly omitted during sampling. This is useful when experimental interaction information is uncertain or ambiguous.

That is **not the physical situation represented by the PyeDNA attachment restraints**.

PyeDNA knows which attachment atoms should eventually become covalently bonded, so those geometric constraints should not disappear randomly during the search.

The template therefore keeps the attachment restraints active rather than treating them as uncertain interaction information.

## `unambig_scale`

This controls the force constant associated with the unambiguous restraint potential during rigid-body docking.

A larger value makes restraint violations energetically more costly during the actual coordinate optimization.

This parameter determines how strongly the restraint acts on the **dynamics/minimization**.

It is different from `w_air` or `w_dist`, which determine how strongly restraint energies contribute when HADDOCK **scores** the resulting model.

This distinction is important:

```text
unambig_scale
    ↓
changes the physical restraint force used during optimization

w_air / w_dist
    ↓
changes how much the resulting restraint energy contributes to ranking
```

---

# Intermolecular physical interactions

Rigid-body minimization is not governed solely by attachment restraints.

PyeDNA also controls

```text
inter_rigid = ...
elecflag = ...
```

which determine how intermolecular interactions participate during the rigid-body stage.

At a schematic level, the energetic landscape relevant to docking contains terms such as

\[
E_{\mathrm{dock}}
=
E_{\mathrm{vdW}}
+
E_{\mathrm{elec}}
+
E_{\mathrm{rest}}
+\cdots .
\]

The attachment restraint attempts to bring the future bonding atoms into compatible positions, while the non-bonded terms prevent the molecules from simply passing through one another or adopting arbitrarily unfavorable orientations.

This competition is physically important.

For example, satisfying a distance restraint by placing a dye directly through the DNA backbone would strongly increase short-range repulsion. HADDOCK must therefore search for a geometry that satisfies the attachment geometry **and** remains molecularly plausible.

---

# HADDOCK model scoring

After generating candidate structures, HADDOCK assigns scores based on a weighted combination of energetic and geometric terms.

A useful schematic representation is

\[
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
+\cdots
\]

where the exact set of terms and weights depends on the HADDOCK module.

The PyeDNA rigid-body template explicitly controls

```text
w_air
w_vdw
w_elec
w_desolv
w_bsa
w_dist
```

These terms should be interpreted as follows.

### \(E_{\mathrm{vdW}}\): van der Waals interaction energy

Represents short-range intermolecular packing.

It contains the strong repulsive interaction that prevents severe atomic overlap as well as attractive dispersion interactions at appropriate separation.

This term is therefore particularly important for avoiding structures in which a dye satisfies its attachment restraints only by sterically penetrating DNA.

### \(E_{\mathrm{elec}}\): electrostatic interaction energy

Represents electrostatic interactions between atomic partial charges.

DNA is strongly charged, and dyes/linkers can themselves carry substantial partial or net charge, making electrostatics potentially important for the relative orientation of the components.

### \(E_{\mathrm{desolv}}\): desolvation term

Provides an empirical contribution associated with transferring molecular surfaces from solvent exposure into an intermolecular interface.

This is a scoring term rather than an explicit simulation of solvent molecules.

### \(E_{\mathrm{AIR}}\): restraint energy

Measures violation of the supplied interaction/distance restraints.

For PyeDNA this term is closely connected to whether a proposed geometry is compatible with the intended DNA–linker attachment geometry.

### BSA: buried surface area

Measures surface area buried upon formation of the molecular complex.

Depending on its weight, this term can reward or penalize particular amounts of intermolecular contact.

Importantly, **the HADDOCK score is a model-ranking function, not an ab initio molecular energy or a thermodynamic free energy**.

A lower HADDOCK score indicates a structure favored according to the chosen combination of restraint satisfaction, intermolecular energetics, and geometric terms. It should not be interpreted directly as an equilibrium binding free energy.

---

## Why PyeDNA uses restraint-dominated docking

The scientific question in PyeDNA is not

> Where would a free dye spontaneously bind to DNA?

Instead it is

> Given that this dye is chemically attached to these specific DNA sites, what plausible three-dimensional configurations can satisfy that connectivity?

That difference is fundamental.

The attachment topology is known information and should therefore strongly restrict the allowed structural ensemble.

The non-bonded energy terms then discriminate between geometries within that restricted space.

Schematically,

\[
\text{known connectivity}
\quad+\quad
\text{steric compatibility}
\quad+\quad
\text{electrostatics}
\quad+\quad
\text{local flexibility}
\]

determine the candidate structures.

---

# Auxiliary rigid-body restraints

The template also controls

```text
cmrest = ...
surfrest = ...
ranair = ...
```

These options provide additional ways of guiding ab-initio docking, such as center-of-mass or surface-based restraints.

PyeDNA does not rely on these generic docking strategies because it already has more specific structural information: the known DNA–linker attachment atoms.

Using the explicit chemical attachment restraints avoids introducing unrelated assumptions about generic binding surfaces.

---

## `rigidtrans`

```text
rigidtrans = ...
```

controls whether intermolecular translation remains available as part of the rigid-body search.

For PyeDNA, the dyes must be able to translate relative to DNA so that HADDOCK can find positions compatible with the attachment restraints.

---

# 3. Selecting models: `[seletop]`

Rigid-body docking may generate many candidate models.

The next template stage is

```text
[seletop]

select = ...
```

This selects the highest-ranked rigid-body models and passes only those structures to flexible refinement.

The workflow therefore performs a funnel:

```text
many rigid-body candidates
          │
       scoring
          │
          ▼
selected best candidates
          │
          ▼
expensive flexible refinement
```

This avoids performing the more expensive simulated-annealing refinement on every sampled orientation.

---

# 4. Flexible refinement: `[flexref]`

Rigid-body docking determines the approximate arrangement of DNA and dyes but cannot fully resolve local molecular strain.

The selected models therefore enter

```text
[flexref]
```

HADDOCK's flexible refinement stage uses simulated annealing and molecular dynamics in torsion-angle space.

Conceptually, the progression is

```text
global placement
     ↓
rigid-body refinement
     ↓
progressive local flexibility
     ↓
locally accommodated DNA–linker–dye structure
```

This is particularly important for PyeDNA because linkers are intrinsically flexible molecules.

A geometry may have correct global dye placement while still requiring changes in linker torsions to satisfy both attachment restraints without introducing steric strain.

---

# Restraints during simulated annealing

The PyeDNA template again supplies

```text
unambig_fname = "..."
randremoval = ...
```

so that the same attachment constraints remain active during flexible refinement.

The corresponding restraint strengths are stage-dependent:

```text
unambig_hot
unambig_cool1
unambig_cool2
unambig_cool3
```

This allows the restraint potential to be changed during successive phases of simulated annealing.

Conceptually, simulated annealing explores a temperature-dependent configurational landscape:

\[
P(\mathbf R) \propto
\exp\left[
-\frac{E(\mathbf R)}{k_{\mathrm B}T}
\right].
\]

At high effective temperature, larger structural rearrangements are accessible.

As the system cools, the search becomes progressively more localized around low-energy structures.

The restraint force constants determine how strongly deviations from the required attachment geometry are penalized during each of these stages.

---

# Flexible-refinement stages

PyeDNA explicitly controls

```text
mdsteps_rigid
mdsteps_cool1
mdsteps_cool2
mdsteps_cool3
```

These define how many molecular-dynamics steps are performed during the different stages of HADDOCK's simulated-annealing protocol.

More steps generally allow more conformational search but increase computational cost.

The progression can be understood schematically as

```text
high-temperature / rigid stage
           ↓
        cooling 1
           ↓
        cooling 2
           ↓
        cooling 3
           ↓
     refined structure
```

The later stages progressively allow the selected molecular regions to adapt around the docked interface.

---

# Dye flexibility

The PyeDNA template contains

```text
{{ flexibility_lines }}
```

which is replaced by one flexibility definition for every dye molecule.

This allows PyeDNA to identify which residues of each dye/linker component should participate in flexible refinement.

The important physical idea is that the dye does not have to remain in precisely the same internal conformation generated before docking.

In particular, linker torsions can provide the degrees of freedom required to reconcile

- DNA attachment geometry,
- dye orientation,
- steric exclusion,
- and non-bonded interactions.

At the same time, PyeDNA does not need to make the complete DNA structure freely deformable simply to accommodate a linker.

---

# Preserving DNA structure

The template explicitly includes

```text
dnarest_on = ...
tadfactor = ...
temp_cool3_init = ...
```

for nucleic-acid refinement.

## `dnarest_on`

Activates HADDOCK's automatic DNA restraint machinery.

DNA base pairing and backbone geometry can otherwise deteriorate during flexible molecular-dynamics refinement. DNA restraints therefore provide an additional structural prior that preserves the nucleic-acid architecture while local linker/dye degrees of freedom adapt.

This is an important second class of restraint in the PyeDNA docking problem:

```text
attachment restraints
    → enforce intended DNA ↔ linker geometry

DNA structural restraints
    → preserve the DNA duplex architecture
```

The two restraint classes encode different pieces of known physics/chemistry.

## `tadfactor`

Controls the torsion-angle-dynamics treatment used for nucleic-acid refinement.

PyeDNA explicitly exposes this because nucleic-acid systems benefit from settings different from generic protein docking.

## `temp_cool3_init`

Controls the starting temperature of the final cooling phase.

Again, the PyeDNA template uses nucleic-acid-aware refinement settings instead of relying blindly on generic protein-oriented HADDOCK defaults.

---

# Electrostatics during flexible refinement

The template includes

```text
elecflag = ...
```

inside `[flexref]`.

This determines whether electrostatic interactions are included during the actual refinement dynamics.

As in the rigid-body stage, this should be distinguished from

```text
w_elec
```

which controls the electrostatic contribution to the **score used to rank models**.

The distinction between *forces used to generate structures* and *weights used to rank structures* applies throughout HADDOCK.

---

# Flexible-refinement scoring

The template controls

```text
w_air
w_vdw
w_elec
w_desolv
w_bsa
```

during flexible refinement.

The same physical quantities therefore continue to participate in model ranking, although HADDOCK permits their relative weights to differ between docking stages.

This makes sense because the role of the scoring function changes during the workflow:

```text
rigid-body stage
    → identify promising global arrangements

flexible stage
    → distinguish locally refined structures
```

The numerical weights should therefore always be interpreted in the context of the particular HADDOCK module rather than as universal physical constants.

---

# 5. Final evaluation: `[caprieval]`

The last template stage is

```text
[caprieval]

allatoms = ...
```

`caprieval` evaluates the resulting structures and computes structural quantities that can be used to characterize and rank the generated models.

PyeDNA requests evaluation using the DNA and dye atoms rather than restricting the analysis to a protein-style subset.

The resulting HADDOCK models are still **candidate structures**.

PyeDNA subsequently processes the selected structures, restores its own residue/atom conventions, reconstructs the final DNA–dye ordering and attachment information, and prepares the selected structure for Amber.

---

# The physical picture of PyeDNA docking

The complete procedure can be thought of as minimizing and sampling an effective structural objective of the schematic form

\[
E_{\mathrm{effective}}(\mathbf R)
=
E_{\mathrm{molecular}}(\mathbf R)
+
E_{\mathrm{attachment}}(\mathbf R)
+
E_{\mathrm{DNA-restraints}}(\mathbf R),
\]

where

\[
E_{\mathrm{molecular}}
\approx
E_{\mathrm{vdW}}
+
E_{\mathrm{elec}}
+\cdots
\]

represents intermolecular molecular-mechanics interactions,

\[
E_{\mathrm{attachment}}
\]

penalizes violation of the known future DNA–linker bonds, and

\[
E_{\mathrm{DNA-restraints}}
\]

helps preserve known DNA structural features during flexible refinement.

The balance is important.

If only the attachment restraints were present, physically implausible structures could satisfy them.

If only molecular mechanics were present, there would be no reason for the dye to find the particular attachment site required by the chemical construct.

PyeDNA therefore combines

\[
\boxed{
\text{known chemical connectivity}
+
\text{molecular interactions}
+
\text{controlled flexibility}
}
\]

to obtain plausible starting structures for Amber MD.

---

# HADDOCK3 is not the final physical ensemble

A HADDOCK structure should not be interpreted as a thermally equilibrated configuration of the DNA–dye system.

The docking calculation is primarily a **structure-generation procedure**.

Its purpose in PyeDNA is to provide a physically reasonable starting geometry satisfying the known attachment constraints.

The subsequent Amber workflow performs

```text
solvation
    ↓
energy minimization
    ↓
heating / equilibration
    ↓
production molecular dynamics
```

which generates the thermally sampled molecular trajectory used for later structural and quantum-mechanical analysis.

Thus,

```text
HADDOCK
    → constrained structural search

Amber MD
    → dynamical sampling of the finalized molecular system
```

should be regarded as separate scientific stages.

---

# What users should inspect

Before accepting a HADDOCK model for Amber preparation, the structure should be inspected for obvious geometric problems.

In particular, check

- whether all intended DNA–linker attachment sites are geometrically satisfied;
- whether the linker follows a physically sensible path between DNA and dye;
- whether the dye penetrates the DNA backbone or bases;
- whether there are severe dye–DNA or dye–dye steric clashes;
- whether the DNA remains structurally intact;
- whether the orientation of the dye is reasonable for the intended construct;
- and whether different highly ranked HADDOCK models represent meaningfully different possible starting conformations.

A favorable HADDOCK score alone does not guarantee that a structure is appropriate for the intended molecular system.

> **AUTHOR INPUT REQUIRED**
>
> Add any project-specific structural acceptance criteria used for selecting a HADDOCK model before Amber preparation, for example typical acceptable attachment distances, visual checks, or cases in which a lower-ranked model should be preferred over the nominal top-ranked model.

---

# PyeDNA-specific configuration

Users normally do not edit the complete HADDOCK configuration shown above.

The full HADDOCK input is rendered by PyeDNA from

```text
data/haddock_templates/docking_config.cfg
```

using the PyeDNA structure configuration and internal HADDOCK defaults.

The user-facing docking options belong to the `structure.toml` workflow and are documented in

[Creating DNA–dye structures](../create_structure/create_structure.md).

Advanced HADDOCK options can be overridden through the PyeDNA docking configuration where supported, but such modifications should be made only when the effect on sampling, restraint strength, or scoring is understood.

For the general meaning of HADDOCK3 modules and parameters, refer to the HADDOCK3 user manual.