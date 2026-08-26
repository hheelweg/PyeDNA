# Force Fields

PyeDNA combines DNA force fields, small-molecule force fields, charge fitting, and Amber parameter files to prepare dye-labeled DNA systems for MD.

## DNA Force Fields

The current examples and defaults use `OL15`, which PyeDNA converts to the `tleap` source `leaprc.DNA.OL15`. Linker charge-restraint inference currently reads reference charges from `AMBERHOME/dat/leap/lib/DNA.OL15.lib`.

## GAFF / GAFF2

Dyes and linkers are treated as small molecules using Amber/GAFF-style atom typing. The current default is `gaff2`. PyeDNA passes the selected force field to AmberTools during `antechamber` and `parmchk2` steps and uses it to organize library directories.

## RESP Charges

The component workflows currently support RESP charge fitting. PyeDNA computes an electrostatic potential from an optimized capped molecule and runs two-stage Amber RESP fitting. Dye fitting constrains the uncapped core group to the target formal charge. Linker fitting uses fixed charge restraints inferred from an OL15-like DNA cap and a group charge restraint for the retained linker residue.

> **AUTHOR INPUT REQUIRED**
>
> Explain the scientific rationale for the current dye and linker RESP restraint choices and how users should validate resulting charges.

## MOL2

MOL2 files store atom names, atom types, coordinates, bonding, residue names, and fitted partial charges for dye/linker residue templates. PyeDNA loads MOL2 templates into `tleap` during Amber setup.

## FRCMOD

FRCMOD files store missing or customized Amber parameters such as bonds, angles, dihedrals, impropers, and nonbonded terms. PyeDNA creates dye/linker FRCMOD files with `parmchk2`. DNA-linker compatibility parameters are expected in a manually curated `connectparams.frcmod` under `LNK_DIR/connect/<dye_forcefield>/<dna_forcefield>/`.

## tleap

`tleap` combines force-field sources, MOL2 templates, FRCMOD parameters, a prepared PDB, and explicit covalent bond commands into final Amber topology and coordinate files. In PyeDNA, `tleap` writes the `prmtop`, `rst7`, and solvated PDB used for MD.
