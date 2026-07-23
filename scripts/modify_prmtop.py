#!/usr/bin/env python3

import argparse
import copy

import parmed as pmd
from parmed.tools import addLJType


DYE_RESNAMES = {"CY3", "CY5"}

DNA_RESNAMES = {
    "DA", "DA3", "DA5",
    "DT", "DT3", "DT5",
    "DG", "DG3", "DG5",
    "DC", "DC3", "DC5",
}


def is_dye(atom):
    return atom.residue.name.strip() in DYE_RESNAMES


def is_dna(atom):
    return atom.residue.name.strip() in DNA_RESNAMES


def crosses_dye_dna_boundary(atoms):
    return (
        any(is_dye(atom) for atom in atoms)
        and any(is_dna(atom) for atom in atoms)
    )


def make_nb_off(input_prmtop, output_prmtop):
    parm = pmd.load_file(input_prmtop)

    dye_atoms = [atom for atom in parm.atoms if is_dye(atom)]

    if not dye_atoms:
        raise ValueError("No CY3 or CY5 atoms were found.")

    # Remove dye electrostatics.
    for atom in dye_atoms:
        atom.charge = 0.0

    # Give all dye atoms a new LJ type with zero LJ interaction.
    action = addLJType(
        parm,
        ":CY3,CY5",
        radius=0.0,
        epsilon=0.0,
        radius_14=0.0,
        epsilon_14=0.0,
    )
    action.execute()

    parm.save(output_prmtop, overwrite=True)

    print(f"Wrote {output_prmtop}")
    print(f"Zeroed charge and LJ parameters for {len(dye_atoms)} dye atoms.")


def make_bonded_off(input_prmtop, output_prmtop):
    parm = pmd.load_file(input_prmtop)

    n_bonds = 0
    n_angles = 0
    n_dihedrals = 0

    for bond in parm.bonds:
        atoms = (bond.atom1, bond.atom2)

        if crosses_dye_dna_boundary(atoms):
            bond.type = copy.deepcopy(bond.type)
            bond.type.k = 0.0
            n_bonds += 1

    for angle in parm.angles:
        atoms = (
            angle.atom1,
            angle.atom2,
            angle.atom3,
        )

        if crosses_dye_dna_boundary(atoms):
            angle.type = copy.deepcopy(angle.type)
            angle.type.k = 0.0
            n_angles += 1

    for dihedral in parm.dihedrals:
        atoms = (
            dihedral.atom1,
            dihedral.atom2,
            dihedral.atom3,
            dihedral.atom4,
        )

        if crosses_dye_dna_boundary(atoms):
            dihedral.type = copy.deepcopy(dihedral.type)
            dihedral.type.phi_k = 0.0
            n_dihedrals += 1

    parm.save(output_prmtop, overwrite=True)

    print(f"Wrote {output_prmtop}")
    print(f"Zeroed cross-boundary bonds:     {n_bonds}")
    print(f"Zeroed cross-boundary angles:    {n_angles}")
    print(f"Zeroed cross-boundary dihedrals: {n_dihedrals}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_prmtop")
    parser.add_argument("nb_off_prmtop")
    parser.add_argument("bonded_off_prmtop")

    args = parser.parse_args()

    make_nb_off(
        args.input_prmtop,
        args.nb_off_prmtop,
    )

    make_bonded_off(
        args.input_prmtop,
        args.bonded_off_prmtop,
    )


if __name__ == "__main__":
    main()