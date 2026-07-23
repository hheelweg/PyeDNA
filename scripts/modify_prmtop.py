#!/usr/bin/env python3

import argparse
import copy
from collections import defaultdict

import parmed as pmd
from parmed.tools import addLJType

# Residue names that define the two dye molecules.
DYE_RESNAMES = {"CY3", "CY5"}

# Residue names treated as DNA.
DNA_RESNAMES = {
    "DA", "DA3", "DA5",
    "DT", "DT3", "DT5",
    "DG", "DG3", "DG5",
    "DC", "DC3", "DC5",
}


def is_dye(atom):
    """Return True when an atom belongs to DYE_RESNAMES"""
    return atom.residue.name.strip() in DYE_RESNAMES


def is_dna(atom):
    """Return True when an atom belongs to DNA."""
    return atom.residue.name.strip() in DNA_RESNAMES


def crosses_dye_dna_boundary(atoms):
    """
    A bonded term is a dye-DNA cross term when its atom list contains
    at least one dye atom and at least one DNA atom.
    Examples:
        CY3-CY3 bond:             False
        DNA-DNA bond:             False
        CY3-DNA bond:             True
        CY3-CY3-DNA angle:        True
        CY3-DNA-DNA-CY3 torsion:  True
    """
    return (
        any(is_dye(atom) for atom in atoms)
        and any(is_dna(atom) for atom in atoms)
    )


def make_nb_off(input_prmtop, output_prmtop):
    """
    Create a topology in which all nonbonded interactions involving
    DYE_RESNAMES are disabled.
    Electrostatics are disabled by setting dye charges to zero.
    Lennard-Jones interactions are disabled by assigning the dye atoms
    new LJ types with epsilon = 0.
    """
    parm = pmd.load_file(input_prmtop)

    dye_atoms = [atom for atom in parm.atoms if is_dye(atom)]

    if not dye_atoms:
        raise ValueError("No DYE_RESNAME atoms were found.")

    # ---------------------------------------------------------------
    # Electrostatic interactions
    # ---------------------------------------------------------------
    # Amber electrostatic interactions are proportional to q_i q_j.
    # Setting every dye charge to zero removes dye-environment and
    # dye-dye Coulomb interactions.
    #
    # Charges of DNA, water, ions, etc. are unchanged.
    for atom in dye_atoms:
        atom.charge = 0.0

    # ---------------------------------------------------------------
    # Lennard-Jones interactions
    # ---------------------------------------------------------------
    # atom.nb_idx identifies the original Amber LJ atom type.
    #
    # Dye atoms may have several different LJ types. We therefore group
    # them by nb_idx and create one new zero-epsilon LJ type per original
    # dye LJ type.
    atoms_by_lj_type = defaultdict(list)
    for atom in dye_atoms:
        atoms_by_lj_type[atom.nb_idx].append(atom)

    for old_nb_idx, atoms in atoms_by_lj_type.items():
        # ParmEd/Amber atom numbering in masks starts at 1,
        # whereas atom.idx in Python starts at 0.
        atom_numbers = [str(atom.idx + 1) for atom in atoms]
        mask = "@" + ",".join(atom_numbers)
        # Create a new LJ type only for this selected group.
        #
        # epsilon = 0 removes the ordinary LJ interaction.
        # radius is set to 0 as well because its value is irrelevant
        # when epsilon is zero.
        #
        # radius_14 and epsilon_14 matter for Chamber-style topologies.
        action = addLJType(
            parm,
            mask,
            radius=0.0,
            epsilon=0.0,
            radius_14=0.0,
            epsilon_14=0.0,)

        action.execute()
        print(
            f"Disabled LJ type {old_nb_idx} "
            f"for {len(atoms)} dye atoms.")

    # Update Amber's raw topology arrays from the modified objects.
    parm.remake_parm()
    # Write the new Amber topology.
    parm.save(
        output_prmtop,
        format="amber",
        overwrite=True,)
    
    print(f"Wrote {output_prmtop}")
    print(
        f"Disabled dye nonbonded interactions for "
        f"{len(dye_atoms)} atoms.")


def make_bonded_off(input_prmtop, output_prmtop):
    """
    Create a topology in which bonded terms spanning the dye-DNA
    boundary have zero force constants.
    Internal dye terms and internal DNA terms remain unchanged.
    """
    parm = pmd.load_file(input_prmtop)
    n_bonds = 0
    n_angles = 0
    n_dihedrals = 0

    # ---------------------------------------------------------------
    # Bonds
    # ---------------------------------------------------------------
    # parm.bonds comes directly from the connectivity encoded in the
    # original Amber topology. No distance-based bond inference occurs.
    for bond in parm.bonds:
        atoms = (bond.atom1, bond.atom2)
        if crosses_dye_dna_boundary(atoms):
            # Parameter types may be shared by many bonds. Therefore,
            # copy the type before changing k so unrelated bonds that
            # use the same parameter remain unchanged.
            new_type = copy.deepcopy(bond.type)
            new_type.k = 0.0
            # Register the new type in ParmEd's tracked type list so
            # it can be written correctly to the output topology.
            parm.bond_types.append(new_type)
            bond.type = new_type
            n_bonds += 1

    # ---------------------------------------------------------------
    # Angles
    # ---------------------------------------------------------------
    for angle in parm.angles:

        atoms = (
            angle.atom1,
            angle.atom2,
            angle.atom3,)

        if crosses_dye_dna_boundary(atoms):
            new_type = copy.deepcopy(angle.type)
            new_type.k = 0.0
            parm.angle_types.append(new_type)
            angle.type = new_type
            n_angles += 1

    # ---------------------------------------------------------------
    # Proper and Amber-style improper torsions
    # ---------------------------------------------------------------
    # Standard Amber topologies store both proper and periodic improper
    # torsions in parm.dihedrals.
    for dihedral in parm.dihedrals:

        atoms = (
            dihedral.atom1,
            dihedral.atom2,
            dihedral.atom3,
            dihedral.atom4,)

        if crosses_dye_dna_boundary(atoms):
            new_type = copy.deepcopy(dihedral.type)
            # A torsion may contain one Fourier term or a list of terms.
            # Set every Fourier amplitude to zero.
            if hasattr(new_type, "phi_k"):
                new_type.phi_k = 0.0
            else:
                for term in new_type:
                    term.phi_k = 0.0
            parm.dihedral_types.append(new_type)
            dihedral.type = new_type
            n_dihedrals += 1

    # Rebuild the low-level Amber arrays from the modified objects.
    parm.remake_parm()
    parm.save(
        output_prmtop,
        format="amber",
        overwrite=True,)

    print(f"Wrote {output_prmtop}")
    print(f"Disabled cross-boundary bonds:     {n_bonds}")
    print(f"Disabled cross-boundary angles:    {n_angles}")
    print(f"Disabled cross-boundary dihedrals: {n_dihedrals}")


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