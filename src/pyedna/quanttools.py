import numpy as np 
from pyscf import gto, scf, geomopt, tdscf, lib, dft, lo, solvent
import MDAnalysis as mda
import subprocess
import scipy

from joblib import dump, load
import os

# from current package
from . import utils
from . import fileproc as fp
from . import const
from . import structure
from . import trajectory
from . import pyscf_utils


# convert and optimize molecule in *.cdx (ChemDraw) format into *.pdb file (unconstrained pre-optimization)
def optimizeStructureFF(dye_name, suffix = 'preopt', stepsNo = 50000, econv = 1e-12, FF = 'UFF'):
    from openbabel import openbabel
    # (1) convert *.cdx into *.smi (SMILES string)
    command = f'obabel -icdx {dye_name}.cdx -osmi -O {dye_name}.smi'
    subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    
    # (2) read smiles string:
    with open(dye_name + '.smi', "r") as file:
        smiles = file.readline().strip()
    smiles = fr"{smiles}" # convert into raw string
    
    # (3) read SMILES string into OpenBabbel
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "pdb")
    # Create an OBMol object
    mol = openbabel.OBMol()
    # Read the SMILES string into the OBMol object
    obConversion.ReadString(mol, smiles)
    # Add explicit hydrogens to the molecule
    mol.AddHydrogens()
    # Generate 3D coordinates (preserves connectivity)
    builder = openbabel.OBBuilder()
    builder.Build(mol)  # Generate 3D geometry without changing connectivity


    # (4) perform forcefield optimization
    # econv defines the convergence tolerance for the energy, stepsNo the step number of the optimization
    forcefield = openbabel.OBForceField.FindForceField(FF)
    forcefield.Setup(mol)
    forcefield.FastRotorSearch(True)
    forcefield.ConjugateGradients(stepsNo, econv)  
    forcefield.GetCoordinates(mol)
    
    # Save the molecule as an PDB file
    output_file = dye_name + f"_{suffix}.pdb"
    obConversion.WriteFile(mol, output_file)

    # delete SMILES .smi file
    subprocess.run(f"rm -f {dye_name}.smi", shell = True)


def optimizeStructureFFSymmetry(in_pdb_file, out_pdb_file, constraint = None, point_group = None, econv = 1e-12, FF = 'UFF'):
    
    from openbabel import openbabel
    from openbabel import pybel

    # (1) import .pdb file and load molecule
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdb")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, in_pdb_file)

    # (2) function that enforces symmetry onto molecule mol 
    def enforceSymmetry(mol, point_group = "C2"):
        """
        Enforces point group symmetry by:
        1. Identifying atoms on each side of the symmetry axis.
        2. Replacing negative-side atoms with rotated positive-side atoms.
        """

        if point_group != "C2":
            raise NotImplementedError("Only C2 point group implemented")

        # (1) Identify Rotation Axis
        def getAxisInfo(mol, H_cutoff=1.7):
            """
            Identifies the most central Carbon atom and selects the closest Hydrogen within a cutoff distance to define the C2 axis.
            
            Arguments:
                mol (OBMol): OpenBabel molecule object.
                H_cutoff (float): Distance threshold (in Å) for selecting the bonded Hydrogen.
            
            Returns:
                axis_vec (numpy.array): Normalized axis direction vector.
                axis_point (numpy.array): A point on the axis.
                ref_vec (numpy.array): A perpendicular reference vector.
                axis_pair (tuple): Indices of the two atoms defining the axis (central C and closest H).
            """
            carbons, hydrogens, nitrogens = [], [], []
            
            # Extract atomic coordinates and classify atoms
            for i in range(1, mol.NumAtoms() + 1):
                atom = mol.GetAtom(i)
                symbol = atom.GetType()[0]  # Extract atomic symbol
                coord = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
                
                if symbol == "C":
                    carbons.append((i, coord))
                elif symbol == "N":
                    nitrogens.append((i, coord))
                elif symbol == "H":
                    hydrogens.append((i, coord))

            # Compute geometric center
            geometric_center = np.mean([coord for _, coord in nitrogens], axis=0)

            # Find the most central Carbon
            sorted_carbons = sorted(carbons, key=lambda c: np.linalg.norm(c[1] - geometric_center))
            central_C = sorted_carbons[0]  # Closest C (used for axis)
            second_C = sorted_carbons[1]  # Second closest C (used for classification)

            central_C_idx, central_C_coord = central_C

            # Find the closest Hydrogen to central Carbon within H_cutoff distance
            central_H_idx, central_H_coord = None, None
            min_distance = H_cutoff  # Initialize minimum distance as the cutoff value

            for H_idx, H_coord in hydrogens:
                distance = np.linalg.norm(H_coord - central_C_coord)
                if distance < min_distance:  # Check if within cutoff
                    min_distance = distance
                    central_H_idx, central_H_coord = H_idx, H_coord  # Update to the closest H

            if central_H_idx is None:
                raise ValueError(f"No Hydrogen found within {H_cutoff} Å of central Carbon {central_C_idx}.")

            # Define the C2 axis as the vector connecting the central C and its closest H
            axis_vec = central_H_coord - central_C_coord
            axis_vec /= np.linalg.norm(axis_vec)  # Normalize the vector

            # Define the perpendicular reference plane using the second closest Carbon
            second_C_idx, second_C_coord = second_C
            ref_vec = second_C_coord - central_C_coord
            ref_vec -= np.dot(ref_vec, axis_vec) * axis_vec  # Make perpendicular to the axis
            ref_vec /= np.linalg.norm(ref_vec)  # Normalize

            # print(f"Selected C2 axis between Carbon {central_C_idx} {central_C_coord} and closest Hydrogen {central_H_idx} {central_H_coord}")
            # print(f"Computed C2 axis vector: {axis_vec}")
            # print(f"Computed reference vector (from second closest C): {ref_vec}")

            return axis_vec, central_C_coord, ref_vec, (central_C_idx, central_H_idx)

        axis_vec, axis_point, ref_vec, axis_pair = getAxisInfo(mol)

        # (0) Remove bonds to avoid weird connectivity issues
        for bond in openbabel.OBMolBondIter(mol):
            mol.DeleteBond(bond)


        # (2) Identify Atoms on One Side of the C₂ Axis
        positive_atoms = []
        negative_atoms = []
        threshold = 1e-3  # Small threshold to avoid floating-point issues

        for i in range(1, mol.NumAtoms() + 1):
            atom = mol.GetAtom(i)
            if i in axis_pair:
                continue  # Skip central C-H axis atoms

            atom_pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])

            # Compute the projection onto the C₂ axis
            projection = axis_point + np.dot(atom_pos - axis_point, axis_vec) * axis_vec
            displacement = atom_pos - projection  # Vector perpendicular to axis

            # **Compute signed distance relative to the reference vector**
            signed_distance = np.dot(displacement, ref_vec)

            if signed_distance > threshold:
                positive_atoms.append((i, atom, atom_pos))  # Store index, atom, and position
            elif signed_distance < -threshold:
                negative_atoms.append((i, atom, atom_pos))

        # Ensure equal number of atoms on each side
        if len(positive_atoms) != len(negative_atoms):
            print(f"Warning: Unequal number of atoms on both sides ({len(positive_atoms)} vs {len(negative_atoms)}).")
            min_atoms = min(len(positive_atoms), len(negative_atoms))
            positive_atoms = positive_atoms[:min_atoms]
            negative_atoms = negative_atoms[:min_atoms]

        # (3) **Replace Negative-Side Atoms Without Deleting/Inserting**
        for (pos_idx, pos_atom, pos_coord), (neg_idx, neg_atom, _) in zip(positive_atoms, negative_atoms):
            # Rotate positive-side atom by 180° around C₂ axis
            projection = axis_point + np.dot(pos_coord - axis_point, axis_vec) * axis_vec
            displacement = pos_coord - projection
            rotated_coord = projection - displacement  # 180° rotated

            # **Directly overwrite the negative atom's position & element type**
            neg_atom.SetVector(*rotated_coord)
            neg_atom.SetAtomicNum(pos_atom.GetAtomicNum())  # Copy element type
        
        # (4) add bonds again
        mol.ConnectTheDots()  # Generates connectivity based on distances
        mol.PerceiveBondOrders()  # Assigns proper bond orders


    # (3) initialize forcefield
    forcefield = openbabel.OBForceField.FindForceField(FF)
    
    # (3) (optional) implement constraint 
    if constraint is not None:
        constraint_ = openbabel.OBFFConstraints() 
        # (3.1) find atoms to constrain
        atomic_number_constraint = pybel.ob.GetAtomicNum(constraint[0])
        constraint_atoms = [atom for atom in openbabel.OBMolAtomIter(mol) if atom.GetAtomicNum() == atomic_number_constraint]       # indices of atoms to constrain
        # (3.2) set constraint
        if constraint[1] == 'distance':
            assert(len(constraint_atoms) == 2)
            atom1_idx = constraint_atoms[0].GetIndex() + 1
            atom2_idx = constraint_atoms[1].GetIndex() + 1
            constraint_.AddDistanceConstraint(atom1_idx, atom2_idx, constraint[2])
            forcefield.SetConstraints(constraint_)
        else:
            raise NotImplementedError('Only distance constraints implemented!')

    # (4) optimize with C2 symmetry constraint and distance constraint on distance between P-atom
    # enforceSymmetry(mol, point_group)
    for _ in range(100):
        forcefield.Setup(mol)                                   # need to feed back symmetry-corrected coordinates into forcefield
        forcefield.ConjugateGradients(5000, econv)              # conjugate gradient optimization
        enforceSymmetry(mol, point_group)                       # enforce symmetry of molecule 
    forcefield.GetCoordinates(mol)
    enforceSymmetry(mol, point_group)                           # ensure output molecule has desired symmetry


    # (5) center geometry of molecule at (0,0,0)
    def center_molecule(mol):
        """
        Centers the molecule's center of geometry at (0,0,0) using OpenBabel.
        
        Arguments:
            mol (OBMol): OpenBabel molecule object.
        
        Returns:
            OBMol: Centered molecule.
        """

        num_atoms = mol.NumAtoms()
        if num_atoms == 0:
            raise ValueError("Molecule contains no atoms.")

        # Step 1: Compute the center of geometry
        coords = np.array([[mol.GetAtom(i).GetX(), mol.GetAtom(i).GetY(), mol.GetAtom(i).GetZ()]
                        for i in range(1, num_atoms + 1)])
        center_of_geometry = np.mean(coords, axis=0)

        # Step 2: Shift all atoms to center at (0,0,0)
        for i in range(1, num_atoms + 1):
            atom = mol.GetAtom(i)
            atom.SetVector(atom.GetX() - center_of_geometry[0],
                        atom.GetY() - center_of_geometry[1],
                        atom.GetZ() - center_of_geometry[2])

        print(f"Molecule centered at (0,0,0). Original Center of Geometry: {center_of_geometry}")
        return mol
    
    mol = center_molecule(mol)

    # (6) align symmetry axis with z-axis
    # (6.1) get symmetr axis
    def getAxisInfo(mol, H_cutoff=1.7):
        """
        Identifies the most central Carbon atom and selects the closest Hydrogen within a cutoff distance to define the C2 axis.
        
        Arguments:
            mol (OBMol): OpenBabel molecule object.
            H_cutoff (float): Distance threshold (in Å) for selecting the bonded Hydrogen.
        
        Returns:
            axis_vec (numpy.array): Normalized axis direction vector.
            axis_point (numpy.array): A point on the axis.
            ref_vec (numpy.array): A perpendicular reference vector.
            axis_pair (tuple): Indices of the two atoms defining the axis (central C and closest H).
        """
        carbons, hydrogens, nitrogens = [], [], []
        
        # Extract atomic coordinates and classify atoms
        for i in range(1, mol.NumAtoms() + 1):
            atom = mol.GetAtom(i)
            symbol = atom.GetType()[0]  # Extract atomic symbol
            coord = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
            
            if symbol == "C":
                carbons.append((i, coord))
            elif symbol == "N":
                nitrogens.append((i, coord))
            elif symbol == "H":
                hydrogens.append((i, coord))

        # Compute geometric center
        geometric_center = np.mean([coord for _, coord in nitrogens], axis=0)

        # Find the most central Carbon
        sorted_carbons = sorted(carbons, key=lambda c: np.linalg.norm(c[1] - geometric_center))
        central_C = sorted_carbons[0]  # Closest C (used for axis)
        second_C = sorted_carbons[1]  # Second closest C (used for classification)

        central_C_idx, central_C_coord = central_C

        # Find the closest Hydrogen to central Carbon within H_cutoff distance
        central_H_idx, central_H_coord = None, None
        min_distance = H_cutoff  # Initialize minimum distance as the cutoff value

        for H_idx, H_coord in hydrogens:
            distance = np.linalg.norm(H_coord - central_C_coord)
            if distance < min_distance:  # Check if within cutoff
                min_distance = distance
                central_H_idx, central_H_coord = H_idx, H_coord  # Update to the closest H

        if central_H_idx is None:
            raise ValueError(f"No Hydrogen found within {H_cutoff} Å of central Carbon {central_C_idx}.")

        # Define the C2 axis as the vector connecting the central C and its closest H
        axis_vec = central_H_coord - central_C_coord
        axis_vec /= np.linalg.norm(axis_vec)  # Normalize the vector

        # Define the perpendicular reference plane using the second closest Carbon
        second_C_idx, second_C_coord = second_C
        ref_vec = second_C_coord - central_C_coord
        ref_vec -= np.dot(ref_vec, axis_vec) * axis_vec  # Make perpendicular to the axis
        ref_vec /= np.linalg.norm(ref_vec)  # Normalize

        # print(f"Selected C2 axis between Carbon {central_C_idx} {central_C_coord} and closest Hydrogen {central_H_idx} {central_H_coord}")
        # print(f"Computed C2 axis vector: {axis_vec}")
        # print(f"Computed reference vector (from second closest C): {ref_vec}")

        return axis_vec, central_C_coord, ref_vec, (central_C_idx, central_H_idx)

    # (6.2) rotate molecule s.t. axis_vec aligns with (0,0,1) vector
    def rotate_molecule_to_z(mol, axis_vec):
        """
        Rotates the molecule so that the given C₂ axis (axis_vec) aligns with the Z-axis.
        
        Arguments:
            mol (OBMol): Centered OpenBabel molecule.
            axis_vec (numpy.array): The C₂ axis vector to be aligned with the Z-axis.

        Returns:
            OBMol: Rotated molecule.
        """

        # Normalize the input C₂ axis vector
        axis_vec = axis_vec / np.linalg.norm(axis_vec)

        # Define the target Z-axis vector
        z_axis = np.array([0, 0, 1])

        # Compute rotation axis (cross product) and angle (dot product)
        rotation_axis = np.cross(axis_vec, z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm == 0:
            print("C2 axis is already aligned with the Z-axis. No rotation needed.")
            return mol  # Already aligned

        rotation_axis /= rotation_axis_norm  # Normalize rotation axis
        rotation_angle = np.arccos(np.dot(axis_vec, z_axis))  # Angle in radians

        # Compute rotation matrix using Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        I = np.identity(3)
        R = I + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

        # Apply rotation to all atoms
        for i in range(1, mol.NumAtoms() + 1):
            atom = mol.GetAtom(i)
            pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
            rotated_pos = np.dot(R, pos)  # Apply rotation matrix
            atom.SetVector(*rotated_pos)

        print(f"Molecule rotated to align C₂ axis with the Z-axis.")
        return mol

    axis_vec, _, _, _ = getAxisInfo(mol)
    mol = rotate_molecule_to_z(mol, axis_vec)

    # (5) save the molecule as an PDB file
    obConversion.WriteFile(mol, out_pdb_file)



# finer geometry optimization incorporating C2 symmetry of chromophore molecules and disance constraint between adjacent phosphor atoms
# TODO : (old) manual function to optimize molecule with C2 symmetry
def optimizeStructureSymmetryFF(path, moleculeNamePDB, stepsNo = 50000, econv = 1e-12, FF = 'MMFF94'):

    from openbabel import openbabel

    # (0) load symmetry information
    symm_file = './createStructure/cy5/cy5_symm_info.txt'
    def readSymmetry(symm_file):
        axis_pair = None
        symmetry_pairs = []
        with open(symm_file, 'r') as file:
            for i, line in enumerate(file):
                # Parse the line as two integers
                columns = line.strip().split()
                pair = list(map(int, columns))
                # First row defines the two atoms that make the rotation axis
                if i == 0:
                    axis_pair = pair
                # every other rows defines 
                else:
                    symmetry_pairs.append(pair)
        symmetry_pairss = np.array(symmetry_pairs).T
        return axis_pair, symmetry_pairss
    # read indices (0-indexed !) for rotation axis and symmetry pairs
    axis_pair, symm_pairs = readSymmetry(symm_file)

    # (1) import .pdb file and load molecule
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdb")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, path + moleculeNamePDB + "_preopt.pdb")

    # (2) CONSTRAINED geometry optimization
    # (2.1) function to enforce C2 symmetry of molecule
    def enforceC2Symmetry(mol):
        # (1) get rotation axis vector and point in axis
        def getAxisInfo(mol, axis_pair):
            axis_atom1 = mol.GetAtom(axis_pair[0] + 1)
            axis_pos1 = np.array([axis_atom1.GetX(), axis_atom1.GetY(), axis_atom1.GetZ()])
            axis_atom2 = mol.GetAtom(axis_pair[1] + 1)
            axis_pos2 = np.array([axis_atom2.GetX(), axis_atom2.GetY(), axis_atom2.GetZ()])
            # get point in axis and rotation axis vector (normalized)
            axis_point = axis_pos1
            axis_vec = (axis_pos2 - axis_pos1) / np.linalg.norm(axis_pos2 - axis_pos1)
            return axis_vec, axis_point
        
        axis_vec, axis_point = getAxisInfo(mol, axis_pair)

        # (2) rotate atoms in first half of the molecule
        for i, atom_idx in enumerate(symm_pairs[0]):
            atom = mol.GetAtom(int(atom_idx) + 1)
            atom_pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
            # translate atom 
            atom_pos_trans = atom_pos - axis_point
            # Decompose into parallel and perpendicular components
            parallel_component = np.dot(atom_pos_trans, axis_vec) * axis_vec
            perpendicular_component = atom_pos_trans - parallel_component
            # Calculate the 180° rotated position
            atom_pos_rot= axis_point + parallel_component - perpendicular_component
            # overwrite position of symm_pairs[1] symmetry partner
            atom_symm = mol.GetAtom(int(symm_pairs[1][i]) + 1)
            atom_symm.SetVector(*atom_pos_rot)

    # (2.2) find phosphorus atoms to constrain them
    P_atoms = [atom for atom in openbabel.OBMolAtomIter(mol) if atom.GetAtomicNum() == 15]   # indices of phosphorus atoms
    # need to get 1-indexed indices
    P1_idx = P_atoms[0].GetIndex() + 1
    P2_idx = P_atoms[1].GetIndex() + 1

    # (3) optimize with C2 symmetry constraint and distance constraint on distance between P-atoms
    forcefield = openbabel.OBForceField.FindForceField(FF)
    # add distance constraint directly 
    constraint = openbabel.OBFFConstraints() 
    constraint.AddDistanceConstraint(P1_idx, P2_idx, 6.49)
    forcefield.SetConstraints(constraint)
    # NOTE : might want to play around with FastRotorSearch versus WeightedRotorSearch etc.
    # the current implementation seems to make the distance between the P-atoms smmaller, so one could choose a more hand-wavy
    # approach and aritficially make the distance in  constraint.AddDistanceConstraint() a little bit bigger than desired
    for _ in range(100):
        forcefield.Setup(mol)                           # need to feed back C2-coorected coordinates into forcefield
        forcefield.FastRotorSearch(True)
        forcefield.ConjugateGradients(1000, econv)      # conjugate gradient optimization
        enforceC2Symmetry(mol)                          # enforce C2 symmetry of molecule 
    forcefield.GetCoordinates(mol)
    enforceC2Symmetry(mol)                              # ensure output molecule has C2 symmetry

    # Save the molecule as an PDB file
    output_file = path + moleculeNamePDB + "_opt.pdb"
    obConversion.WriteFile(mol, output_file)


# NOTE : new function for geometry optimization with pyscf in the beginning
# NOTE : currently implemented for .pdb input file
# might also want to make this a constrained optimization s.t. the P-P bond-length is "roughly" equal to the one in DNA
# usage for constraint: constraint = [atom_name1, atom_name2, distance, x]
# this means that we enforce a distance of x (Angstrom) between atom_name1 and atom_name2
def geometryOptimizationDFT_gpu(in_pdb_file, dye_name, constraint = None, point_group = None, basis = '6-31g', xc = 'b3lyp', 
              density_fit = False, charge = 0, spin = 0, scf_cycles = 200, verbosity = 4):
    

    # (0) define instance of Chromophore class based on .pdb file
    dye = structure.Chromophore(mda.Universe(in_pdb_file, format = "PDB"))
    # (0) write constraint if specified
    # find atoms to constrain with specific name
    if constraint is not None:
        if not constraint[1] == 'distance':
            raise NotImplementedError("Only distance constraints implemented for DFT geometry optimization!")
        # find (1-indexed) indices of atoms to constrain
        atom_indices = np.where(dye.names == constraint[0])[0]
        atom1 = atom_indices[0] + 1
        atom2 = atom_indices[1] + 1
        # write (temporary) constraint file
        f = open(f"constraints.txt", "w")
        f.write("$set\n")
        f.write(f"distance {atom1} {atom2} {constraint[2]}")
        f.close()
        
    # (1) transform .pdb to readable format for pyscf
    molecule_converted = trajectory.Trajectory.convertChromophore(dye, conversion='pyscf')

    # (2) perform geometry optimization 
    mol = doDFT_geomopt(molecule_converted, point_group, basis, xc, density_fit, charge, spin, scf_cycles, verbosity)

    # (3) update coordinates in Angstrom
    optimized_coords = mol.atom_coords() * const.BOHR2AA
    dye.chromophore_u.atoms.positions = optimized_coords

    # (4) write .pdb file and delete "constraints.txt" file
    writePySCF2PDB(mol, dye_name)

    if os.path.isfile("constraints.txt"):
        subprocess.run("rm -f constraints.txt", shell = True)
    
    # # (5) delete input .pdb file
    # subprocess.run(f"rm -f {in_pdb_file}", shell = True)


# auxiliary function to write pyscf mol object to .pdb file
# TODO : might shift this somewhere else?
def writePySCF2PDB(pyscf_mol, dye_name):

    from openbabel import openbabel

    # Create OpenBabel Molecule Object
    obmol = openbabel.OBMol()
    for i in range(pyscf_mol.natm):
        atom_num = pyscf_mol.atom_charge(i)       # Atomic number
        x, y, z = pyscf_mol.atom_coords()[i]      # Coordinates

        atom = obmol.NewAtom()
        atom.SetAtomicNum(atom_num)
        atom.SetVector(x * const.BOHR2AA, y * const.BOHR2AA, z * const.BOHR2AA)

    # Convert to OpenBabel moeclule to .pdb file
    conv = openbabel.OBConversion()
    conv.SetOutFormat("pdb")
    pdb_filename_0 = "tmp0.pdb"
    conv.WriteFile(obmol, pdb_filename_0)


    # Read and reformat the .pdb file manually in order to enable proper "cleaning" of the file
    pdb_filename_1 = "tmp1.pdb"
    with open(pdb_filename_0, "r") as infile, open(pdb_filename_1, "w") as outfile:
        atom_index = 1  # Start from 1 for proper numbering
        for line in infile:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                # Extract relevant fields from OpenBabel's output
                parts = line.split()
                element = parts[-1]                                             # Last column should be the element name
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])     # Extract coordinates

                # Manually format lines
                formatted_line = (
                    f"HETATM{atom_index:5d}  {element:<2}  UNL     1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element}\n"
                )
                outfile.write(formatted_line)
                atom_index += 1
            else:
                # Preserve any other lines like CONECT
                outfile.write(line)  

    # "Clean" .pdb file
    structure.cleanPDB(f"tmp1.pdb", f"{dye_name}.pdb", res_code = dye_name)

    # Remove temporary .pdb files
    subprocess.run("rm -f tmp0.pdb", shell = True)
    subprocess.run("rm -f tmp1.pdb", shell = True)

    
   


#--------------------------------------------------------------------------------------------------------------------------------------------
# functions e.g. for analyzing MD trajectories


# perform DFT calculation on molecule
def doDFT(molecule, basis = '6-31g', xc = 'b3lyp', 
          density_fit = False, charge = 0, spin = 0, scf_cycles = 200, verbosity = 4):

    # (1) make PySCF molecular structure object
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin)
    mol.verbose = verbosity

    # # TODO: Marias code here (What is this for?)
    # # optimize cap
    # if opt_cap is not None:
    #     mf = scf.RHF(mol)
    #     mol = contrained_opt(mf, opt_cap)

    # (2) initialize SCF object
    # mf = scf.RKS(mol)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.max_cycle = scf_cycles               
    mf.conv_tol = 1e-5
    if density_fit:                         # optional: use density fit for accelerating computation
        mf.density_fit(auxbasis="weigend")

    # (3) run with with SMD/ddCOSMO implicit solvent model
    mf = mf.SMD()
    mf.with_solvent.method = 'DDCOSMO'
    mf.kernel()       

    # (4) output quantities of interest:
    mo = mf.mo_coeff                        # MO Coefficients
    occ = mo[:, mf.mo_occ != 0]             # occupied orbitals
    virt = mo[:, mf.mo_occ == 0]            # virtual orbitals

    return mf, occ, virt


# perform TDDFT calculation on pyscf mf object
def doTDDFT(molecule_mf, occ_orbits, virt_orbits, state_ids = [0], TDA = True):

    # (1) number of states
    nstates = len(state_ids)

    # (2) run TDDFT with or without TDA (Tamm-Dancoff approximation)
    td = molecule_mf.TDA().run(nstates = nstates) if TDA else molecule_mf.TDDFT(nstates = nstates).run()

    # (3) extract excitation energies and transition dipole moments
    exc_energies = [td.e[id] for id in state_ids]
    trans_dipoles = [td.transition_dipole()[id] for id in state_ids]

    # (4) compute oscillator strengths
    # (4.1) for all possible transitions
    # TODO : Maria does not have the 2/3 pre-factor (find reason for this!)
    osc_strengths = [2/3 * exc_energies[i] * np.linalg.norm(trans_dipoles[i])**2 for i in range(len(exc_energies))]
    # (4.2) find strongest transition
    osc_idx = np.argmax(osc_strengths) if not any(np.array(osc_strengths) > 0.1) else np.argwhere(np.array(osc_strengths) > 0.1)[0][0]

    # (5) compute TDM (Tarnsition Density Matrix) for all states
    # td.xy[i] is tuple (X_i, Y_i) with X_i contains the expansion coefficients for the excitation part of the i-th excited state
    # and Y_1 the expansion coefficients for the de-excitation part; in TDDFT td.xy[i][0] quantifies how the virtual orbitals mix
    # with the occupied orbital in the i-th excitation
    tdms = [np.sqrt(2) * occ_orbits.dot(td.xy[id][0]).dot(virt_orbits.T) for id in state_ids]

    return exc_energies, trans_dipoles, osc_strengths, tdms, osc_idx


# perform constrained optimization on capped H-atoms first
def constrainedOptimization(mf, molecule_idx, freeze_atom_string):

    from pyscf.geomopt.geometric_solver import optimize

    # (1) Write (temporary) constraints file
    f = open(f"constraints_{molecule_idx}.txt", "w")
    f.write("$freeze\n")
    f.write("xyz " + freeze_atom_string)
    f.close()

    # (2) Load parameters for constrained optimization
    params = {"constraints" : f"constraints_{molecule_idx}.txt",
              "verbose"     : 0
              }

    # (3) Store gradients for analysis and optimize molecule subject to constraints
    gradients = []
    def callback(envs):
        gradients.append(envs['gradients'])

    molecule_eq = optimize(mf, maxsteps=10, callback=callback, **params)

    # (4) delete constraints.txt file
    subprocess.run(f"rm -f constraints_{molecule_idx}.txt", shell = True)

    return molecule_eq


def getMolPySCF(molecule,  
                basis = '6-31g',
                charge = 0, 
                spin = 0
                ):
    # mol object
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin)
    return mol

# do DFT with GPU support
# TODO : merge with doDFT()
def doDFT_gpu(molecule, molecule_id,  
              basis = '6-31g',
              xc = 'b3lyp', 
              density_fit = False,
              charge = 0, 
              spin = 0, 
              scf_cycles = 200, 
              verbosity = 4, 
              optimize_cap = False
              ):
    
    # (0) import gpu4pyscf and GPU support
    from gpu4pyscf import scf, solvent, tdscf
    from gpu4pyscf.dft import rks
    import cupy as cp


    # (1) make PySCF molecular structure object 
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin)
    mol.verbose = verbosity


    # (2) (optional) only optimize the capped atoms first
    # NOTE : the capped atoms are the last ones to have been added to molecule, so their indices are the last two ones
    if optimize_cap:
         # optimize with density fitting
         mf_opt = rks.RKS(mol, xc = xc).density_fit()
         mf_opt.verbose = 0
         # NOTE : atoms are 1-index for pyscf geometric solvers
         freeze_atom_string = f'1-{len(molecule) - 2}'
         mol = constrainedOptimization(mf_opt, molecule_id, freeze_atom_string)


    # (3) initialize SCF object
    mf = rks.RKS(mol)
    mf.xc = xc
    mf.max_cycle = scf_cycles               
    mf.conv_tol = 1e-6                      
    # mf = mf.SMD()                             # TODO : look up this model
    # mf.with_solvent.method = 'DDCOSMO'        # COSMO implicit solvent model 
    mf = mf.PCM()
    mf.with_solvent.method = 'COSMO'
    if density_fit:                             # optional: use density fit for accelerating computation
        mf.density_fit()

    # (4) run DFT
    mf.kernel()       

    # (5) output
    mo = mf.mo_coeff                            # MO coefficients
    occ = mo[:, mf.mo_occ != 0]                 # occupied orbitals
    virt = mo[:, mf.mo_occ == 0]                # virtual orbitals
    orbit_enrgs = mf.mo_energy                  # MO energies

    return mol, mf, occ, virt, orbit_enrgs


# TODO : this is just a test function (i.e. delete this!)
def checkSymmetryPYSCF(molecule, point_group = None, basis = '6-31g', xc = 'b3lyp', 
              density_fit = False, charge = 0, spin = 0, scf_cycles = 200, verbosity = 4):
    #from gpu4pyscf import dft
    from pyscf import dft, symm, gto

    # (1) make PySCF molecular structure object 
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin,
                unit = 'Angstrom',
                )
    mol.verbose = verbosity

    # (1.1) (optional) check if point group aligned with structure
    if point_group is not None:
        # symmetry detection and initialization
        # NOTE : one might have to make this a little bit bigger so as to make the symmetry identification a bit more robust aginst slighlty
        # distorted molecular geometries
        symm.geom.TOLERANCE = 0.1
        mol.symmetry = True
        mol.symmetry_subgroup = point_group
        mol.build()
    print(f"*** PySCF detected point group: {mol.groupname}", flush = True)


# do DFT with geometry optimization in each step
def doDFT_geomopt(molecule, point_group = None, basis = '6-31g', xc = 'b3lyp', 
              density_fit = False, charge = 0, spin = 0, scf_cycles = 200, verbosity = 4):
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"                       # Assign GPU
    
    # (0) import gpu4pyscf and GPU support
    from gpu4pyscf import scf, solvent, tdscf
    from gpu4pyscf.dft import rks
    #from gpu4pyscf import dft
    from pyscf import dft, symm
    from pyscf.geomopt.geometric_solver import optimize

    # (1) make PySCF molecular structure object 
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin,
                unit = 'Angstrom',
                )
    mol.verbose = verbosity

    # (1.1) (optional) check if point group aligned with structure
    if point_group is not None:
        # symmetry detection and initialization
        mol.symmetry = True
        mol.symmetry_subgroup = point_group
        mol.build()
    print(f"*** PySCF detected point group: {mol.groupname}", flush = True)

    # (2) geometry optimization
    mf_GPU = dft.RKS(mol, xc = xc)
    mf_GPU.grids.level = 8
    mf_GPU = mf_GPU.PCM()
    mf_GPU.with_solvent.method = 'COSMO'
    # optional : constraint parameters
    params = {}
    if os.path.isfile("constraints.txt"):
        params["constraints"] = "constraints.txt"
    mol_eq = optimize(mf_GPU, maxsteps=20, **params)

    # (3) get DFT at optimized geometry
    mf = dft.RKS(mol_eq)
    mf.xc = xc
    mf.max_cycle = scf_cycles               
    mf.conv_tol = 1e-10   
    mf = mf.PCM()
    mf.with_solvent.method = 'COSMO'
    mf.kernel() 

    return mol_eq



# do TDDFT with GPU support
# TODO : merge with doTDDFT()
def doTDDFT_gpu(molecule_mol, molecule_mf, occ_orbits, virt_orbits, quantum_dict,
                fragments = None,  
                state_ids = [0],
                TDA = False,
                singlet = True,
                ):

    # (0) import gpu4pyscf and GPU support
    from gpu4pyscf import scf, solvent, tdscf
    from gpu4pyscf.dft import rks
    import cupy as cp

    print('quantum dict', quantum_dict, flush = True)
    print('fragments', fragments, flush = True)

    # (1) number of states
    nstates = len(state_ids)

    # (2) run TDDFT with or without TDA (Tamm-Dancoff approximation)
    molecule_td = molecule_mf.TDA().run(nstates = nstates) if TDA else molecule_mf.TDDFT().run(nstates = nstates)

    # (3) extract excitation energies and transition dipole moments
    exc_energies = [molecule_td.e[id] for id in state_ids]
    trans_dipoles = [molecule_td.transition_dipole()[id] for id in state_ids]

    # (4) compute oscillator strengths
    # (4.1) for all possible transitions
    osc_strengths = [2/3 * exc_energies[i] * np.linalg.norm(trans_dipoles[i])**2 for i in range(len(exc_energies))]
    # (4.2) find strongest transition
    osc_idx = np.argmax(osc_strengths) if not any(np.array(osc_strengths) > 0.1) else np.argwhere(np.array(osc_strengths) > 0.1)[0][0]

    # (5) compute TDM (Transition Density Matrix) for all states
    # td.xy[i] is tuple (X_i, Y_i) with X_i contains the expansion coefficients for the excitation part of the i-th excited state
    # and Y_1 the expansion coefficients for the de-excitation part; in TDDFT td.xy[i][0] quantifies how the virtual orbitals mix
    # with the occupied orbital in the i-th excitation
    tdms = [cp.sqrt(2) * cp.asarray(occ_orbits).dot(cp.asarray(molecule_td.xy[id][0])).dot(cp.asarray(virt_orbits).T) for id in state_ids]

    # (6) write 'baseline' quantities
    tddft_output = {}
    if quantum_dict['exc']:
        tddft_output['exc'] = np.array(exc_energies)
    if quantum_dict['tdm']:
        tddft_output['tdm'] = np.array([tdm.get() for tdm in tdms])
    if quantum_dict['dip']:
        tddft_output['dip'] = np.array(trans_dipoles)
    if quantum_dict['osc']:
        tddft_output['osc'] = np.array(osc_strengths)
    if quantum_dict['idx']:
        tddft_output['idx'] = osc_idx
    
    # (07/29/25) NOTE : added this for interference TDMs for interference term between transition density matrices
    # for INTRAMOLECULAR transfer between the TDMs within one molecules
    # we here only assume two states
    # TDDFT excitation vectors
    virt_orbits = cp.asnumpy(virt_orbits)
    occ_orbits = cp.asnumpy(occ_orbits)
    x1, y1 = molecule_td.xy[0]
    x2, y2 = molecule_td.xy[1]
    x1_np, y1_np = cp.asnumpy(x1), cp.asnumpy(y1)
    x2_np, y2_np = cp.asnumpy(x2), cp.asnumpy(y2)

    x1x2T = x1_np @ x2_np.T     
    y1y2T = y1_np.T @ y2_np     
    gamma_12 = occ_orbits @ x1x2T @ occ_orbits.T + virt_orbits @ y1y2T @ virt_orbits.T
    #tddft_output['tdm_inter'] = cp.asnumpy(molecule_mf.get_ovlp()) #gamma_12
    tddft_output['tdm_inter'] = gamma_12

    # # NOTE : delete this (this is just for debugging)
    # # Get NTOs for state A
    # # Get number of occupied and virtual orbitals
    # nocc = molecule_mf.mol.nelectron // 2
    # nmo = molecule_mf.mo_coeff.shape[1]
    # nvir = nmo - nocc

    # NOTE : I had this in during the testing for intramolecular rates
    # # Get NTOs (hole lives in occ, particle in vir space)
    # r, c = molecule_td.get_nto(state=0)
    # # Promote NTOs to full MO space
    # hole_mo = np.zeros(nmo)
    # elec_mo = np.zeros(nmo)
    # hole_mo[:nocc] = r if r.ndim == 1 else r[:, 0]
    # elec_mo[nocc:] = c if c.ndim == 1 else c[:, 0]
    # # Transform to AO basis
    # hole_ao = cp.asnumpy(molecule_mf.mo_coeff) @ cp.asnumpy(r[:, 0])
    # elec_ao = cp.asnumpy(molecule_mf.mo_coeff) @ cp.asnumpy(c[:, 0])
    # from pyscf.tools import cubegen
    # cubegen.orbital(molecule_mol, 'holeA.cube', hole_ao)
    # cubegen.orbital(molecule_mol, 'elecA.cube', elec_ao)

    # (7) Mulliken analysis for excited states
    if quantum_dict["mull_pops"] or quantum_dict["mull_chrgs"]:
        tddft_output['mull_pops'], tddft_output['mull_chrgs'] = doMullikenAnalysis(molecule_mf, molecule_mol, tdms, state_ids=state_ids)

    # (8) orbital participation analysis for excited states
    if quantum_dict['OPA']:
        tddft_output['OPA'] = doOrbitalParticipationAnalysis(molecule_mol, molecule_td, fragments, state_ids=state_ids, TDA=TDA)


    return tddft_output


# do Mulliken analysis for all (excited) states
def doMullikenAnalysis(molecule_mf, molecule_mol, molecule_tdms, state_ids = [0]):

    atom_pops, atom_charges = [], []
    S = molecule_mf.get_ovlp()
    for i in range(len(state_ids)):

        tdm = molecule_tdms[i]
        assert tdm.shape == (molecule_mol.nao, molecule_mol.nao)

        pop, charges = pyscf_utils.mulliken_pop(molecule_mol, S, tdm)

        atom_pops.append(pop)
        atom_charges.append(charges)
    
    return atom_pops, atom_charges

# do orbital population analysis for all (excited) states
# TODO : provide some more comments on the quantities here
def doOrbitalParticipationAnalysis(molecule_mol, molecule_td, fragments, state_ids = [0], TDA = False):

    # (1) Map AO index -> atom index
    ao2atom = np.array([label[0] for label in molecule_mol.ao_labels(fmt=None)])

    # (2) get fragment maps (fragment_map[ao_idx] = 0 if in frag0, 1 if in frag1, -1 otherwise)
    # TODO : generalize this for more than just two fragments
    assert len(fragments) == 2
    fragment_map = np.full(len(ao2atom), fill_value=-1)
    for frag_id, atom_indices in enumerate(fragments):
        for atom in atom_indices:
            fragment_map[ao2atom == atom] = frag_id
    
    # (3) get MO coefficients
    C = molecule_td._scf.mo_coeff            
    nmo = C.shape[1]

    # (4) compute how much each MO is localized on each fragment
    mo_weights = np.zeros((nmo, len(fragments)))

    S = molecule_td._scf.get_ovlp()  # or mf.get_ovlp() if passed separately
    for mo_idx in range(nmo):
        coeff = C[:, mo_idx]
        for frag_id in range(len(fragments)):
            frag_mask = (fragment_map == frag_id)
            coeff_frag = coeff[frag_mask]
            S_frag = S[np.ix_(frag_mask, frag_mask)]
            mo_weights[mo_idx, frag_id] = coeff_frag @ S_frag @ coeff_frag

    # (5) analyze the excitations 
    result = []
    for state_id in state_ids:
        X, Y = molecule_td.xy[state_id]

        nocc = molecule_td._scf.mol.nelec[0]  # for closed-shell RKS
        nvirt = C.shape[1] - nocc
        transitions = [(i, j) for i in range(nocc) for j in range(nocc, nocc + nvirt)] 

        if TDA:
            amps = X.flatten()                                                      # TDA approximation
        else:
            amps = (X + Y).flatten()                                                # full-TDDFT

        frag_contributions = np.zeros((len(fragments), len(fragments)))  

        for amp, (i_occ, i_virt) in zip(amps, transitions):
            for frag_h in range(len(fragments)):
                for frag_p in range(len(fragments)):
                    weight = mo_weights[i_occ, frag_h] * mo_weights[i_virt, frag_p]
                    frag_contributions[frag_h, frag_p] += (amp**2) * weight
        
        # normalize 
        total = frag_contributions.sum()
        if total > 0:
            frag_contributions /= total
        result.append(frag_contributions)
    
    return result
        




# NOTE : function that calls python ssubprocess to perform DFT/TDDFT on individual GPUs with PySCF
# TODO : make this more flexible with regards to the path where the launcher (DFT_gpu.py) is
def launchQMdriver(molecule_no, gpu_ids):
    """Launch a DFT/TDDFT calculation on a specific GPU."""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] =  ",".join(str(g) for g in gpu_ids)

    # driver for QM (DFT/TDDFT) calculations
    qm_driver_module = 'pyedna.qm_driver'

    # run driver for QM calcualtions as module
    cmd = f"python -u -m {qm_driver_module} {molecule_no}"
    process = subprocess.Popen(cmd, env=env,
                                shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True
                                )

    return process


# do PySCF on molecules = [mol1, mol2] where mol are the nuclear coordinates for PySCF calculations
# TODO : make this also without GPU-support depending on the available resources
def doQM_gpu(molecules, output_keys, fragments = None, verbosity = 0):

    # verbosity = 0 : suppress all the output from the QM calculations (default)
    # verbosity = 1 : only print STDOUT of QM calculations
    # verbosity = 2 : only print STDERR of QM calculations (for debugging)

    # (0) initialize output dictionary for quantities of interest
    # [] stores data for both molecules in a list-type fashion
    output = {key: [] for key, value in output_keys.items() if value}

    # (1) run molecules on different GPUs in parallel
    procs = []
    for i, molecule in enumerate(molecules):
        # create pyscf input for subprocess and store in cache
        dump(molecule, f"input_{i}.joblib")
        if fragments is not None:
            dump(fragments[i], f"fragments_{i}.joblib")
        # run subprocess on single GPU
        procs.append(launchQMdriver(i, gpu_ids = [i]))
    
    # wait for subprocesses to finish and print STDOUT or STDERR if desired
    for i, molecule in enumerate(molecules):
        if verbosity == 0:
            continue
        elif verbosity == 1: # only STDOUT
            for line in procs[i].stdout:
                print(f"[molecule {i} STDOUT] {line}", end="")
        elif verbosity == 2: # only STDERR
            for line in procs[i].stderr:
                print(f"[molecule {i} STDERR] {line}", end="")
        elif verbosity == 3: # both STDOUT and STDERR
            for line in procs[i].stdout:
                print(f"[molecule {i} STDOUT] {line}", end="")
            for line in procs[i].stderr:
                print(f"[molecule {i} STDERR] {line}", end="")

        procs[i].wait()
    

    # (2) load and store relevant data from output of subprocesses
    # TODO : flexibilize this for quantities we are interested in
    for i, molecule in enumerate(molecules):
        for key in output:
            output[key].append(load(f"{key}_{i}.joblib"))

    # (3) clean subprocess cache 
    utils.cleanCache()

    # returns output dictionary with keys specified in output_keys
    return output


# (intermolecular) coupling terms for the computation cJ and cK of molecule A and molecule B
# NOTE : this returns (by default) the couplings in Hartree units 
def getInterCJCK(molA, molB, tdmA, tdmB, get_cK = False):

    from pyscf.scf import jk, _vhf

    # (0) check that dimensions are correct
    assert(tdmA.shape == (molA.nao, molA.nao))
    assert(tdmB.shape == (molB.nao, molB.nao))

    # (1) merge separate molecules together and set up joint density matrix
    molAB = molA + molB
    dm_AB = scipy.linalg.block_diag(tdmA, tdmB) 
    
    # (2) set of HF infrastucture for fast integral evaluation
    # 'int2e' specifies two-electron integrals, 'CVHFnrs8_prescreen' use prescreen options to reduce computational time
    # key idea : instead of directly computing all (ij∣kl) integrals, PySCF uses prescreening techniques to skip irrelevant terms.
    vhfopt = _vhf.VHFOpt(molAB, 'int2e', 'CVHFnrs8_prescreen', 'CVHFsetnr_direct_scf', 'CVHFsetnr_direct_scf_dm')                    
    vhfopt.set_dm(dm_AB, molAB._atm, molAB._bas, molAB._env)             # enables density-based prescreening
    vhfopt._dmcondname = None

    # (3) compute Coulomb integrals
    with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0, molA.nbas, 0, molA.nbas, molA.nbas, molAB.nbas, molA.nbas, molAB.nbas) 
        vJ = jk.get_jk(molAB, tdmB, 'ijkl,lk->s2ij', shls_slice=shls_slice, vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, tdmA)
    
    # (4) compute Exchange integrals
    if get_cK == True:
        with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0, molA.nbas , molA.nbas, molAB.nbas, molA.nbas, molAB.nbas, 0, molA.nbas)  
            vK = jk.get_jk(molAB, tdmB, 'ijkl,jk->il', shls_slice=shls_slice, vhfopt=vhfopt, aosym='s1', hermi=0)
            cK = np.einsum('ia,ia->', vK, tdmA)
        return cJ, cK
    else: 
        return cJ, 0


# (intramolecular) coupling terms for the computation cJ and cK of molecule 
# NOTE : this returns (by default) the couplings in Hartree units 
def getIntraCJCK(mol, tdmA, tdmB, get_cK=False):

    from pyscf.scf import jk, _vhf

    assert tdmA.shape == (mol.nao, mol.nao)
    assert tdmB.shape == (mol.nao, mol.nao)



    # (1) Use original molecule
    dm = tdmB  # this will be used for prescreening

    # (2) Set up HF infrastructure for prescreened integral evaluation
    vhfopt = _vhf.VHFOpt(mol, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf', 'CVHFsetnr_direct_scf_dm')
    vhfopt.set_dm(dm, mol._atm, mol._bas, mol._env)
    vhfopt._dmcondname = None

    # (3) Compute Coulomb integral ⟨tdmA|J[tdmB]⟩
    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        # shls_slice = (i, j, k, l, ...) → match ⟨ij|kl⟩ index order
        shls_slice = (0, mol.nbas, 0, mol.nbas,
                      0, mol.nbas, 0, mol.nbas)
        vJ = jk.get_jk(mol, tdmB, 'ijkl,lk->s2ij',
                       shls_slice=shls_slice, vhfopt=vhfopt,
                       aosym='s4', hermi=1)
        cJ = np.einsum('ij,ij->', vJ, tdmA)

    # (4) Compute Exchange integral ⟨tdmA|K[tdmB]⟩ if requested
    if get_cK:
        with lib.temporary_env(vhfopt._this.contents,
                               fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0, mol.nbas, 0, mol.nbas,
                          0, mol.nbas, 0, mol.nbas)
            vK = jk.get_jk(mol, tdmB, 'ijkl,jk->il',
                           shls_slice=shls_slice, vhfopt=vhfopt,
                           aosym='s1', hermi=0)
            cK = np.einsum('ij,ij->', vK, tdmA)
    else:
        cK = 0.0

    return cJ, cK


# compute coupling terms ('cJ', 'cK', 'electronic', 'both') for the states (S_0^A , S_{stateB + 1}^B) <--> (S_{stateA + 1}^A, S_0^B)
# 'cJ' only returns the electrostatic interaction, 'cK' only the exchange interaction, 'electronic' returns 2 * cJ - cK
# NOTE : stateA and stateB are zero-indexed here so stateA = 0 corresponds to the first excited state of molecule A etc.
# stateA and stateB default to 0 to for the transition (S_0^A , S_1^B) <--> (S_1^A, S_0^B)
def getVCoulombic(mols, tdms, states, coupling_type = 'electronic'):

    from pyscf.tools import cubegen
    import numpy as np


    #assert(len(mols) == len(tdms) == len(states))
    if len(states) == 2:
        intermolecular, intramolecular = True, False
    elif len(states) == 1:
        intermolecular, intramolecular = False, True

    # if intermolecular:
    #     stateA, stateB = states[0], states[1]
    #     molA, molB = mols[0], mols[1]
    #     tdmA, tdmB = tdms[0][stateA], tdms[1][stateB]
    # if intramolecular:
    #     state = states[0]
    #     mol = mols[0]
    #     tdm = tdms[0][state]

    # # NOTE : for intermolecular
    # stateA, stateB = states[0], states[1]
    # molA, molB = mols[0], mols[1]
    # tdmA, tdmB = tdms[0][stateA], tdms[1][stateB]

    # NOTE : for intramolecular
    stateA, stateB = states[0], states[1]
    molA, molB = mols[0], mols[0]
    mol = molA
    tdmA, tdmB = np.squeeze(tdms[0][stateA]), np.squeeze(tdms[0][stateB])

    # print(tdmA.shape, tdmB.shape)
    # print("inner product =",  np.trace(tdmA.conj().T @ tdmB))
    # print("norm A =",  np.trace(tdmA.conj().T @ tdmA))
    # print("norm B =",  np.trace(tdmB.conj().T @ tdmB))

    # Monkey patch _write_cube to use safer float formatting
    # Define your function (copy/paste unchanged)
    import re 
    def fix_cube_spacing(infile, outfile=None, min_abs_val=1e-90):
        outfile = outfile or infile + ".fixed"

        float_fix_pattern = re.compile(r'([Ee][+-]\d{2})(?=[+-]\d)')

        with open(infile, 'r') as f_in, open(outfile, 'w') as f_out:
            lines = f_in.readlines()

        # First 6 lines are cube header
        header_lines = lines[:6]

        # Line 3 gives atom count
        num_atoms = int(header_lines[2].split()[0])
        atom_lines = lines[6:6 + num_atoms]

        # Rest is voxel data
        data_lines = lines[6 + num_atoms:]

        # Write header and atom lines unchanged
        with open(outfile, 'w') as f_out:
            f_out.writelines(header_lines)
            f_out.writelines(atom_lines)

            # Process voxel data lines
            for line in data_lines:
                fixed_line = float_fix_pattern.sub(r'\1 ', line.strip())

                tokens = fixed_line.split()
                cleaned = []
                for tok in tokens:
                    try:
                        val = float(tok)
                        if abs(val) < min_abs_val:
                            val = 0.0
                        cleaned.append(f"{val: .12E}")
                    except ValueError:
                        cleaned.append(tok)

                for i in range(0, len(cleaned), 6):
                    f_out.write(" ".join(cleaned[i:i+6]) + "\n")

        print(f"Cleaned cube file saved to: {outfile}")
        return outfile



    threshold = 1e-99

    print(tdmA.shape)
    print(tdmB.shape)
    print(type(tdmA))
    print(type(tdmB))
    print(np.min(tdmA), np.min(tdmB))

    from pyscf.dft import numint

    def write_thresholded_density_cube(mol, tdm, filename, nx=80, ny=80, nz=80, margin=3.0, threshold=1e-99):
        """
        Generate and write a thresholded density cube file from a transition density matrix.

        Parameters
        ----------
        mol : pyscf.gto.Mole
            PySCF molecule object.
        tdm : np.ndarray
            AO-basis transition density matrix (nAO x nAO).
        outfile : str
            Path to save the cube file.
        nx, ny, nz : int
            Number of grid points in x, y, z directions.
        margin : float
            Angstroms of padding around the molecule bounding box.
        threshold : float
            Minimum absolute value to keep in real-space density; values below are thresholded.
        """
        # Build grid bounding box
        coords = mol.atom_coords()
        origin = coords.min(axis=0) - margin
        boxsize = coords.max(axis=0) - coords.min(axis=0) + 2 * margin
        dx = boxsize[0] / nx
        dy = boxsize[1] / ny
        dz = boxsize[2] / nz

        x = np.linspace(origin[0], origin[0] + dx * (nx - 1), nx)
        y = np.linspace(origin[1], origin[1] + dy * (ny - 1), ny)
        z = np.linspace(origin[2], origin[2] + dz * (nz - 1), nz)

        grid = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T

        # AO basis on grid
        ao = dft.numint.eval_ao(mol, grid)

        # Compute real-space scalar field from TDM
        rho = np.einsum('pi,ij,pj->p', ao, tdm, ao)

        # Threshold very small values
        small = np.abs(rho) < threshold
        rho[small] = np.sign(rho[small]) * threshold
        rho = rho.reshape((nx, ny, nz))

        # Start writing cube
        with open(filename, 'w') as f:
            f.write("CUBE FILE\nGenerated from PySCF TDM\n")
            f.write(f"{mol.natm:5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n")
            f.write(f"{nx:5d}{dx:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
            f.write(f"{ny:5d}{0.0:12.6f}{dy:12.6f}{0.0:12.6f}\n")
            f.write(f"{nz:5d}{0.0:12.6f}{0.0:12.6f}{dz:12.6f}\n")

            for ia in range(mol.natm):
                charge = mol.atom_charge(ia)
                x, y, z = mol.atom_coord(ia)
                f.write(f"{int(charge):5d}{0.0:12.6f}{x:12.6f}{y:12.6f}{z:12.6f}\n")

            # Write volumetric data in blocks of 6
            count = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        val = rho[i, j, k]
                        f.write(f"{val:13.5E}")
                        count += 1
                        if count % 6 == 0:
                            f.write("\n")
                    if count % 6 != 0:
                        f.write("\n")

        print(f"✓ Written cube file to: {filename}")

    
    write_thresholded_density_cube(mol, tdmA, 'tdmA.cube')
    write_thresholded_density_cube(mol, tdmB, 'tdmB.cube')
    


    # # Step 2: Fix the formatting (overwrite original file)
    # _ = fix_cube_spacing('tdmA.cube', outfile='tdmA_1.cube')  # overwrites in-place
    # _ = fix_cube_spacing('tdmB.cube', outfile='tdmB_1.cube')  # overwrites in-place

    # cubegen.density(mol, 'tdmA.cube', tdmA_cleaned, nx=80, ny=80, nz=80)
    # cubegen.density(mol, 'tdmB.cube', tdmB_cleaned, nx=80, ny=80, nz=80)

    # ----------------------- OLD --------------------------------
    # manually obtained indices for one of the fragments
    # frag_A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    # frag_B = [50, 51, 52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103]
    # # compute projector onto frag_ids
    # P_A = np.zeros((mol.nao_nr(), mol.nao_nr()))
    # for i in frag_A:
    #     P_A[i, i] = 1.0
    # # print('projector', P_A)
    # # print('projector shape', P_A.shape)
    # # compute projector onto rest
    # P_B = np.eye(mol.nao_nr()) - P_A
    # gamma_A = P_A @ tdmA @ P_A
    # gamma_B = P_B @ tdmB @ P_B
    
    # # NOTE : for intermolecular (test-wise)
    # tdm_inter = tdms_inter[0]
    # tdm_inter_T = np.conj(tdm_inter).T
    # print(tdm_inter.shape)
    # print(np.trace(tdm_inter.conj().T @ tdm_inter))
    # # compute new tdms from superposition information
    # ratio = np.sqrt(0.65/0.35)
    # theta = np.arctan(ratio)
    # gamma_A = (np.cos(theta)**2) * tdmA + (np.sin(theta)**2) * tdmB + np.sin(theta)*np.cos(theta)*(tdm_inter + tdm_inter_T)
    # gamma_B = (np.sin(theta)**2) * tdmA + (np.cos(theta)**2) * tdmB - np.sin(theta)*np.cos(theta)*(tdm_inter + tdm_inter_T)

    # S = tdms_inter[0]
    # print('S shape', S.shape)
    # gamma_A = project_tdm_fragment_lowdin(tdmA, S, frag_A, scaling=np.trace(tdmA.conj().T @ tdmA))
    # gamma_B = project_tdm_fragment_lowdin(tdmB, S, frag_B, scaling=np.trace(tdmB.conj().T @ tdmB))

    # # run some checks
    # print(gamma_A.shape, gamma_B.shape)
    # inner_product = np.trace(gamma_A.conj().T @ gamma_B)
    # print("inner product =", inner_product)
    # norm_A = np.trace(gamma_A.conj().T @ gamma_A)
    # norm_B = np.trace(gamma_B.conj().T @ gamma_B)
    # print("norm A =", norm_A)
    # print("norm B =", norm_B)
    # -------------------------------------------------------------------------


    if coupling_type in ['electronic', 'cK']:
        # if intermolecular:
        #cJ, cK = getInterCJCK(molA, molB, tdmA, tdmB, get_cK=True)
        #if intramolecular:
        #cJ, cK = getIntraCJCK(mol, gamma_A, gamma_B, get_cK=True)
        cJ, cK = getIntraCJCK(mol, tdmA, tdmB, get_cK=True)
    elif coupling_type in ['cJ']:
        # if intermolecular:
        #cJ, _ = getInterCJCK(molA, molB, tdmA, tdmB, get_cK=False)
        # if intramolecular:
        #cJ, _ = getIntraCJCK(mol, gamma_A, gamma_B, get_cK=False)
        cJ, _ = getIntraCJCK(mol, tdmA, tdmB, get_cK=False)
    else:
        raise NotImplementedError("Invalid coupling type specified!")
    
    results = {'coupling cJ': cJ, 'coupling cK': cK}
    if coupling_type == 'electronic':                                     
        results['coupling V_C'] = 2 * cJ - cK                               # total electronic coupling
    elif coupling_type == 'cK':
        results['coupling V_C'] = - cK                                      # exchange-interaction part of the electronic coupling
    elif coupling_type == 'cJ':
        results['coupling V_C'] = 2 * cJ                                    # electrostatic-interaction part of the electronic coupling
    return results


# get excitation energies for specified transition between states
def getExcEnergiesTransition(excs, states, molecule_names = ["D", "A"], excitation_energy_type = 'default'):

    assert(len(excs) == len(states) == len(molecule_names))
    if len(excs) == 2:
        intermolecular, intramolecular = True, False
    elif len(excs) == 1:
        intermolecular, intramolecular = False, True

    results = {}
    if intermolecular:
        stateA, stateB = states[0], states[1]
        excA, excB = excs[0], excs[1]
        results[f'energy {molecule_names[0]}'] = excA[stateA]
        results[f'energy {molecule_names[1]}'] = excB[stateB]
    if intramolecular:
        state = states[0]
        exc = excs[0]
        results[f'energy {molecule_names[0]}'] = exc[state]
    return results


# get oscillator strengths for specified states
def getOscillatorStrengths(oscs, states, molecule_names = ["D", "A"], osc_strength_type = 'default'):

    assert(len(oscs) == len(states) == len(molecule_names))
    if len(oscs) == 2:
        intermolecular, intramolecular = True, False
    elif len(oscs) == 1:
        intermolecular, intramolecular = False, True

    results = {}
    if intermolecular:
        stateA, stateB = states[0], states[1]
        oscA, oscB = oscs[0], oscs[1]
        results[f'osc_strength {molecule_names[0]}'] = oscA[stateA]
        results[f'osc_strength {molecule_names[1]}'] = oscB[stateB]
    if intramolecular:
        state = states[0]
        osc = oscs[0]
        results[f'osc_strength {molecule_names[0]}'] = osc[state]
    return results

# get transition dipole moments (vector magnitude) for specified states
def getTransitionDipoles(dips, states, molecule_names = ["D", "A"], dipole_moment_type = 'default'):
    
    assert(len(dips) == len(states) == len(molecule_names))
    if len(dips) == 2:
        intermolecular, intramolecular = True, False
    elif len(dips) == 1:
        intermolecular, intramolecular = False, True

    results = {}
    if intermolecular:
        stateA, stateB = states[0], states[1]
        dipA, dipB = dips[0], dips[1]
        results[f'dip_moment {molecule_names[0]}'] = np.dot(dipA[stateA], dipA[stateA])
        results[f'dip_moment {molecule_names[1]}'] = np.dot(dipB[stateB], dipB[stateB])
    if intramolecular:
        state = states[0]
        dip = dips[0]
        results[f'dip_moment {molecule_names[0]}'] = np.dot(dip[state], dip[state])
    return results

# get TDDFT outputs as specified in list which_outs for molecules
def getTDDFToutput(output_qm, which_outs, state_ids, molecule_names = ["D", "A"]):

    results = {}
    for i, molecule_name in enumerate(molecule_names):
        for which_out in which_outs:
            for state_id in state_ids:
                results[f"{molecule_name} {which_out} {state_id}"] = output_qm[which_out][i][state_id]

    return results 

# get occupied and virtual orbital energies from DFT
# TODO : only consider n highest occupied and n lowest virtual orbital energies and load n from .params file
def getOrbitalEnergies(output_qm, orbital_types = ["occ", "virt"], molecule_names = ["D", "A"], orbital_energies_type = 'default'):

    # NOTE: we here only consider the 10 highest occupied and 10 lowest virtual orbitals
    num_orbs = 10 

    results = {}
    for i, molecule_name in enumerate(molecule_names):

        orbital_energies = output_qm["orbit_enrgs"][i]                      # all orbital energies
        nocc = output_qm["mol"][i].nelectron // 2                           # occupied orbitals 
        occ_energies = orbital_energies[(nocc-num_orbs):nocc]                     
        virt_energies = orbital_energies[nocc:(nocc+num_orbs)]
        
        for orbital_type in orbital_types:
            if orbital_type == "occ":
                energies = occ_energies
            elif orbital_type == "virt":
                energies = virt_energies
            results[f"{molecule_name} {orbital_type}"] = energies

    return results

# get excited energies from TDDFT
def getExcitedEnergies(output_qm, molecule_names = ["D", "A"]):

    results = {}
    for i, molecule_name in enumerate(molecule_names):
        results[molecule_name] = output_qm['exc'][i]
    return results

# do Mulliken analysis on excited states specified in state_ids based on specified fragments
def getMullikenFragmentAnalysis(output_qm, state_ids, fragments = None, fragment_names = None, molecule_names = ["D", "A"]):

    results = {}
    for i, molecule_name in enumerate(molecule_names):
        for state_id in state_ids:
            atom_pops = output_qm["mull_pops"][i][state_id]
            atom_participation = np.abs(atom_pops)

            for k, fragment_indices in enumerate(fragments[i]):
                frag_name = fragment_names[i][k]
                key = f"{molecule_name} {state_id} {frag_name}"
                results[key] = sum(atom_participation[j] for j in fragment_indices)
    return results


# compute absorption spectrum from oscillator strength and excitation energies along strajectory
# TODO : need to revisit this function
def getAbsorptionSpectrum(osc_strengths, exc_energies, sigma = 0.1, energy_units = 'eV'):

    # (0) convert excitation energies and oscillator strength to appropriate units
    # TODO : confirm that this is correct
    conv = energyConversion(energy_units)
    exc_energies *= conv
    osc_strengths *= conv

    # (1) define energy grid for the plot
    energy_grid = np.linspace(0, max(exc_energies) + 1, 1000)
    spectrum = np.zeros_like(energy_grid)

    # (2) compute spectrum
    for i, exc_energy in enumerate(exc_energies):
        spectrum += osc_strengths[i] * np.exp(-((energy_grid - exc_energy) ** 2) / (2 * sigma ** 2))

    return energy_grid, spectrum



# function that handles energy conversion from Hartree to energy_units
# TODO : do we need this?
def energyConversion(out_unit):
    if out_unit not in ['cm-1', 'E_h', 'eV']:
        raise ValueError("Specify valid energy unit!")
    # load conversion factors
    if out_unit == 'cm-1':
        factor = const.EH2CM
    elif out_unit == 'eV':
        factor = const.EH2EV
    elif out_unit == 'E_h':
        factor = 1.0
    return factor