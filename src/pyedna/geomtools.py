import numpy as np
import MDAnalysis as mda

# get coordinates for MDAanalysis object
def getCoords(mda_u, selection='all'):
    # selection : str of atom names (make more general if desired)
    if selection == 'all':
        selection_str = selection
    else:
        selection_str = makeAtomSelection(selection)
    # select subset of interest
    u_sel = mda_u.select_atoms(selection_str)
    # coordinates, atom names, centre of mass, residue names
    # TODO : we changed from center_of_mass -> center_of_geometry (MDAnalysis version on cluster seems to require this)
    xyz, names, types, com, resnames = np.array(u_sel.positions), u_sel.atoms.names, u_sel.atoms.types, u_sel.atoms.center_of_geometry(), u_sel.resnames
    return xyz, names, types, com, resnames

# transform list of ATOM names into readible string for MDAanalysis.select_atoms
def makeAtomSelection(atom_list):
    selection_str = 'name'
    for atom in atom_list:
        selection_str = " ".join([selection_str, atom])
    return selection_str

# shift center of geometry of MDAnalysis Universe object to (0,0,0) and align specified molecular axis of molecule with (0,0,1) axis
def shiftAndAlign(mda_u, axis_atom_names):

    from scipy.spatial.transform import Rotation as R

    # Select all atoms
    atoms = mda_u.atoms

    # Compute center of geometry (COG)
    cog = atoms.center_of_geometry()
    
    # Shift the molecule to set COG at (0,0,0)
    atoms.positions -= cog

    # Get the positions of the two defining atoms
    atom1 = mda_u.select_atoms(f"name {axis_atom_names[0]}").positions[0]
    atom2 = mda_u.select_atoms(f"name {axis_atom_names[1]}").positions[0]

    # Compute the axis vector
    axis_vector = atom2 - atom1
    axis_vector /= np.linalg.norm(axis_vector)  # Normalize

    # Define the target vector (z-axis)
    target_vector = np.array([0, 0, 1])

    # Compute the rotation required to align axis_vector with target_vector
    rotation, _ = R.align_vectors([target_vector], [axis_vector])

    # Apply rotation to all atom positions
    atoms.positions = rotation.apply(atoms.positions)

    return mda_u


def enforceSymmetry(mda_u, axis_atom_names, support_name = 'N1', tol = 0.1):
    
    
    # Extract atom positions
    atoms = mda_u.atoms
    coords = atoms.positions.copy()

    # Get positions of reference atoms
    pos1 = mda_u.select_atoms(f"name {axis_atom_names[0]}").positions[0]  # Defines mirror plane
    pos2 = mda_u.select_atoms(f"name {axis_atom_names[1]}").positions[0]  # Defines mirror plane
    pos3 = mda_u.select_atoms(f"name {support_name}").positions[0]  # Used to find normal

    # Compute the axis vector along atom1 → atom2
    axis_vec = pos2 - pos1
    axis_vec /= np.linalg.norm(axis_vec)  # Normalize

    # Compute the normal to the mirror plane (orthogonal part of (atom3 - atom2) onto axis_vec)
    vec3_2 = pos3 - pos2  # Vector from atom2 to atom3
    normal = vec3_2 - np.dot(vec3_2, axis_vec) * axis_vec  # Remove parallel component to axis_vec
    normal /= np.linalg.norm(normal)  # Normalize

    # Compute signed distances of all atoms from the mirror plane
    distances = np.dot(coords - pos1, normal)  # Signed distances to mirror plane
    mask_positive = distances > tol  # Atoms on one side (considering tolerance)
    mask_negative = distances < -tol  # Atoms on the other side (considering tolerance)

    # Ensure Atom1 and Atom2 are always included in the final structure
    atom1_idx = mda_u.select_atoms(f"name {axis_atom_names[0]}").indices[0]
    atom2_idx = mda_u.select_atoms(f"name {axis_atom_names[1]}").indices[0]
    
    # Define atoms exactly inside the mirror plane (within tolerance)
    mask_mirror_plane = np.abs(distances) <= tol  
    mask_mirror_plane[atom1_idx] = True  # Force inclusion of Atom1
    mask_mirror_plane[atom2_idx] = True  # Force inclusion of Atom2

    # Select atoms to keep (those in the mirror plane + those on one side)
    atoms_to_keep = atoms[mask_negative | mask_mirror_plane]  # Keep one side + mirror-plane atoms
    atoms_to_mirror = atoms[mask_negative]  # Mirror only these atoms

    # Compute mirrored positions
    mirrored_coords = atoms_to_mirror.positions - 2 * np.outer(np.dot(atoms_to_mirror.positions - pos1, normal), normal)

    # Create a new universe for mirrored atoms
    mirrored_universe = mda.Merge(atoms_to_mirror)
    mirrored_universe.atoms.positions = mirrored_coords  # Assign new positions

    # Merge the kept atoms + mirrored atoms into one final universe
    new_universe = mda.Merge(atoms_to_keep, mirrored_universe.atoms)

    return new_universe


def getDistance(mda_u, distance_type):
    # extract infromation from distance_type list
    if distance_type[0] == 'residues':
        point_1 = mda_u.select_atoms(f"resid {distance_type[1]}").center_of_geometry()
        point_2 = mda_u.select_atoms(f"resid {distance_type[2]}").center_of_geometry()
    else:
        raise NotImplementedError("Currently only distances between centers of geometry for different residues implemented!")
    
    # compute distance
    distance = np.linalg.norm(point_1-point_2)
    return distance


def getAxisAngle(mda_u, axis_angle_type):

    # (1) compute axis vectors
    axis_vectors = []
    for i in range(2):
        # atom 1
        atom1_info = axis_angle_type[i][0]
        atom1_pos = mda_u.select_atoms(f"resid {atom1_info[0]} and name {atom1_info[1]}")[0].position
        # atom 2
        atom2_info = axis_angle_type[i][1]
        atom2_pos = mda_u.select_atoms(f"resid {atom2_info[0]} and name {atom2_info[1]}")[0].position
        # axis vector
        axis_vectors.append(atom1_pos - atom2_pos)

    # (2) compute angle between axes
    axis1, axis2 = axis_vectors[0], axis_vectors[1]
    cos_theta = np.dot(axis1, axis2) / (np.linalg.norm(axis1) * np.linalg.norm(axis2))

    # if we actually care about the angle (in radians)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0)) 

    return cos_theta

