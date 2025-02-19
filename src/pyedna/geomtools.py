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

# align molecule (mobile) with target (stationary)
# NOTE : currently implemented for len(target) = len(current) = 2 (2 points of attachment)
def alignToTarget(mol_coords, current, target, com, com_target, orientation = -1):

    from scipy.spatial.transform import Rotation as R

    # (1) translate molecule and target so that midpoints are at (0,0,0) (in order to rotate)
    mid_target = np.mean(target, axis = 0)
    mid_current = np.mean(current, axis = 0)
    mol_coords_sh = mol_coords - mid_current
    com_sh = com - mid_current
    current_sh = current - mid_current
    target_sh = target - mid_target
    com_target_sh = com_target - mid_target
    
    # (2) rotation 1 : make target_sh[i] and current_sh[i] align (i = 0 or i = 1 does not matter)
    # (2a) get (normalized) rotation axis
    axis_rot = np.cross(current_sh[0], target_sh[0]) / np.linalg.norm(np.cross(current_sh[0], target_sh[0]))
    # (2b) get rotation angle: α = arccos[(a · b) / (|a| * |b|)]
    angle_rot = np.arccos(np.dot(current_sh[0], target_sh[0]) / (np.linalg.norm(current_sh[0]) * np.linalg.norm(target_sh[0])))
    # (2c) get rotation vector: angle_rot * axis_rot
    vec_rot = axis_rot * angle_rot
    # (2d) perform rotation of molecule
    Rot = R.from_rotvec(vec_rot)
    mol_coords_sh = Rot.apply(mol_coords_sh)
    current_sh = Rot.apply(current_sh)
    com_sh = Rot.apply(com_sh)
    
    # (3) rotation 2 : rotate molecule around its own axis so that com projections onto the molecule axis are antiparallel
    # (3a) get (normalized) molecule axis
    axis_mol = target_sh[0] / np.linalg.norm(target_sh[0])
    # (3b) get orthogonal projections of com vectors onto axis_rot
    com_sh_orth = np.dot(np.identity(3) - np.outer(axis_mol, axis_mol), com_sh)
    com_target_sh_orth = np.dot(np.identity(3) - np.outer(axis_mol, axis_mol), com_target_sh) 
    axis_rot = np.cross(com_sh_orth, com_target_sh_orth) / np.linalg.norm(np.cross(com_sh_orth, com_target_sh_orth))
    # note: axis_rot should be parallel to axis_mol (in order to not mess up the signs for the rotation, we compute axis_rot manually though)
    # (3c) get roation angle (want: antiparallel alignment)
    angle_rot = np.pi - np.arccos(np.dot(com_sh_orth, com_target_sh_orth)/ (np.linalg.norm(com_sh_orth) * np.linalg.norm(com_target_sh_orth)))
    # TODO : for circular dna, make this be axis_rot * angle_rot, for regular project: - axis_rot * angle_rot
    vec_rot = orientation * axis_rot * angle_rot
    # (3d) perform rotation of molecule
    Rot = R.from_rotvec(vec_rot)
    mol_coords_sh = Rot.apply(mol_coords_sh)
    current_sh = Rot.apply(current_sh)
    com_sh = Rot.apply(com_sh)
    # check antiparallel alignement of orthogonal projection of com vector onto molecule axis
    # print("check antiparallel", np.dot(Rot.apply(com_sh_orth), com_target_sh_orth) / (np.linalg.norm(Rot.apply(com_sh_orth)) * np.linalg.norm(com_target_sh_orth)))

    # (4) translation: translate molecule to midpoint of target
    mol_coords_sh += mid_target
    com_sh += mid_target
    return mol_coords_sh, com_sh
    