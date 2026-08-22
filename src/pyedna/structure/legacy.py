"""Legacy structure helpers retained for analysis and trajectory workflows."""

import glob
import os
import subprocess

import numpy as np

from .. import fileproc as fp
from .. import geomtools as geo


# TODO: Remove after trajectory, quanttools, and dye-library construction no
# longer depend on this legacy analysis representation.
class Chromophore():

    def __init__(self, Chromophore_u):
        self.chromophore_u = Chromophore_u                                       # MDAnalysis object
        # parse structure: coordinates, atom names, center of mass
        self.xyz, self.names, self.types, self.com, self.resnames = self.parseStructure()
        self.natoms = len(self.xyz)
        self.dye_name = np.unique(self.resnames)[0]

    # store .pdb source and directory information
    def storeSourcePath(self, path_to_dye):
        # TODO: Retained for scripts/create_dye.py; migrate dye-library creation
        # to DyeDefinition before removing this compatibility method.
        self.path = path_to_dye


    # parse structure
    def parseStructure(self):
        xyz, names, types, com, resnames = geo.getCoords(self.chromophore_u, 'all')
        return xyz, names, types, com, resnames

    # parse attachment info for chromophore
    # TODO : generalize to different linkers
    def parseAttachment(self, change_atom_names = True):
        # TODO: Retained for scripts/create_dye.py; this is not used by
        # StructureBuilder's DNA–dye assembly workflow.
        # (0) load attachment information from file
        try:
            with open(os.path.join(self.path, f"attach_{self.dye_name}.info"), "r") as file:
                self.attach_groups = [line.strip().split() for line in file]
        except FileNotFoundError:
            print("ERROR: Attachment information for this dye does not exist yet!")

        self.attach_num = len(self.attach_groups)                                        # len(attach_points) = number of attachments to DNA
        # (1) extract names of the Os, Ps that won't be deleted
        self.O_term, self.O_conn, self.P= [], [], []
        for attach_group in self.attach_groups:
            self.O_conn.append(attach_group[0])
            self.P.append(attach_group[1])
            self.O_term.append([attach_group[2], attach_group[3]])

        # (1) atoms that need to be deleted:
        self.delete_atoms = []
        # (1.1) need to remove one OH group and one H (last three elements) from both OPO3 groups
        for i in range(self.attach_num):
            self.delete_atoms += self.attach_groups[i][-3:]
        # (1.2) need to delete all O_term and P from 5' end
        self.delete_atoms.append(self.P[0])
        self.delete_atoms += self.O_term[0]
        # (1.3) build delete string for select_atoms method
        self.del_string = str()
        for atom in self.delete_atoms:
            self.del_string += f" and not name {atom}"


        # (3) indices of atoms where Chromophore will be attached to DNA
        self.attach_idx = [np.where(self.names == P)[0][0] for P in self.P]
        self.attach_pos = self.xyz[self.attach_idx]

        # (4) rename atoms in phosphate group that are not getting deleted
        # in order to simulate them with the DNA forcefield
        DNA_O_conn = ["O3'", "O5'"]
        DNA_O_term = ['OP1', 'OP2']
        DNA_P = "P"
        self.rename_atoms = dict()                  # dictionary for renaming atoms
        self.rename_types = dict()                  # dictionary for setting OL15 atom types for these atoms
        for i in range(self.attach_num):
            if self.P[i] not in self.delete_atoms:
                self.rename_atoms[self.P[i]] = DNA_P
                self.rename_types[self.P[i]] = 'P'
            if self.O_conn[i] not in self.delete_atoms:
                self.rename_atoms[self.O_conn[i]] = DNA_O_conn[i]
                self.rename_types[self.O_conn[i]] = 'OS'
            for j in range(2):
                if self.O_term[i][j] not in self.delete_atoms:
                    self.rename_atoms[self.O_term[i][j]] = DNA_O_term[j]
                    self.rename_types[self.O_term[i][j]] = 'O2'

        # use dict to translate old atom names
        self.names = np.array([self.rename_atoms.get(atom, atom) for atom in self.names])
        # update self.names
        if change_atom_names:
            self.chromophore_u.atoms.names = self.names
            self.names = self.chromophore_u.atoms.names


    # write attachment information
    # TODO : generalize this to different linkers
    @staticmethod
    def writeAttachmentInfo(chromophore_u, dye_name, linker_atoms = ['P1', 'P2'], linker_group = 'phosphate'):
        # TODO: Retained for scripts/create_dye.py until dye-library creation
        # writes the current .attach format directly.

        attachment_info = []

        if linker_group != 'phosphate':
            raise NotImplementedError("Only phosphate group as linker currently implemeted!")

        if linker_group == 'phosphate':

            # we store the atom names for the phosphate attachment group as follows for each linker_atom in linker_atoms
            # attachment = [O_bridge, P, O_term, OH_term, H, OH_term, H]
            # where ...-C-O_bridge-P(=O_term)(-OH_term-H)(-OH_term-H) is the -OPO3H2 linker group

            # loop through linkers
            for linker_atom in linker_atoms:

                # get (oxygen) neighbors of linkers (phosphates)
                nearest_neighbors = Chromophore.getNeighborAtoms(chromophore_u, linker_atom)

                # among (oxygen) neighbors, identify which ones are terminal (=0), terminal (-OH) or bridging (-O-)
                OH_terminal, O_terminal, O_bridge = [], [], []
                for neighbor in nearest_neighbors.atoms.names:

                    # get next nearest neighbors
                    next_nearest_neighbors = Chromophore.getNeighborAtoms(chromophore_u, neighbor).select_atoms(f"not name {linker_atom}")

                    # identify type of neighbor and store information
                    if len(next_nearest_neighbors) == 0:
                        O_terminal.append(neighbor)
                    elif len(next_nearest_neighbors) == 1:
                        # determine whether the oxygen is terminal or bridging
                        if 'C' in next_nearest_neighbors.types:
                            O_bridge.append(neighbor)
                        else:
                            OH_terminal.append(neighbor)
                            OH_terminal.append(next_nearest_neighbors.atoms.names[0])

                    else:
                        raise Warning("Incorrect number of bond-neighbors identified in linker! Check PDB. \
                                    Might want to reduce cutoff to identify only nect neighbors")

                # store attachment info as follows
                attachment = O_bridge + [linker_atom] + O_terminal + OH_terminal
                attachment_info.append(attachment)

            # write attachment info to file
            with open(f"attach_{dye_name}.info", "w") as file:
                for attachment in attachment_info:
                    file.write(" ".join(attachment) + "\n")



    @staticmethod
    def getNeighborAtoms(chromophore_u, target_atom_name, search_radius = 2.0):

        from MDAnalysis.lib import NeighborSearch

        # Select all atoms for searching
        all_atoms = chromophore_u.select_atoms("all")

        # Initialize NeighborSearch with all atoms
        ns = NeighborSearch.AtomNeighborSearch(all_atoms)

        # Search around target atom
        target_atom = chromophore_u.select_atoms(f"name {target_atom_name}")[0]
        neighbors = ns.search(target_atom, search_radius).select_atoms(f"not name {target_atom_name}")

        return neighbors



    # create force field with antechamber/parmchk2
    def createFF(self, charge = 0, ff = 'gaff', file_verbosity = 0):
        # TODO: Retained for scripts/create_dye.py; AmberSetup consumes
        # already prepared GAFF files from DYE_DIR.
        # (1) write updated pdb file after deletion of groups for attachment
        fp.deleteAtomsPDB(f'{self.dye_name}' + '.pdb', f'{self.dye_name}' + '_del.pdb', self.delete_atoms)
        # (2) use antechamber
        command = f"antechamber -i '{self.dye_name}_del.pdb' -fi pdb -o {self.dye_name}.mol2 -fo mol2 -c bcc -s 2 -nc {charge} -m 1 -at {ff}"
        run_antechamber = subprocess.Popen(command, cwd = self.path, shell = True)
        run_antechamber.wait()
        # (3) run parmchk2
        command = f"parmchk2 -i {self.dye_name}.mol2 -f mol2 -o {self.dye_name}.frcmod -s gaff"
        run_parmchk2 = subprocess.Popen(command, cwd = self.path, shell = True)
        run_parmchk2.wait()
        # (4) delete axuiliary files in ff directory
        if file_verbosity == 0:
            for prefix in ["ANTECHAMBER", "sqm", "ATOMTYPE"]:
                pattern = os.path.join(f'{self.path}', f"{prefix}*")  # Match files with the prefix
                files_to_delete = glob.glob(pattern)                    # Get list of matching files

                for file in files_to_delete:
                    try:
                        os.remove(file)
                        print(f"Deleted: {file}")
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
        else:
            pass




# clean PDB files
# TODO : do we actually need this ?? If yes, put this somewhere else
def cleanPDB(inPath, outPath, res_code='DYE', mol_title='Dye molecule', printCONNECT = False):
    from .. import fileproc
    lpdb = fileproc.PDB_DF()
    lpdb.read_file(inPath)

    # Clean existing names (if already numbered)
    atom_names = lpdb.data['HETATM']['atom_name']
    clean_atoms = fp.clean_numbers(atom_names)

    # Sort atoms in the dataframe
    hetatm = lpdb.data['HETATM']
    hetatm['atom_name'] = clean_atoms
    hetatm_sorted = hetatm.sort_values(by=['atom_name', 'atom_id'],ascending = [True, True])
    sorted_atoms = hetatm_sorted['atom_name']

    # print(hetatm.head())

    # Replace atom names by numbered names
    asymbol, acounts = np.unique(sorted_atoms, return_counts = True)
    fixed_names = fp.make_names(asymbol,acounts)
    hetatm_sorted['atom_name'] = fixed_names

    # Replace res names
    # res_names = fp.set_res(res_code, hetatm_sorted)
    # hetatm_sorted['res_name'] = res_names

    # Save pdb file
    lpdb.data['MOLECULE'] = mol_title
    lpdb.data['HETATM'] = hetatm_sorted

    lpdb.write_file(outPath, resname = res_code, print_connect = printCONNECT)
