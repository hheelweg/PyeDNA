import numpy as np
import MDAnalysis as mda
import subprocess
import os
from collections import defaultdict

# from current package
from . import fileproc as fp
from . import geomtools as geo


# class for attaching structure of DNA structure and chromophore
class CompositeStructure():

    def __init__(self, dnaPDB):#, path_to_structures):
        # create instance of DNA class
        self.dna = DNA(mda.Universe(dnaPDB, format = "PDB"))            # create instance of DNA class
        self.dna_nresidues = self.dna.nresidues                         # number of DNA residues (this should stay constant)  
        # self.path = path_to_structures                                  # path to structure directory (base directory)               
        # store attachments to DNA and bond information
        self.bonds = []                                                 # store bond information for tleap
        self.chromophore_list = []                                      # list of attached Chromophore objects
        self.attachment = defaultdict(list)
        self.charge = - (self.dna_nresidues - 2)                        # charge of DNA composite


    # parse attachment info, and overwrite DNA universe with DNA + chromophore
    def prepareAttachment(self, path_to_dye, dye_name, attach_info_dna, orientation = -1):

        # (0) load chromophore
        self.chromophore = Chromophore(mda.Universe(os.path.join(path_to_dye, f'{dye_name}.pdb'), format = "PDB"))      # create instance of Chromophore class
        self.chromophore.storeSourcePath(path_to_dye)                                                                   # store path to chromophore file     

        # (1) parse attachment info for chromophore and dna; write bond information
        self.chromophore.parseAttachment()
        self.dna.parseAttachment(attach_info_dna)
        self.writeBondInfo()

        # (2) align chromophore with dna based on attachment info
        self.chromophore.alignChromophore(self.dna.target_pos, self.dna.com, orientation)

        # (3) delete atoms in chromophore and dna objects adn adjust charge of composite
        self.chromophore.delete(self.chromophore.del_string)
        self.dna.delete(self.dna.del_string)
        self.charge += 1

        # TODO : this does not seem to do anything!
        # TODO : in case of calling this method multiple times for multiple attachments, this does not work (debug!)
        # (4) clean DNA (remove incomplete nucleotide)
        #self.dna.DNA_u = self.cleanDNA()

        # (5) merge coordinates together into DNA universe and re-load self.dna
        self.mergeCoordinates()

        # (6) update attachment infos to DNA and store attachment info
        self.chromophore_list.append(self.chromophore)
        self.attachment[dye_name].append(self.chromophore.resid)
    
    
    # get nt distance
    def getNTdistance(self): 
        if 'CY5' not in self.attachment or 'CY3' not in self.attachment:
            raise Warning("Need to attach CY5 and CY3 before computing nt!")
        else:
            nt = min([abs(item - self.attachment['CY5'][0]) for item in self.attachment['CY3']]) - 1
        return nt
    

    # write bond information
    def writeBondInfo(self):
        bond_atoms = [["O3'", 'P'], ["P", "O3'"]]
        res_idxs = [self.dna.res_idx - 1, self.dna.res_idx + 1]
        for i, bond_atom in enumerate(bond_atoms):
            bond = []
            # store as [res_id,atom_name] for each atom in bond
            bond.append([self.dna.res_ids[res_idxs[i]], bond_atom[0]])
            bond.append([self.dna.res_ids[self.dna.res_idx], bond_atom[1]])
            bond = sorted(bond, key=lambda x: x[0])
            if bond not in self.bonds:                                          # only write unique new bonds
                self.bonds.append(bond)


    # write AMBER input after all the attachments have been done
    def writeAMBERinput(self):
        # (0) get nt distance between donor and acceptor
        nt = self.getNTdistance()
        # (1) write .pdb file of dna (unclean) in new directory
        dir_name = 'dna' + '_'.join([chromophore.dye_name for chromophore in self.chromophore_list]) + f'_{nt}nt'
        subprocess.run("mkdir -p " + dir_name, cwd = self.path, shell = True)
        file_name = f'dna_{nt}nt'
        self.dna.DNA_u.atoms.write(os.path.join(self.path, dir_name, file_name + "_unclean"))

        # (2) clean .pdb file with pdb4amber (clean)
        subprocess.run(f"pdb4amber -i {file_name}_unclean.pdb -o {file_name}.pdb", cwd = os.path.join(self.path, dir_name), shell = True)
        # delete pdb4amber cache files
        for ftrash in ['sslink', 'renum.txt', 'nonprot.pdb']:
            myfile = os.path.join(self.path, dir_name, f"{file_name}_{ftrash}")
            if os.path.isfile(myfile):
                os.remove(myfile)
            else:
                print("Error: %s file not found" % myfile)
        
        # (3) write leap file to make bond and run it (this generates the AMBER input)
        suff_leap = '_tleap.in'
        fp.writeLeap(os.path.join(self.path, dir_name), file_name, file_name + suff_leap,
                    self.bonds, self.chromophore_list, self.charge)
        subprocess.run(f"tleap -f {os.path.join(self.path, dir_name, file_name + suff_leap)}", shell = True)

    
    # merge coordinates together into DNA_u MDAnalysis object
    def mergeCoordinates(self, ordering = True):
        # (1) clean chromophore structure so that we only have one segment (might want to check this)
        # also adust the ID
        self.chromophore.resid = self.dna.target_res_ids[1]
        self.chromophore.cleanChromophore(self.chromophore.resid)
        # (2) insert the chromophore into the deleted gap
        resids_before = " or ".join(f"resid {i}" for i in range(1, self.dna.target_res_ids[1]))
        resids_after = " or ".join(f"resid {i}" for i in range(self.dna.target_res_ids[1] + 1, self.dna_nresidues + 1))
        residues_before = self.dna.DNA_u.select_atoms(resids_before)
        residues_after = self.dna.DNA_u.select_atoms(resids_after)
        # overwrite DNA universe and re-initialize self.dna
        self.dna.DNA_u = mda.Merge(residues_before, self.chromophore.chromophore_u.atoms, residues_after)
        self.dna = DNA(self.dna.DNA_u)
    
    # TODO : can remove this, this only for DMREF picture
    def writePDB(self, out_path):
        self.dna.DNA_u.atoms.write(out_path + 'out.pdb')


    # TODO : do we need this?
    def cleanDNA(self, st_atoms = 25):
        self.del_ids = [int(self.dna.target_res_ids[1])]

        # 1) Select nucleotides and non DNA groups
        dna = self.dna.DNA_u.select_atoms('nucleic')
        non_dna = self.dna.DNA_u.select_atoms('not nucleic')
        # 2) If the res is a nucleotide (give this at input), check the total number of atoms
        total_valid = 0
        invalid_res = []

        for nuc in dna.residues:
        # if the atoms are complete, merge. "Complete" includes having both edge oxygens.
            if len(nuc.atoms.select_atoms("name O3' or name O5'")) > 1 and len(nuc.atoms) >= st_atoms:
                if total_valid==0:
                    non_dna = nuc.atoms
                else:
                    non_dna = mda.Merge(non_dna.atoms, nuc.atoms)
                total_valid += 1
            else:
                invalid_res.append(nuc.resid)
        #print(f'A total of {len(dna.residues)-total_valid} incomplete residues were deleted')
        updated_unit = non_dna.select_atoms('all')


        def find_edge_residues(atom_sel):
            reslist = atom_sel.select_atoms("nucleic").residues
            atom_list = list(atom_sel.atoms.ids)
            edge_atoms = []
            for r, res in enumerate(reslist):
                if r>0:
                    P = res.atoms.select_atoms("name P").ids
                    if P:
                        P = np.argwhere(atom_list==P[0])[0][0]
                        #print(list(atom_sel.atoms)[P], list(atom_sel.atoms)[P-1])
                        bond = atom_sel.atoms[[P, P-1]].bond    
                        if bond.value()>7:
                            edge_atoms.append(res.resid)      
            return edge_atoms


        # cap edges
        mol = updated_unit
        res_list = [r.resid for r in mol.residues]
        first = min(res_list)
        dye1 = self.del_ids[0]
        dye_edges = [dye1-1, dye1+1]

        edges = [first] + find_edge_residues(mol)
        edges = [x for x in edges if x not in dye_edges]

        for e in edges:
            mol = mol.select_atoms(f'all and not (resid {e} and (name P or name OP1 or name OP2))')
        return mol




# class for DNA 
class DNA():

    def __init__(self, DNA_u, chain = None):
        self.DNA_u = DNA_u                      # MDAnalysis object
        self.chain_sel = None
        # select chain
        self.selectChain(chain)
        # get properties
        self.getProperties()
        # get bond information
        self.getBondInformation()


    # select chain
    def selectChain(self, chain = None):
        if chain: 
            self.chain_sel = self.DNA_u.select_atoms("segid " + chain)
        else: 
            self.chain_sel = self.DNA_u.select_atoms("all")
    
    # get properties associated with chain selection
    def getProperties(self):
        if self.chain_sel is None:
            raise ValueError("Need to specify DNA chain of interested!")
        
        self.atoms = self.chain_sel.atoms               # AtomGroup
        self.residues = self.chain_sel.residues         # ResidueGroup
        self.positions = self.atoms.positions           # atom positions
        self.names = self.atoms.names                   # atom names
        self.com = self.atoms.center_of_geometry()      # center of geometry TODO : find out why center_of_mass() does not seem to work
        self.natoms = len(self.names)                   # number of atoms
        self.nresidues = len(self.residues)             # number of residues

        # list of non-H atoms
        # NOTE : do we need this?
        self.atIds = np.arange(self.natoms)
        nonHs = np.invert(np.char.startswith(self.names.astype(str), 'H'))
        self.heavyIds = self.atIds[nonHs]               # IDs of non-H atoms
    
    # get information of all P-atoms in all nucleotides of self.chain_sel
    def getBondInformation(self):
        if self.chain_sel is None:
            raise ValueError("Need to specify DNA chain of interested!")
        # get bonding information
        residue_list = self.chain_sel.residues
        self.O3_pos, self.P_pos = [], []            # position of O3' atom and P in each residue
        self.res_ids,  self.res_names = [], []      # ids of residues, names of residues
        for residue in residue_list:
            # (1) location of O3' in each residue
            self.O3_pos.append((residue.atoms.select_atoms("name O3'").positions[0]).tolist())
            # (2) location of P-groups 
            # NOTE : the 5' end of the DNA strand starts with a nucleotide that does NOT have a P (use [9999, 9999, 9999] then)
            self.P_pos.append(residue.atoms.select_atoms("name P").positions[0].tolist()) if len(residue.atoms.select_atoms("name P").positions) > 0 else self.P_pos.append(np.array([9999,9999,9999]).tolist())

            self.res_ids.append(residue.resid)
            self.res_names.append(residue.resname)
        # label residues customized with name + ID
        self.res_labels = [self.res_names[i] + str(self.res_ids[i]) for i in range(len(self.res_names))]

    # parse attachment information:
    def parseAttachment(self, res_id):
        # residue with res_id is removed 
        self.res_idx = self.res_ids.index(res_id)
        # get target positions, labels where we need to perform attachment [P_5end, P_3end]
        self.target_pos = [self.P_pos[self.res_idx + 1], self.P_pos[self.res_idx]]
        self.target_labels = [self.res_labels[self.res_idx + 1], self.res_labels[self.res_idx]]
        self.target_res_ids = [self.res_ids[self.res_idx + 1], self.res_ids[self.res_idx]]
        self.target_res_names = [self.res_names[self.res_idx + 1], self.res_names[self.res_idx]]
        # TODO : do we need this?
        self.target_bonds = [self.O3_pos[self.res_idx + 1], self.O3_pos[self.res_idx]]
        # delete string for select_atoms method
        self.del_string = str()
        self.del_string += f" and not (resname {self.target_res_names[1]} and resid {self.target_res_ids[1]})"
        

    # get DNA box
    def getDNAbox(self, center, box_size = 20):
        '''
        creates box-shaped MDAnalysis selection around center coordinate
        '''
        box = self.DNA_u.select_atoms(f'point {center[0]} {center[1]} {center[2]} {box_size}')
        box = DNA(box)
        return box

    # delete atoms
    def delete(self, delete_string):
        # delete atoms
        DNA_u_new = self.DNA_u.select_atoms('all' + delete_string)
        # re-edit
        # overwrite 
        self.__init__(DNA_u_new)




# class for chromophore
class Chromophore():

    def __init__(self, Chromophore_u):
        self.chromophore_u = Chromophore_u                                       # MDAnalysis object
        # parse structure: coordinates, atom names, center of mass
        self.xyz, self.names, self.types, self.com, self.resnames = self.parseStructure() 
        self.natoms = len(self.xyz)
        self.dye_name = np.unique(self.resnames)[0]
    
    # store .pdb source and directory information
    def storeSourcePath(self, path_to_dye):
        self.path = path_to_dye
        print('path test ', self.path)
        print('another test ', os.path.join(self.path, f"attach_{self.dye_name}.info"))


    # parse structure
    def parseStructure(self):
        xyz, names, types, com, resnames = geo.getCoords(self.chromophore_u, 'all')
        return xyz, names, types, com, resnames

    # parse attachment info for chromophore
    def parseAttachment(self, change_atom_names = True):
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


        # # (3) indices of atoms where Chromophore will be attached to DNA
        self.attach_idx = [np.where(self.names == P)[0][0] for P in self.P]
        self.attach_pos = self.xyz[self.attach_idx]

        # # (4) rename atoms in phosphate group that are not getting deleted
        # # in order to simulate them with the DNA forcefield
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

    # create force field
    def createFF(self, charge = 0, ff = 'gaff'):
        # (1) write updated pdb file after deletion of groups for attachment
        fp.deleteAtomsPDB(self.path + f'{self.dye_name}' + '.pdb', self.path + f'{self.dye_name}' + '_del.pdb', self.delete_atoms)
        # (2) use antechamber 
        makedir_ff = subprocess.run(f"mkdir -p {self.path}/ff_new", shell = True)       # make forecielf directory
        command = f"antechamber -i '../{self.dye_name}_del.pdb' -fi pdb -o {self.dye_name}_del.mol2 -fo mol2 -c bcc -s 2 -nc {charge} -m 1 -at {ff}"
        run_antechamber = subprocess.Popen(command, cwd = f'{self.path}/ff_new', shell = True)
        run_antechamber.wait()
        # (3) run parmchk2
        command = f"parmchk2 -i {self.dye_name}_del.mol2 -f mol2 -o {self.dye_name}_del.frcmod -s gaff"
        run_parmchk2 = subprocess.Popen(command, cwd = f'{self.path}/ff_new', shell = True)
        run_parmchk2.wait()


    # align the Chromophore at its attachments sites with the attachment sites of the DNA
    def alignChromophore(self, attach_loc, com_target, orientation):
        # update coordinates based on attachment position
        self.xyz_new, self.com_new = geo.alignToTarget(self.xyz, self.attach_pos, attach_loc, self.com, com_target, orientation)
        # change positions:
        self.chromophore_u.atoms.positions = self.xyz_new
        self.xyz = self.chromophore_u.atoms.positions

    # clean chromophore Universe and add arbitrary residue ID res_id
    def cleanChromophore(self, res_id):
        # initialiye
        self.chromophore_u = mda.Universe.empty(n_atoms = self.natoms,
                                                    n_residues = 1,
                                                    n_segments = 1,
                                                    atom_resindex = [0] * self.natoms,
                                                    residue_segindex = [0],
                                                    trajectory = True)
        # fill in 
        self.chromophore_u.add_TopologyAttr('name', self.names)
        self.chromophore_u.add_TopologyAttr('type', self.types)
        self.chromophore_u.add_TopologyAttr('resname', [self.dye_name])                 
        self.chromophore_u.add_TopologyAttr('resid', [res_id])                # segment ID
        self.chromophore_u.add_TopologyAttr('segid', ['SEG'])                 # single segment
        self.chromophore_u.add_TopologyAttr('id', list(range(self.natoms)))
        self.chromophore_u.add_TopologyAttr('record_types', ['HETATM'] * self.natoms)
        self.chromophore_u.atoms.positions = self.xyz

    # delete atoms
    def delete(self, delete_string):
        chromophore_u_new = self.chromophore_u.select_atoms('all' + delete_string)
        self.__init__(chromophore_u_new)

        
        







# clean PDB files
# TODO : do we actually need this ??
def cleanPDB(inPath, outPath, res_code='DYE', mol_title='Dye molecule', printCONNECT = False):
    from . import fileproc
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