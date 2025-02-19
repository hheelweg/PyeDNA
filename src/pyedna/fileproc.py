import numpy as np
import pandas as pd
import os
from . import trajectory as traj

class PDB_DF():

    def __init__(self) -> None:

        # Initializing pdb dictionary
        self.data = {}
        self.data['MOLECULE'] = ""
        self.data['AUTHOR'] = ""
        self.data['ATOM'] = pd.DataFrame()
        self.data['HETATM'] = pd.DataFrame()
        self.data['CONNECT'] = []
        self.data['MASTER'] = []

        # Defining keys
        self.atom_keys = ['type', 'atom_id', 'atom_name', 'res_name', 'res_id', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'atom_type']

        # Defining formatters
        self.atom_formatter = [lambda x: f'{x}  ', lambda x: f'{x} ', lambda x: f'{x:<3}', lambda x: f'{x}    ',
                               lambda x: f'{x}   ', lambda x: f'{x:.3f} ', lambda x: f'{x:.3f} ', lambda x: f'{x:.3f} ',
                               lambda x: f'{x:.2f} ', lambda x: f'{x:.2f} ', lambda x: f'      {x}']

    def read_file(self, pdb_file, names=None):
    
        name_str = ""
        author_str = ""
        a_noindex = []
        h_noindex = []
        c_line = ""
        m_line = []
    
        with open(pdb_file) as f:
            lines = f.readlines()
            total_lines = len(lines)
            # Save line indexing of groups
            #print(lines)
            for num, line in enumerate(lines):
                if 'COMPND' in line:
                   name_str = line
                if 'AUTHOR' in line:
                    author_str += line[:-2]
                if 'ATOM' not in line:
                    a_noindex.append(num)
                if 'HETATM' not in line:
                    h_noindex.append(num)
                if 'CONECT' in line:
                    c_line += line
                if 'MASTER' in line:
                    m_line = line.split()[1:]
        self.data['MOLECULE'] = "".join(name_str.split()[1:])
        self.data['AUTHOR'] = author_str
        self.data['CONNECT'] = c_line
        self.data['MASTER'] = m_line
        
        keys = names if names else self.atom_keys
        if total_lines-len(a_noindex)>0:
            self.data['ATOM'] = pd.read_csv(pdb_file, sep=' ', header=None, names=keys, index_col=False, 
                                                skiprows=a_noindex, skipinitialspace=True, engine='python') 
        if total_lines-len(h_noindex)>0:
            self.data['HETATM'] = pd.read_csv(pdb_file, sep=' ', header=None, names=keys, index_col=False,
                                                skiprows=h_noindex, skipinitialspace=True, engine='python') 
     

    def write_file(self, path, resname=None, print_connect=True, reset_ids=False):

        
        atom_data = self.data['ATOM']
        hetatom_data = self.data['HETATM']
        
        # Fix atom and bond numbers
        atom_number = len(atom_data)
        hetatom_number = len(hetatom_data)
        mol_name = self.data['MOLECULE']
        connect_data = self.data['CONNECT']
        if len(self.data['AUTHOR']) >0:
            author_line = self.data['AUTHOR'] #+ "AND POST PROCESSED WITH DYE-SCREEN"
        else:
            author_line = "AUTHOR    GENERATED WITH DYE-SCREEN"
        m_data = self.data['MASTER']

        # Print header
        with open(path, "w") as f:
            f.write(f"COMPND    {mol_name}\n")
            f.write(author_line+ "\n")

            # Print atom section
            if not atom_data.empty:
                if reset_ids:
                    atom_data = reset_atomids(atom_data)
                if resname is not None:
                    atom_data['res_name'] = np.array([resname]*len(atom_data))
                atom_data = atom_data.sort_values(by=['atom_id'])
                atom_str = atom_data.to_string(header=None, col_space=[6,1,1,3,1,5,5,5,4,4,1], index = False, 
                                               justify='right', formatters = self.atom_formatter)
                f.write(atom_str+"\n")
            # Print hetatm section               
            if not hetatom_data.empty:
                if reset_ids:
                    hetatom_data = reset_atomids(hetatom_data)
                if resname is not None:
                    hetatom_data['res_name'] = np.array([resname]*len(hetatom_data))
                    #print(hetatom_data.head())
                hetatom_data = hetatom_data.sort_values(by=['atom_id'])
                hetatm_str = hetatom_data.to_string(header=None, col_space=[6,1,1,3,1,5,5,5,4,4,1], index = False,
                                                    justify='right', formatters = self.atom_formatter)
                f.write(hetatm_str+"\n")
            #Print bonds section
            if print_connect:
                f.write(connect_data)
            f.write("END")



# delete specific line with atom names from .pdb file
def deleteAtomsPDB(in_pdb, out_pdb, atoms_to_delete):
    with open(in_pdb, 'r') as pdb_file:
        lines = pdb_file.readlines()

    # Filter lines that do not match the atom names in the third column
    cleaned_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            # Extract the atom name (columns 13-16 in standard PDB format)
            atom_name = line[12:16].strip()
            if atom_name not in atoms_to_delete:
                cleaned_lines.append(line)
        else:
            # Keep non-ATOM/HETATM lines unchanged
            cleaned_lines.append(line)
    # Write the cleaned lines to the output file
    with open(out_pdb, 'w') as cleaned_file:
        cleaned_file.writelines(cleaned_lines)



# write leap file for AMBER input
def writeLeap(path, pdb_file, leap_file, 
              bonds, chromophore_list, charge,
              add_ions = ['NA', 'O'], save_pdb = True, water_box = 10.0):
    '''
    chromophore_list = list of instances of Chromophore class
    '''

    loaded_dyes = dict()                                    # store which chromophore templates have already been imported

    with open(os.path.join(path, leap_file), 'w') as f:
        # (1) load force fields
        f.write("source leaprc.DNA.OL15\n")                 # load DNA forcefield 
        f.write("source leaprc.gaff\n")                     # load other forcefield 
        f.write("source leaprc.water.tip3p\n")              # load water forcefield
        # load forcefield parameters for overlap region between DNA 
        f.write(f"loadAmberParams ./createStructure/connectparms.frcmod\n")

        # (2) load information about parameters and connectivity for each attached chromophore
        for chromophore in chromophore_list:
            # avoid double loading of chromophore
            if chromophore.dye_name in loaded_dyes:
                continue
            else:
                loaded_dyes[chromophore.dye_name] = True       
                # (a) lead molecule and AMBER parameters from 
                f.write(f"{chromophore.dye_name} = loadmol2 ./createStructure/{chromophore.dye_name}/ff_new/{chromophore.dye_name}_del.mol2\n")
                f.write(f"loadAmberParams ./createStructure/{chromophore.dye_name}/ff_new/{chromophore.dye_name}_del.frcmod\n")
                # f.write(f"loadAmberParams ./createStructure/{chromophore.dye_name}/ff/{chromophore.dye_name}.frcmod\n")

                # # delete atoms in mol2 template 
                # for atom in chromophore.delete_atoms:
                #     f.write(f"remove {chromophore.dye_name} {chromophore.dye_name}.1.{atom}\n")
                # (b) change atom types in mol2 template to OL15 nomenclature and also adjust atom names
                for atom in chromophore.rename_atoms:
                    f.write(f"set {chromophore.dye_name}.1.{atom} type {chromophore.rename_types[atom]}\n")     # types
                    f.write(f"set {chromophore.dye_name}.1.{atom} name {chromophore.rename_atoms[atom]}\n")     # names

                # (d) define connect0 and connect1 atoms to specify connectivity wih neighboring residues
                # NOTE : the P and O3' of each chromophore are typically the connecting atoms for residues
                f.write(f"set {chromophore.dye_name}.1 connect0 {chromophore.dye_name}.1.P\n")
                f.write(f"set {chromophore.dye_name}.1 connect1 {chromophore.dye_name}.1.O3'\n")
        
        # (3) load structure.pdb file
        f.write(f"mol = loadpdb {os.path.join(path, pdb_file + '.pdb')} \n")
        # NOTE : SANITY CHECK(S)   
        # f.write(f"desc mol\n")                            # prints all residue names to log file
        # f.write(f"desc mol.6\n")                          # prints all atoms and also connect0 connect1 for residue 6      
        # f.write(f"charge mol\n")                          # prints total charge of the molecule (shoudl be zero)
        # f.write(f"charge mol.5\n")                        # prints charge of residue 5 

        # (4) make bonds
        if len(bonds) > 0:
            for bond in bonds:
                bond0 = ".".join(str(el) for el in bond[0])
                bond1 = ".".join(str(el) for el in bond[1])
                f.write(f"bond mol.{bond0} mol.{bond1} \n")

        # (5) # add ions to make cell neutral
        if add_ions is not None:
             f.write(f"addIons mol {add_ions[0]} {-charge}\n")

        # (6) add water
        f.write(f"solvatebox mol TIP3PBOX {water_box}\n")
        # (7) export AMBER input
        f.write(f"saveAmberParm mol {os.path.join(path, pdb_file)}.prmtop {os.path.join(path, pdb_file)}.rst7\n")
        f.write(f"quit")



def clean_numbers(thearray):
    def check_st(string):
        only_alpha = ""
        for char in string:
            if char.isalpha():
                only_alpha += char
        return only_alpha
    return [check_st(astring) for astring in thearray]

def make_names(symbols,counts):
    all_names = []
    for iatm, sym in enumerate(symbols):
        irange = np.arange(counts[iatm])+1
        sym_array = [ sym + str(aindex) for aindex in irange]
        all_names += sym_array
    return np.array(all_names)

def reset_atomids(mol_df):
    natoms = len(mol_df)
    new_ids = np.arange(1,natoms+1)
    mol_df['atom_id'] = new_ids
    return mol_df


# read QM (DFT/TDDFT) input parameters and return dictionary
def readParams(filename):

    if filename is None:
        return {}
    
    else:
        import ast

        user_params = {}                                        # initialize parameter dictionary
        
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):            # Ignore empty lines and comments
                    continue

                key, value = line.split("=", 1)                 # Split key-value pair
                key = key.strip()
                value = value.strip()

                # Convert values to appropriate types
                try:
                    value = ast.literal_eval(value)             # Safely parse numbers, booleans, lists
                except (ValueError, SyntaxError):
                    pass                                        # Keep as string if not evaluable

                user_params[key] = value

        return user_params

