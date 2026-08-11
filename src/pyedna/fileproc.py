import numpy as np
import pandas as pd

# class that us handling .pdb files
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
