import numpy as np
import pandas as pd
import os
import subprocess

# from current package
from . import trajectory as traj

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


# class that is handling ORCA input
class ORCAInput():
    
    def __init__(self, file_name, pyscf_mol, settings_dft, do_tddft = True, do_geom = True, implicit_sol = True):
        self.file_name = file_name
        self.pyscf_mol = pyscf_mol
        self.settings_dft = settings_dft
        # specify boolean options
        self.do_tddft = do_tddft
        self.do_geom = do_geom
        self.implicit_sol = implicit_sol

    # write ORCA input script
    def write(self):

        with open(self.file_name, "w") as f:

            # write methods and options
            self.write_methods(f)

            # write TDDFT
            if self.do_tddft:
                self.write_tddft(f)

            # write geometry optimization
            if self.do_geom:
                self.write_geom(f)

            # write implicit solvent
            if self.implicit_sol:
                self.write_solvent(f)
            
            # write parallization instructions
            f.write("%pal nprocs 8 end \n")
            f.write("\n")

            # write coordinates
            self.write_coords_from_pyscf(f)

    def write_methods(self, f):
        self.basis = self.settings_dft["basis"].upper()
        self.xc = self.settings_dft["xc"].upper()
        # information for TDDFT and geometry optimization
        if self.do_tddft:
            f.write(f"! {self.xc} {self.basis} TightSCF TightOpt CPCM\n")
        else:
            f.write(f"! {self.xc} {self.basis} TightSCF CPCM\n")
        f.write("\n")
    

    def write_tddft(self, f, nroots = 5, root = 1):
        f.write("%tddft\n")
        f.write(f"  NRoots {nroots}\n")
        f.write(f"  IRoot {root}\n")
        f.write("end\n")
        f.write("\n")

    def write_geom(self, f):
        f.write("%geom\n")
        f.write(f"  calc_hess false\n")
        f.write("end\n")
        f.write("\n")

    def write_solvent(self, f, epsilon = 78.4):
        f.write("%cpcm\n")
        f.write(f"  epsilon {epsilon}\n")
        f.write("end\n")
        f.write("\n")

    # write coordinates
    def write_coords_from_pyscf(self, f):
        f.write(f"* xyz {self.pyscf_mol.charge} {self.pyscf_mol.multiplicity}\n")
        for i, coord in enumerate(self.pyscf_mol.atom_coords()):
            symbol = self.pyscf_mol.atom_symbol(i)
            f.write(f"{symbol}    {coord[0]:.4f}   {coord[1]:.4f}   {coord[2]:.4f}\n")
        f.write("*\n")
            
    
    def run(self):
        orca_home = os.getenv("ORCAHOME")
        mpi_home = os.getenv("MPIHOME", os.path.expanduser("~/opt/openmpi-4.1.6"))

        if not os.path.exists(os.path.join(mpi_home, "bin", "mpirun")):
            raise EnvironmentError("OpenMPI 4.1.6 not found. Please install or set MPIHOME.")

        orca_bin = os.path.join(orca_home, "orca")
        if not os.path.isfile(orca_bin):
            raise FileNotFoundError(f"ORCA binary not found at: {orca_bin}")

        env = os.environ.copy()
        env["PATH"] = f"{os.path.join(mpi_home, 'bin')}:{orca_home}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{os.path.join(mpi_home, 'lib')}:{orca_home}:{env.get('LD_LIBRARY_PATH', '')}"
        env["RSH_COMMAND"] = "ssh"
        env["OMP_NUM_THREADS"] = "8"

        outfile = os.path.splitext(self.file_name)[0] + ".out"
        cmd = [orca_bin, self.file_name]

        with open(outfile, "w") as out_f:
            process = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True, env=env)
            _, stderr = process.communicate()

        if process.returncode != 0:
            print(f"ORCA run failed. STDERR:\n{stderr}")
        else:
            print(f"ORCA run completed successfully. Output written to {outfile}")
    



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
def writeLeap(pdb_file, leap_file, 
              bonds, chromophore_list, charge,
              add_ions = ['NA', 'O'], save_pdb = True, water_box = 10.0):
    '''
    chromophore_list = list of instances of Chromophore class
    '''

    loaded_dyes = dict()                                    # store which chromophore templates have already been imported

    #with open(os.path.join(path, leap_file), 'w') as f:
    with open(leap_file, 'w') as f:

        # (1) load force fields
        f.write("source leaprc.DNA.OL15\n")                 # load DNA forcefield 
        f.write("source leaprc.gaff\n")                     # load other forcefield for custom dyes
        f.write("source leaprc.water.tip3p\n")              # load water forcefield

        # (2) load information about parameters and connectivity for each attached chromophore
        for chromophore in chromophore_list:
            # avoid double loading of chromophores
            if chromophore.dye_name in loaded_dyes:
                continue
            else:
                loaded_dyes[chromophore.dye_name] = True       
                # (a) load molecule and AMBER ff parameters from 
                #(a.1)  NOTE : load .mol2 and .frcmod created for the dye with deleted residues for attachment. If we don't use the structure with 
                # deleted atoms, then tleap has problems processing the produced charges by antechamber for .frcmod. 
                f.write(f"{chromophore.dye_name} = loadmol2 {os.path.join(chromophore.path, f'{chromophore.dye_name}_del.mol2')}\n")
                f.write(f"loadAmberParams {os.path.join(chromophore.path, f'{chromophore.dye_name}_del.frcmod')}\n")
                # (a.2) load forcefield for connecting region (TODO : this is the same for CY3/CY5 but might be different if we add more dyes in the future)
                f.write(f"loadAmberParams {os.path.join(chromophore.path, 'connectparms.frcmod')}\n")

                # (b) change atom types in mol2 template to OL15 nomenclature and also adjust atom names
                for atom in chromophore.rename_atoms:
                    f.write(f"set {chromophore.dye_name}.1.{atom} type {chromophore.rename_types[atom]}\n")     # types
                    f.write(f"set {chromophore.dye_name}.1.{atom} name {chromophore.rename_atoms[atom]}\n")     # names

                # (d) define connect0 and connect1 atoms to specify connectivity wih neighboring residues
                # NOTE : the P and O3' of each chromophore are typically the connecting atoms for residues
                f.write(f"set {chromophore.dye_name}.1 connect0 {chromophore.dye_name}.1.P\n")
                f.write(f"set {chromophore.dye_name}.1 connect1 {chromophore.dye_name}.1.O3'\n")
        
        # (3) load structure.pdb file
        #f.write(f"mol = loadpdb {os.path.join(path, pdb_file + '.pdb')} \n")
        f.write(f"mol = loadpdb {pdb_file + '.pdb'} \n")


        # NOTE : SANITY CHECK(S)   
        # f.write(f"desc mol\n")                            # prints all residue names to log file
        # f.write(f"desc mol.6\n")                          # prints all atoms and also connect0 connect1 for residue 6      
        # f.write(f"charge mol\n")                          # prints total charge of the molecule (should be integer)
        # f.write(f"charge mol.8\n")                        # prints charge of residue 8 


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

        # (7) optional: save .pdb file
        if save_pdb:
            f.write(f"savepdb mol {pdb_file}.pdb\n")

        # (8) export AMBER input
        f.write(f"saveAmberParm mol {pdb_file}.prmtop {pdb_file}.rst7\n")
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

