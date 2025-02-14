import numpy as np
import fileProcessing as fp
import subprocess
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
import re
import structure
import quantumTools as qm
import time
import pandas as pd



# TODO : write class to perform MD simulation
class MDSimulation():

    def __init__(self, params):
        self.params = params                    # load MD simulation parameters
        self.trajectory_file = None             # placeholder for trajectory file from AMBER
    
    # run initial minimization
    def runMin(self):
        pass

    def runMD(self):
        # (1) write *.in files for AMBER
        # TODO : add for initial equilibration etc
        # (1.1) production run
        self.writeProductionInput()

        # (2) write SBATCH script
        # TODO : add function in fp

        # (3) run SBATCH script
        pass

    # write input for production run
    def writeProductionInput(self):
        pass



class Trajectory():

    def __init__(self, MDsim, path, trajectory):
        self.path = path
        self.prmtop = trajectory[0]                                     # load *.prmtop
        self.nc = trajectory[1]                                         # load *.nc from Amber MD simulation
        self.out = trajectory[2]                                        # load *.out file
        # make sure *.nc file is NetCDF3 (as required for MDAnalysis) and not NetCDF4 (as created by Amber)
        self.convertTrajectory()

        # create MD analysis object
        self.trajectory_u = mda.Universe(path + self.prmtop, path + self.nc)
        self.num_frames = self.trajectory_u.trajectory.n_frames         # number of frames in trajectory

        # load MDSimulation object which contains all information
        self.MD = MDsim                             # TODO : do we need this?
        if not isinstance(self.MD, MDSimulation):
            raise ValueError("MDsim needs to be instance of MDSimulation class!")
        
        # TODO : make this more flexible
        # parse output information for QM and MD simulations
        self.qm_outs, self.outs_quant, self.outs_class = parseOutput(path + 'qm_out.params', parse_post=True)
        print(self.qm_outs)
        print(self.outs_quant)


    # initialize output based on desired output parameters 
    def initOutput(self):
        # (1) define QM states we are interested in (0-indexed), i.e. (S_0^A , S_{stateB + 1}^B) <--> (S_{stateA + 1}^A, S_0^B)
        # TODO : maybe feed multiple of state pairs here to have more things to examine
        self.states= [self.outs_quant["stateA"], self.outs_quant["stateB"]]
        # this might be the best way forward:
        self.transitions = self.outs_quant["transitions"]

        # TODO: make two df's (one for classical output, one for quantum output) 
        #columns_class = [self.outs[keyfor key in self.outs if]
        print('class dict', self.outs_class)
        columns_class = [key for key, value in self.outs_class.items() if isinstance(value, bool) and value]
        print('column names', columns_class)
        self.output_class = pd.DataFrame(index = range(self.num_frames), columns = columns_class)
        print(self.output_class)

        # (2) which trajectory-ensemble outputs are we interested in:
        # (2.1) classical MD output parameters:
        columns_quant = []


        # (2.2) QM-based output parameters:
        if self.outs_quant["coupling"] == 'both':
            pass

    
    # initialize molecules of shape [molecule_A, molecule_B] where molecule_A/B list with residue indices
    # TODO : add check whether molecule is actually valid (consecutive integers etc.)
    def initMolecules(self, molecules):
        self.molecules = molecules


    # get MDAnalysis object of specified residues at specified time slice
    def getChromophoreSnapshot(self, idx, molecule, conversion = None, cap = True):
        # (1) set time step
        self.trajectory_u.trajectory[idx]
        # (2) get positions of all residues specified in residue_ids
        for id in molecule:
            molecule_u = self.trajectory_u.select_atoms(f'resid {id}')
        # (3) need to cap residues with hydrogens (O3' and OP1)
        molecule_u = self.capResiduesH(molecule_u) if cap else molecule_u
        # (4) define instance of Chromophore class 
        chromophore = structure.Chromophore(molecule_u)
        # (5) convert to other input format for processing of trajectory
        chromophore_conv = self.convertChromophore(chromophore, conversion) if conversion else None

        return chromophore, chromophore_conv
        
    # converts Chromophore instance into desired format for trajectory processing
    # TODO : might want to add this to Chromophore class
    def convertChromophore(self, chromophore, conversion):
        # can only convert to PySCF or QChem input
        if conversion not in ['pyscf', 'qchem']:
            raise ValueError("Specify valid format to Chromophore object to.")
        # convert Chromophore object to PySCF input 
        if conversion == 'pyscf':
            xyz, names = chromophore.xyz, chromophore.names
            molecule_conv = []
            for i, coords in enumerate(xyz):
                atom = [names[i][0], tuple(coords)]
                molecule_conv.append(atom)
        # convert Chromophore object to QChem input
        # TODO : add this
        elif conversion == 'qchem':
            pass

        return molecule_conv
    


    # analyze trajectory based on specific molecules of interest
    def analyzeTrajectory(self, molecules, time_slice = None, **params):
        # (0) unpack arguments, i.e. quantities of interest for the trajectory
        # TODO : make this more flexible and stream-line this better
        conversion = params['conversion']
        
        # (1) time range of interest: time_slice = [idx_start, idx_end]
        # TODO : change this to actual time and not just frame index
        if time_slice is None:                                          # study the whole trajectory
            self.time_slice = [0, self.num_frames - 1]
        else:
            self.time_slice = time_slice

        # (2) initialize molecules of interest
        self.initMolecules(molecules)
        # (2) intialize output of interest
        self.initOutput()

        # (3) analyze trajectory
        # distances = []
        for idx in range(self.time_slice[0], self.time_slice[1] + 1):

            start_time = time.time()
            print(f"*** Running Time Step {idx} ...")

            # # (1) get Chromophores of interest 
            # self.chromophores = []
            # self.chromophores_conv = []
            # for molecule in self.molecules:
            #     chromophore, chromophore_conv = self.getChromophoreSnapshot(idx, molecule, conversion)
            #     self.chromophores.append(chromophore)
            #     self.chromophores_conv.append(chromophore_conv)


            # # (2) get distance between chromophores:
            # distances.append(self.getDistance(self.chromophores[0], self.chromophores[1]))

            # (3) analyze with respect to QM quantities of interest
            # NOTE : test-wise DFT/TDDFT calculation
            # # (3.1) run QM calculation
            # output_qm = qm.doQM_gpu(self.chromophores_conv, self.qm_outs)
            
            # (3.2) post-processing of QM output



            # take time
            end_time = time.time()
            print(f"Elapsed time for step {idx}: {end_time- start_time} seconds")





    # get disatnce between chromophores
    # TODO : add various distance measure to compare them
    def getDistance(self, chromophore_1, chromophore_2):
        com_1, com_2 = chromophore_1.com, chromophore_2.com
        distance = np.linalg.norm(com_1 - com_2)
        return distance


    # TODO : might want to make this more general for atom types to cap etc.
    # especially when we want to move to differen molecules/chromophores
    def capResiduesH(self, molecule):

        # (1) compute position of H-caps 

        # (1.1) get capped atom
        def getAtom2Cap(molecule, atom_name):
            return molecule.select_atoms(f"name {atom_name}").positions[0]    
        # (1.2) find bond partners 
        def getBondPartner(molecule, atom_name):
            atom = molecule.select_atoms(f"name {atom_name}")[0]
            bonded_atom = [bond.partner(atom) for bond in atom.bonds][0]
            return bonded_atom.position
        # (1.3) compute final position based on standard OH bond length and HOX angle (X = C, P)
        # NOTE : we use some literature data here (this does not need to be super exact since the QM computation will take care of that)
        def getHPosition(atom2cap_pos, bondpartner_pos, bond_length = 0.97, angle = 109.5):
            # Convert angle to radians
            angle_rad = np.radians(angle)

            # Compute CO vector and normalize
            v = atom2cap_pos - bondpartner_pos
            u = v / np.linalg.norm(v)
            
            # Find a perpendicular vector (avoid collinearity with z-axis)
            reference = np.array([0, 0, 1]) if abs(np.dot(u, [0, 0, 1])) < 0.9 else np.array([1, 0, 0])
            v_perp = np.cross(u, reference)
            v_perp /= np.linalg.norm(v_perp)

            # Compute the rotated OH vector
            v_OH = np.cross(u, v_perp)
            v_OH /= np.linalg.norm(v_OH)  # Ensure it's a unit vector
            
            # Apply rotation matrix for 120-degree placement
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)
            rotated = cos_theta * v_OH + sin_theta * np.cross(u, v_OH)

            # Scale to OH bond length
            H_pos = atom2cap_pos + bond_length * rotated
            return H_pos

        H_positions = []                            
        for name in ["O3'", "OP1"]:
            # (1.1) get capped atoms
            atom2cap = getAtom2Cap(molecule, name)
            # (1.2) get bond neighbor
            bonded_atom = getBondPartner(molecule, name)
            # (1.3) get H positions
            H_positions.append(getHPosition(atom2cap, bonded_atom))
        
        # (2) define new MDAnalysis object for each molecule

        # (2.1) attach hydrogens to molecule
        def attachHs(molecule, H_positions):
            # (1) atom names, atom types, etc.
            max_H_idx = max([int(re.search(r'H(\d+)', name).group(1)) for name in molecule.atoms.names if re.match(r'H\d+', name)])
            new_atom_names = [f'H{max_H_idx + 1}', f'H{max_H_idx + 2}']
            new_atom_types = ['H', 'H']
            new_residue_name = np.unique(molecule.resnames)[0]
            new_residue_id = np.unique(molecule.resids)[0]
            new_elements = ['H', 'H']
            # (2) H positions
            new_atom_positions = np.array(H_positions)
            # (3) initialize new hydrogen Universe
            Hs = mda.Universe.empty(n_atoms = len(H_positions), trajectory=True)
            # (4) fill in hydrogen Universe
            Hs.add_TopologyAttr("name", new_atom_names)
            Hs.add_TopologyAttr("type", new_atom_types)
            Hs.add_TopologyAttr("element", new_elements)
            Hs.add_TopologyAttr("resname", [new_residue_name])
            Hs.add_TopologyAttr("resid", [new_residue_id])
            Hs.atoms.positions = new_atom_positions
            # (5) merge hydrogen Universe with molecule Universe
            u = mda.Merge(molecule.atoms, Hs.atoms)
            return u
    
        molecule = attachHs(molecule, H_positions)
        return molecule





    # convert *.nc file to NetCDF3 and overwrite
    # TODO : might want to store convert_traj.sh in more global directory
    def convertTrajectory(self):
        cpptraj_command = f"bash convert_traj.sh {self.prmtop} {self.nc}"
        run_conversion = subprocess.Popen(cpptraj_command, shell = True, cwd = self.path, stdout = subprocess.DEVNULL)

    # analyze *.out file
    # TODO : generalize this to other quanity_of_interest
    def analyzeOut(self, path_to_perl, quantity_of_interest = 'ETOT', unit = '(kcal/mol)', plot = False):
        # (1) make directory for MD output
        dir_name = 'perl'
        subprocess.Popen("mkdir -p " + dir_name, cwd = self.path, shell = True)
        # (2) use perl to analyze self.out
        subprocess.Popen(f"{path_to_perl} ../{self.out}" , cwd = os.path.join(self.path, dir_name), shell = True, stdout=subprocess.DEVNULL)
        # (3) analyze quantity of interest
        quantity = np.loadtxt(os.path.join(self.path, dir_name, f'summary.{quantity_of_interest}'), skiprows=1) 
        if plot:
            plt.plot(quantity[:, 0], quantity[:, 1], label = quantity_of_interest)
            plt.xlabel("Time Step")
            plt.ylabel(f"{quantity_of_interest} {unit}")
            plt.legend()
            plt.show()
    
    
    


# set parameters for QM (DFT/TDDFT) simulation
# TODO : allow file not to exist without problem
def setQMSettings(file):
    # default settings
    qm_settings = {
        "basis": "6-31g",
        "xc": "b3lyp",
        "density_fit": False,
        "charge": 0,
        "spin": 0,
        "scf_cycles": 200,
        "verbosity": 4,
        "state_ids": [0],
        "TDA": True,
        "gpu": True,
        "do_tddft": True
    }

    # read in user parameters from file
    user_params = fp.readParams(file)

    # update default parameters
    qm_settings.update(user_params)

    # split into dictionaries for keys related to DFT and TDDFT
    settings_dft = {key: qm_settings[key] for key in ["basis", "xc", "density_fit", "charge", "spin", "scf_cycles", "verbosity"]}
    settings_tddft = {key: qm_settings[key] for key in ["state_ids", "TDA", "do_tddft"]}

    return settings_dft, settings_tddft


# parse output information for QM calculations
# TODO : allow file not to exist without problem
def parseOutput(file, parse_post = False):

    # output default parameters
    # TODO : add to this
    out = {
            "exc" : True,
            "mf"  : False,
            "occ" : False,
            "virt": False,
            "mol" : True,
            "tdm" : True,
            "dip" : False,
            "osc" : False,
            "idx" : False
    }
    # specify user parameters
    user_out = fp.readParams(file)

    # update default settings
    out.update(user_out)

    # split the output parameters into parameters that are relevant only to
    # conductiong QM (DFT/TDDFT) simulations or to post-processing of the QM results
    # TODO : add to this
    qm_outs = {key: out.get(key) for key in ["exc", "mol", "tdm", "mf", "occ", "virt", "dip", "osc", "idx"]}                          # NOTE : only boolean key values
    post_qm = {key: out.get(key) for key in ["stateA", "stateB", "coupling", "transitions"]}
    post_class = {key: out.get(key) for key in ["distance", "distance_type"]} 

    # TODO : add list intialization of quantities we are eventually interested in 

    if parse_post:
        return qm_outs, post_qm, post_class
    else:
        return qm_outs
