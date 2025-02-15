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
# TODO : only for debugging
from joblib import dump, load


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

    def __init__(self, MDsim, path, trajectory, dt):
        self.path = path
        self.prmtop = trajectory[0]                                     # load *.prmtop
        self.nc = trajectory[1]                                         # load *.nc from Amber MD simulation
        self.out = trajectory[2]                                        # load *.out file
        # make sure *.nc file is NetCDF3 (as required for MDAnalysis) and not NetCDF4 (as created by Amber)
        self.convertTrajectory()

        # create MD analysis object
        self.trajectory_u = mda.Universe(path + self.prmtop, path + self.nc)
        self.num_frames = self.trajectory_u.trajectory.n_frames         # number of frames in trajectory
        self.dt = dt                                                    # time step in (ps)

        # load MDSimulation object which contains all information
        self.MD = MDsim                                                 # TODO : do we need this?
        if not isinstance(self.MD, MDSimulation):
            raise ValueError("MDsim needs to be instance of MDSimulation class!")
        
        # TODO : make this more flexible
        # parse output information for QM and MD simulations
        self.qm_outs, self.quant_info, self.class_info = parseOutput(path + 'qm_out.params', parse_trajectory_out=True)


    # initialize output based on desired output parameters 
    def initOutput(self, output_length):

        # (1) define QM states we are interested in (0-indexed), i.e. (S_0^A , S_{stateB + 1}^B) <--> (S_{stateA + 1}^A, S_0^B)
        self.transitions = self.quant_info[0]["transitions"]

        # TODO : might also want to add DataFrame for the direct QM (DFT/TDDFT) outputs 

        # (2) which trajectory-ensemble outputs are we interested in:

        # (2.1) classical MD output parameters:
        columns_class = ["time"] + [key for key, value in self.class_info[0].items() if isinstance(value, bool) and value]
        if columns_class:
            self.output_class = pd.DataFrame(index = range(output_length), columns = columns_class)
        else:
            self.output_class = pd.DataFrame()
        
        # (2.2) quantum output parameters (output the same outputs for every transition in self.transitions)
        # NOTE : since states are 0-indexed, 0 actually corresponds to the 1st excited state of molecule A/B, 1 to the
        # 2nd excited state of molecule A/B etc.
        self.transition_names = [f"[A({states[0] + 1}), B(0)] <--> [A(0), B({states[1] + 1})]" for states in self.transitions]
        self.quant_info[0].pop("transitions")
        columns_per_transitions = [key for key, value in self.quant_info[0].items() if isinstance(value, bool) and value]
        if columns_per_transitions:
            columns_quant = pd.MultiIndex.from_tuples(
                [("time", "")] +
                [(transition_name, value_name) for transition_name in self.transition_names for value_name in columns_per_transitions]
            ) 
            self.output_quant = pd.DataFrame(index = range(output_length), columns = columns_quant)
            print('check', self.output_quant.columns)
        else:
            self.output_quant = pd.DataFrame()
        

        print("*** Intialization of output done!")
        
    # TODO : write simulation data into the header
    def writeOutputFiles(self, data_frame, file_name, write_meta_data = True):
        # TODO : write meta data into header
        # store DateFrame (classical or quantum) with meta data header (optional)
        if not data_frame.empty:
            with open(file_name, "w") as f:
                # optional: write meta data
                if write_meta_data:
                    pass
                # write output
                data_frame.to_csv(f, sep = "\t", index=False)

        
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
    

    # analyze trajectory time step with regards to 
    # TODO : also put qm.doDFT_gpu() in here eventually to save QM calculation if we don't want to do it
    def analyzeSnapshotQuantum(self, time_idx, output_qm):
        # TODO : implement check whether we even have to run this if nothing specified

        # (0) time (ps)
        self.output_quant.loc[time_idx, ("time", "")] = time_idx * self.dt


        # (1) loop over all specified transitions
        for i, states in enumerate(self.transitions):
            
            # (a) get Coulombic coupling information if desired
            if self.quant_info[0]["coupling"]: 
                # compute coupling based on QM (DFT/TDDFT) output
                coupling_out = qm.getVCoulombic(output_qm['mol'], output_qm['tdm'], states, coupling_type=self.quant_info[1]['coupling'])
                # further scaffold the self.outpu_quant array to aacount for all coupling information
                sub_columns = ['coupling cJ', 'coupling cK', 'coupling V_C']
                df = pd.DataFrame(index = range(self.num_frames), columns=pd.MultiIndex.from_product([[self.transition_names[i]], sub_columns]))
                self.output_quant = self.output_quant.drop(columns=[(self.transition_names[i], "coupling")]).join(df)
                # add to output dict
                self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in coupling_out.keys()]] = list(coupling_out.values())

            # (b) get excitation energies
            if self.quant_info[0]["excited_energies"]:
                # get excited state energies based on QM (DFT/TDDFT) output
                energies_out = qm.getExcEnergies(output_qm['exc'], states, excitation_energy_type=self.quant_info[1]['excited_energies'])
                # further scaffold the self.outpu_quant array to aacount for all excited state energies
                sub_columns = ['energy A', 'energy B']
                df = pd.DataFrame(index = range(self.num_frames), columns=pd.MultiIndex.from_product([[self.transition_names[i]], sub_columns]))
                self.output_quant = self.output_quant.drop(columns=[(self.transition_names[i], "excited_energies")]).join(df)
                # add to output dict
                self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in energies_out.keys()]] = list(energies_out.values())


    # TODO : this function needs to be updated a lot and more functionalities implemeted
    def analyzeSnapshotClassical(self, time_idx):

         # (0) time (ps)
        self.output_class.loc[time_idx, "time"] = time_idx * self.dt

        # (1) compute distance metric:
        # TODO : add an actual function here and not just some kind of dummy
        if self.class_info[0]["distance"]:
            self.output_class.loc[time_idx] = 4
        
                
            
    # analyze trajectory based on specific molecules of interest
    def loopTrajectory(self, molecules, time_slice = None, **params):
        # (0) unpack arguments, i.e. quantities of interest for the trajectory
        # TODO : make this more flexible and stream-line this better
        conversion = params['conversion']
        
        # (1) time range of interest: time_slice = [idx_start, idx_end]
        # TODO : change this to actual time and not just frame index
        if time_slice is None:                                          # study the whole trajectory
            self.time_slice = [0, self.num_frames - 1]
        else:
            self.time_slice = time_slice

        self.initMolecules(molecules)                                   # intialize molecule information
        self.initOutput(self.time_slice[1]  - self.time_slice[0])       # initialize outputs

        # (3) analyze trajectory
        for idx in range(self.time_slice[0], self.time_slice[1] + 1):

            start_time = time.time()
            print(f"*** Running Time Step {idx} ...")

            # (1) get Chromophores of interest 
            self.chromophores = []
            self.chromophores_conv = []
            for molecule in self.molecules:
                chromophore, chromophore_conv = self.getChromophoreSnapshot(idx, molecule, conversion)
                self.chromophores.append(chromophore)
                self.chromophores_conv.append(chromophore_conv)


            # # (2) get distance between chromophores:
            # distances.append(self.getDistance(self.chromophores[0], self.chromophores[1]))

            # # (3) analyze with respect to QM quantities of interest
            # # NOTE : test-wise DFT/TDDFT calculation
            # # (3.1) run QM calculation
            # output_qm = qm.doQM_gpu(self.chromophores_conv, self.qm_outs)
            # # # temporarily store ouput_qm for debugging
            # print('tim idx', idx)
            # dump(output_qm, f"output_qm_{idx}.joblib")


            # (3.2) post-processing of QM output
            # TODO : load for simplicity here
            output_qm = load(f"output_qm_{idx}.joblib")
            print('output DFT/TDDFT', output_qm['exc'])

            # TODO : only do this if we have quantum aspects to analyze
            self.analyzeSnapshotQuantum(idx, output_qm)
            # TODO : only do the following if we have classical aspects to study
            self.analyzeSnapshotClassical(idx)
            

            # take time per time step
            end_time = time.time()
            print(f"Elapsed time for step {idx}: {end_time- start_time} seconds")
            print('TT', self.output_quant)

        # (4) write output files
        # (4.1) quantum output
        self.writeOutputFiles(self.output_quant, "out_quant.txt")
        # (4.2) classical output
        self.writeOutputFiles(self.output_class, "out_class.txt")



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
def parseOutput(file, parse_trajectory_out = False, verbose = True):

    # output default parameters
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
    # read user parameters four output
    user_out = fp.readParams(file)

    # update default settings
    out.update(user_out)

    # split the output parameters into parameters that are relevant only to
    # conductiong QM (DFT/TDDFT) simulations or to post-processing of the trajectory 
    # (1) QM (DFT/TDDFT) outputs (NOTE : only boolean)
    qm_outs = {key: out.get(key) for key in ["exc", "mol", "tdm", "mf", "occ", "virt", "dip", "osc", "idx"]}     
    # TODO : in order to evaluate some of the post-processing output, we need to have some of this flags set to True
    # might want to implement a checkpoint here               

    # (2) trajectory-based outputs per time steps
    # (2.1) quantum-mechanical based parameters and methods
    post_qm = {key: out.get(key) for key in ["transitions", "coupling", "coupling_type", "excited_energies"]}               # all QM options                         
    qm_flags = {key: value for key, value in post_qm.items() if isinstance(value, bool) and value}                          # NOTE : only bool/True param
    qm_flags.update({"transitions": post_qm["transitions"]})
    # for each flag we either set specified methods_type or default
    qm_methods = {
        key: post_qm.get(f"{key}_type", "default") for key in qm_flags if isinstance(qm_flags[key], bool)
    }

    # (2.2) classical parameters and methods
    post_class = {key: out.get(key) for key in ["distance", "distance_type"]}                                               # all MD options
    class_flags = {key: value for key, value in post_class.items() if isinstance(value, bool) and value}                    # NOTE : only bool/True param
    # for each flag we either set specified methods_type or default
    class_methods = {
        key: post_class.get(f"{key}_type", "default") for key in class_flags
    }

    if parse_trajectory_out:
        if verbose:
            # print parsed output for trajectory analysis
            print(" *** Parsed Output for Trajectory Analysis:")
            print(f"(1) classical parameters to evaluate at each time step: {', '.join(class_flags.keys())}")
            print(f"(1) we use the following methods (in order): {', '.join(class_methods.values())}")
            print(f"(2) we study the following state transitions [stateA, stateB]: {', '.join(str(transition) for transition in qm_flags['transitions'])}")
            print(f"(2) quantum parameters to evaluate at each time step for each transition: {', '.join(key for key, value in qm_flags.items() if isinstance(value, bool))}")
            print(f"(2) we use the following methods (in order): {', '.join(qm_methods.values())}")
        return qm_outs, [qm_flags, qm_methods], [class_flags, class_methods]
    else:
        return qm_outs
