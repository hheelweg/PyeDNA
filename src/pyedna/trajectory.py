import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
import re
import time
import pandas as pd

# from current package
from . import structure
from . import quanttools as qm
from . import fileproc as fp
from . import config 
from . import utils



class MDSimulation():

    def __init__(self, dna_params, params_file, sim_name = os.path.dirname(os.path.abspath(__file__))):

        self.params_file = params_file                              # load MD simulation parameters
        self.simulation_name = sim_name                             # name of MD simulation

        # load structural information of DNA structure
        self.dna_params = dna_params                                
        # load input parameters for minimization/MD
        self.md_params = self.parseInputParams(self.dna_params, params_file) 
        # store important parameters to class
        self.storeMDinfo()

        # initialize prmtop and rst7 file attributes
        self.prmtop, self.rst7 = None, None      
    
    
    # store Productiopn run attributes to class for later reference
    def storeMDinfo(self):
        # external parameters (temperature, pressure)
        self.temp = self.md_params["temp"]
        self.pressure = self.md_params["pres"]
        # time step parameters
        self.dt = self.md_params["dt"]                                      # in ps
        self.total_time = self.md_params["prod_nstlim"] * self.dt           # in ps
        # number of timesteps
        self.traj_steps = self.md_params["prod_nstlim"] // self.md_params["prod_ntwx"]
        self.total_steps = self.md_params["prod_nstlim"]

    
    # TODO : this is writen for double-helix DNA (constraints for MD/energy minimization might change if we move
    # to different DNA structures)
    @staticmethod
    def parseInputParams(dna_params, file = None):
        
        if dna_params["dna_type"] != 'double_helix':
            raise NotImplementedError("MD Simulations currently only implemented for 'double_helix' DNA type")

        # (1) set default parameters
        # (1.1.) static energy minimization 
        min_params = {
                        'min_imin'      :       1,
                        'min_maxcyc'    :       1000,		
                        'min_ncyc'      :       500, 		
                        'ntb'           :       1,			
                        'ntr'           :       1,			
                        'iwrap'         :       1,			
                        'cut'           :       8.0			
        }
        # (1.2) equilibration
        eq_params = {
                        'md_imin'       :       0,
                        'eq1_nstlim'    :       10000,
                        'eq2_nstlim'    :       50000,
                        'dt'            :       0.002,
                        'eq1_irest'     :       0,
                        'eq2_irest'     :       1,
                        'eq1_ntx'       :       1,
                        'eq2_ntx'       :       5,
                        'ntc'           :       2,
                        'ntf'           :       2,
                        'temp_i'        :       0.0,
                        'temp'          :       300.0,
                        'ntp'           :       2,
                        'pres'          :       1.0,
                        'taup'          :       2.0,
                        'ntt'           :       3,
                        'gamma_ln'      :       5.0,
                        'ig'            :       -1,
                        'ioutfm'        :       0,
                        'eq1_ntpr'      :       100,
                        'eq2_ntpr'      :       5000,
                        'eq1_ntwx'      :       100,
                        'eq2_ntwx'      :       5000,
                        'eq1_ntwr'      :       100,
                        'eq2_ntwr'      :       5000
        }
        # (1.3) production
        prod_params = {
                        'prod_nstlim'   :       1000000,
                        'ntx'           :       5,
                        'irest'         :       1,
                        'prod_ntpr'     :       5000,
                        'prod_ntwx'     :       5000,
                        'prod_ntwr'     :       50000
        }

        # merge all dicts together
        md_params = dict()
        md_params.update(min_params)
        md_params.update(eq_params)
        md_params.update(prod_params)

        # (2) read user parameters
        user_params = fp.readParams(file)

         # (3) update default settings
        md_params.update({key: user_params[key] for key in md_params if key in user_params})

        # (4) custom restraints for DNA structure based on DNA structure
        # NOTE : currently only implemented for dna_type = 'double_helix'
        seq_length = len(dna_params["dna_sequence"])
        num_residues = 2 * seq_length
        restr_params = {
                        'res_start'     :       1,
                        'res_end'       :       num_residues,
                        'res_mask'      :       f"'(:1,{seq_length},{seq_length + 1},{num_residues})'",
                        'res_fstrong'   :       500.0,
                        'res_fweaker'   :       10.0,
                        'res_fweak'     :       5.0
        }

        # add restraint information to md_params dict
        md_params.update(restr_params)

        return md_params

    # load MD templates for AMBER input
    @staticmethod
    def loadTemplate(template_name):
        # get directory for MD templates
        md_template_dir = os.path.join(config.PROJECT_HOME, 'data', 'md_templates')
        # find template
        template_file = utils.findFileWithName(f"{template_name}.in", dir=md_template_dir)
        # load template
        with open(template_file, "r") as file:
            template = file.read()
        return template
    
    # function that writes Amber .in files 
    @staticmethod
    def writeAMBERInput(md_params, input_type, name = 'test'):
        # (0) check if input type is valid
        valid_input_types = ["eq1", "eq2", "min1", "min2", "prod"]
        if input_type not in valid_input_types:
            raise KeyError("Specify valid input type to write .in file")
        # (1) load template
        template = MDSimulation.loadTemplate(template_name=input_type)
        # (2) fill in template
        filled_template = template.format(**md_params)
        # (3) write AMBER input file
        with open(f"{input_type}_{name}.in", "w") as file:
            file.write(filled_template)


    # initialize Simulation by loading .prmtop and .rst7 files
    def initSimulation(self, prmtop_file, rst7_file):
        self.prmtop, self.prmtop_name = prmtop_file, os.path.basename(prmtop_file)
        self.rst7, self.rst7_name = rst7_file, os.path.basename(rst7_file)


    # make AMBER executbale command
    @staticmethod
    def makeCommand(executable, in_file, out_file,
                    topology_file, in_coord_file, out_coord_file, ref_coord_file, netcdf_file = None):
        command = " ".join([
                            f"srun {executable} -O",                                # -O to overwrite output
                            f"-i {in_file}", f"-o {out_file}",                      # names of .in and .out file
                            f"-p {topology_file}",                                  # topology files
                            f"-c {in_coord_file}", f"-r {out_coord_file}",          # in and out coordinate/sturcture files 
                            f"-ref {ref_coord_file}"                                # file with reference coordinates
                            ])
        if netcdf_file:
            command += f" -x {netcdf_file}"                                         # NetCDF file for trajectory analysis
        return command


    # run minimizations
    def runMinimization(self):

        # (1) write AMBER input for minimizations
        # (1.1) solvent + ion relaxation
        MDSimulation.writeAMBERInput(self.md_params, input_type = 'min1', name = self.simulation_name)
        # (1.2) entire system
        MDSimulation.writeAMBERInput(self.md_params, input_type = 'min2', name = self.simulation_name)

        # (2) run minimization
        # (2.1) solvent + ions
        command = MDSimulation.makeCommand( executable = "sander",
                                            in_file = f"min1_{self.simulation_name}.in",
                                            out_file = f"min1_{self.simulation_name}.out",
                                            topology_file = self.prmtop_name,
                                            in_coord_file = self.rst7_name,
                                            out_coord_file = f"min1_{self.simulation_name}.ncrst",
                                            ref_coord_file = self.rst7_name
                                            )
        subprocess.run(command, shell = True)
        # (2.2) entire system
        command = MDSimulation.makeCommand( executable = "sander",
                                            in_file = f"min2_{self.simulation_name}.in",
                                            out_file = f"min2_{self.simulation_name}.out",
                                            topology_file = self.prmtop_name,
                                            in_coord_file = f"min1_{self.simulation_name}.ncrst",
                                            out_coord_file = f"min_{self.simulation_name}.ncrst",
                                            ref_coord_file = f"min1_{self.simulation_name}.ncrst"
                                            )
        subprocess.run(command, shell = True)



    # run equilibration
    def runEquilibration(self):

        # (1) write AMBER input for equilibrations
        # (1.1) heat system with DNA restraint 
        MDSimulation.writeAMBERInput(self.md_params, input_type = 'eq1', name = self.simulation_name)
        # (1.2) NPT equilibration and slowly remove DNA restraint
        MDSimulation.writeAMBERInput(self.md_params, input_type = 'eq2', name = self.simulation_name)

        # (2) run equilibration
        # (2.1) heat system with DNA restraint 
        command = MDSimulation.makeCommand( executable = "pmemd.cuda",
                                            in_file = f"eq1_{self.simulation_name}.in",
                                            out_file = f"eq1_{self.simulation_name}.out",
                                            topology_file = self.prmtop_name,
                                            in_coord_file = f"min_{self.simulation_name}.ncrst",                # minimization output
                                            out_coord_file = f"eq1_{self.simulation_name}.ncrst",
                                            ref_coord_file = f"min_{self.simulation_name}.ncrst",               # minimization output
                                            netcdf_file = f"eq1_{self.simulation_name}.nc"
                                            )
        subprocess.run(command, shell = True)
        # (2.2) NPT equilibration and slowly remove DNA restraint
        command = MDSimulation.makeCommand( executable = "pmemd.cuda",
                                            in_file = f"eq2_{self.simulation_name}.in",
                                            out_file = f"eq2_{self.simulation_name}.out",
                                            topology_file = self.prmtop_name,
                                            in_coord_file = f"eq1_{self.simulation_name}.ncrst",                # equilibration output
                                            out_coord_file = f"eq2_{self.simulation_name}.ncrst",
                                            ref_coord_file = f"min_{self.simulation_name}.ncrst",               # minimization output
                                            netcdf_file = f"eq2_{self.simulation_name}.nc"
                                            )
        subprocess.run(command, shell = True)



    # run production
    def runProduction(self):

        # (2) write AMBER input for MD production run 
        MDSimulation.writeAMBERInput(self.md_params, input_type = 'prod', name = self.simulation_name)

        # (2) run production run
        command = MDSimulation.makeCommand( executable = "pmemd.cuda",
                                            in_file = f"prod_{self.simulation_name}.in",
                                            out_file = f"prod_{self.simulation_name}.out",
                                            topology_file = self.prmtop_name,
                                            in_coord_file = f"eq2_{self.simulation_name}.ncrst",                # equilibration output
                                            out_coord_file = f"{self.simulation_name}.ncrst",
                                            ref_coord_file = f"min_{self.simulation_name}.ncrst",               # minimization output
                                            netcdf_file = f"{self.simulation_name}.nc"                          # trajectory file of interest
                                            )
        subprocess.run(command, shell = True)


    # clean desired files as desired by user
    def cleanFiles(self, clean_level):
        # only keep trajectory output file .nc and .out for production run
        if clean_level == 0:
            # remove all .in files in cwd
            subprocess.run("rm *.in", shell = True) 
            # remove all .out files in cwd except for production run          
            keep_file = f"prod_{self.simulation_name}.out"
            subprocess.run(["find", ".", "-type", "f", "-name", "*.out", "!", "-name", keep_file, "-delete"])
            # remove all .ncrst files in cwd except for the one from minimization
            keep_file = f"min_{self.simulation_name}.ncrst"
            subprocess.run(["find", ".", "-type", "f", "-name", "*.ncrst", "!", "-name", keep_file, "-delete"])
            # remove all .nc files except for production run
            keep_file = f"{self.simulation_name}.nc"
            subprocess.run(["find", ".", "-type", "f", "-name", "*.nc", "!", "-name", keep_file, "-delete"])

        # only keep trajectory output file .nc for production run and all .out files
        elif clean_level == 1:
            # remove all .in files in cwd
            subprocess.run("rm *.in", shell = True)
            # remove all .ncrst files in cwd except for the one from minimization
            keep_file = f"min_{self.simulation_name}.ncrst"
            subprocess.run(["find", ".", "-type", "f", "-name", "*.ncrst", "!", "-name", keep_file, "-delete"])
            # remove all .nc files except for production run
            keep_file = f"{self.simulation_name}.nc"
            subprocess.run(["find", ".", "-type", "f", "-name", "*.nc", "!", "-name", keep_file, "-delete"]) 

        # keep everything but the .ncrst files
        elif clean_level == 2:
            # remove all .ncrst files in cwd except for the one from minimization
            keep_file = f"min_{self.simulation_name}.ncrst"
            subprocess.run(["find", ".", "-type", "f", "-name", "*.ncrst", "!", "-name", keep_file, "-delete"])
            # remove all .nc files except for production run
            keep_file = f"{self.simulation_name}.nc"
            subprocess.run(["find", ".", "-type", "f", "-name", "*.nc", "!", "-name", keep_file, "-delete"])

        # keep everything
        elif clean_level == 3:
            pass






class Trajectory():

    def __init__(self, MDsim, trajectory_data, traj_params_file = 'traj.params'):

        self.prmtop = trajectory_data[0]                                # load *.prmtop
        self.nc = trajectory_data[1]                                    # load *.nc from Amber MD simulation
        self.out = trajectory_data[2]                                   # load *.out file
        # make sure *.nc file is NetCDF3 (as required for MDAnalysis) and not NetCDF4 (as created by Amber)
        self.convertTrajectory()

        # create MDAnalysis object
        self.trajectory_u = mda.Universe(self.prmtop, self.nc)
        self.num_frames = self.trajectory_u.trajectory.n_frames         # number of frames in trajectory

        # load MDSimulation object which contains all information
        self.MD = MDsim                                                 
        if not isinstance(self.MD, MDSimulation):
            raise ValueError("MDsim needs to be instance of MDSimulation class!")
        self.dt = self.MD.dt
        
        # TODO : make this more flexible with regards to path
        # parse output information for QM and MD simulations
        self.qm_outs, self.quant_info, self.class_info, self.time_slice = self.parseParameters(traj_params_file, parse_trajectory_out=True)

        self.defined_molecules = False                                  # flag to track whether molecules have been defined


    # set parameters for QM (DFT/TDDFT) simulation
    # TODO : allow file not to exist without problem
    @staticmethod
    def setQMSettings(file = None):
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

    # parse output information for trajectory analysis (classical + quantum) 
    @staticmethod
    def parseParameters(file, parse_trajectory_out = False, verbose = True):

        # default parameters
        # TODO : extend this list to have every possible parameter defaulted to something
        out = {
                "time_slice" :  None,
                "exc" :         True,
                "mf"  :         False,
                "occ" :         False,
                "virt":         False,
                "mol" :         True,
                "tdm" :         True,
                "dip" :         False,
                "osc" :         False,
                "idx" :         False,
                "file_qm" :     "out_quant.txt",
                "file_class":   "out_class.txt"

        }
        # read user parameters
        user_out = fp.readParams(file)
        # TODO : check if there is some key words not specified in out and throw parsing error

        # update default settings
        out.update(user_out)

        # (0) time range of interest
        time_range = out["time_slice"]

        # split the parameters into parameters that are relevant only to
        # conductiong QM (DFT/TDDFT) simulations or to post-processing of the trajectory 
        # (1) QM (DFT/TDDFT) outputs (NOTE : only boolean)
        qm_outs = {key: out.get(key) for key in ["exc", "mol", "tdm", "mf", "occ", "virt", "dip", "osc", "idx"]}    
        # TODO : in order to evaluate some of the post-processing output, we need to have some of this flags set to True
        # might want to implement a checkpoint here               

        # (2) trajectory-based outputs per time steps
        # (2.1) quantum-mechanical based parameters and methods
        qm_options = ["transitions", "coupling", "coupling_type", "excited_energies", "dipole_moments", "osc_strengths"]
        post_qm = {key: out.get(key) for key in qm_options}                
        qm_flags = {key: value for key, value in post_qm.items() if isinstance(value, bool) and value}                          # NOTE : only bool/True param
        qm_out_file = out["file_qm"]
        # checkpoints: manually check if flags in out match with qm_flags:
        # TODO : maybe there is a better way to do this?
        qm_outs['exc'] = True if post_qm["excited_energies"] else qm_outs['exc']
        qm_outs['dip'] = True if post_qm["dipole_moments"] else qm_outs['dip']
        qm_outs['osc'] = True if post_qm["osc_strengths"] else qm_outs['osc']
        qm_outs['mol'] = True if post_qm["coupling"] else qm_outs['mol']
        qm_outs['tdm'] = True if post_qm["coupling"] else qm_outs['tdm']

        qm_flags.update({"transitions": post_qm["transitions"]})
        # for each flag we either set specified methods_type or default
        qm_methods = {
            key: post_qm.get(f"{key}_type", "default") for key in qm_flags if isinstance(qm_flags[key], bool)
        }

        # (2.2) classical parameters and methods
        post_class = {key: out.get(key) for key in ["distance", "distance_type"]}                                               # all MD options
        class_flags = {key: value for key, value in post_class.items() if isinstance(value, bool) and value}                    # NOTE : only bool/True param
        class_out_file = out["file_class"]
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
            return qm_outs, [qm_flags, qm_methods, qm_out_file], [class_flags, class_methods, class_out_file], time_range
        else:
            return qm_outs


    # initialize molecules of shape [molecule_A, molecule_B] where molecule_A/B list with residue indices
    # TODO : add check whether molecule is actually valid (consecutive integers etc.)
    @staticmethod
    def parseMolecules(file):

        # molecule default parameters
        mols = {
                "molecule_1" :      None,
                "molecule_2" :      None,
                "name_1" :          "D",
                "name_2" :          "A",
        }

        # read user parameters for molecules
        user_mols = fp.readParams(file)

        # update default settings
        mols.update(user_mols) 

        # store molecule IDs
        molecules = [value for key, value in mols.items() if key.startswith("molecule_") and value is not None]
        # store molecule names
        molecule_names = [value for key, value in mols.items() if key.startswith("name_") and value is not None]#.sort(key=lambda x: int(x.split('_')[1]))

        # checkpoint
        assert(len(molecule_names) == len(molecules))
        if len(molecules) == 0:
            raise Warning("Need to specify at least one molecule in mols.params")
        elif len(molecules) > 2:
            raise NotImplementedError("More than 2 molecules (currently) not implemented!")

        return molecules, molecule_names

    # read and parse DataFrame trajectory analysis output
    @staticmethod
    def readOutputFiles(file, output_type, output_info):
        # (1) read file and parse output info 
        # (1.1) DataFrame with quantum information
        if output_type == 'quantum':
            df = pd.read_csv(file, sep='\t', header=[0,1])
            df.columns = [(col[0] if col[0] == "time" else col) for col in df.columns]
            # parse output information contained within data_frame
            _, qm_info, _  = Trajectory.parseOutput(output_info, parse_trajectory_out=True, verbose=False)
            # get names of the transitions under study
            transition_dict = {}
            for states in qm_info[0]["transitions"]:
                key = Trajectory.generateTransitionString(states)
                transition_dict[str(states)] = key
            # return df, transition name dict, and output information
            return df, transition_dict, qm_info
        # (1.2) DataFrame with classical information
        elif output_type == 'classical':
            df = pd.read_csv(file, sep='\t', header=0)
            # parse output information contained within data_frame
            _, _, class_info  = Trajectory.parseOutput(output_info, parse_trajectory_out=True, verbose=False)
            # return df and output information
            return df, class_info
        else:
            raise TypeError("Output type does not exist!")


    # write a function that produces string for storing transition
    @staticmethod
    def generateTransitionString(states, molecule_names = ["D", "A"]):
        stateA, stateB = states[0], states[1]
        nameA, nameB = molecule_names[0], molecule_names[1]
        return f"[{nameA}({stateA + 1}), {nameB}(0)] <--> [{nameA}(0), {nameB}({stateB + 1})]"


    # initialize output based on desired output parameters 
    def initOutput(self, output_length):

        # (1) define QM states we are interested in (0-indexed), i.e. (S_0^A , S_{stateB + 1}^B) <--> (S_{stateA + 1}^A, S_0^B)
        self.transitions = self.quant_info[0]["transitions"]

        # TODO : might also want to add DataFrame for the direct QM (DFT/TDDFT) outputs 

        # (2) which trajectory-ensemble outputs are we interested in:

        # (2.1) classical MD output parameters:
        columns_class = [key for key, value in self.class_info[0].items() if isinstance(value, bool) and value]
        if not columns_class:
            self.output_class = pd.DataFrame()
        else:
            self.output_class = pd.DataFrame(index = range(output_length), columns = ["time"] + columns_class)
            
        
        # (2.2) quantum output parameters (output the same outputs for every transition in self.transitions)
        # NOTE : since states are 0-indexed, 0 actually corresponds to the 1st excited state of molecule A/B, 1 to the
        # 2nd excited state of molecule A/B etc.
        self.transition_names = [self.generateTransitionString(states, self.molecule_names) for states in self.transitions]
        self.quant_info[0].pop("transitions")
        columns_per_transitions = [key for key, value in self.quant_info[0].items() if isinstance(value, bool) and value]
        # get columns for each transition
        columns_per_transitions = []
        # initialize columns for Coulomb coupling
        if self.quant_info[0]["coupling"]:
            columns_per_transitions += ['coupling cJ', 'coupling cK', 'coupling V_C']
        # initialize columns for excitaion energies
        if self.quant_info[1]["excited_energies"]:
            columns_per_transitions += [f'energy {self.molecule_names[0]}', f'energy {self.molecule_names[1]}']

        # TODO : add more as desired later
        
        if not columns_per_transitions:
            self.output_quant = pd.DataFrame()
        else:
            columns_quant = pd.MultiIndex.from_tuples(
                [("time", "")] +
                [(transition_name, value_name) for transition_name in self.transition_names for value_name in columns_per_transitions]
            )
            self.output_quant = pd.DataFrame(index = range(output_length), columns = columns_quant)

        print("*** Intialization of output done!")
        

    # TODO : write simulation data into the header
    @staticmethod
    def writeOutputFiles(data_frame, file_name, write_meta_data = True):
        # TODO : write meta data into header
        # store DateFrame (classical or quantum) with meta data header (optional)
        if not data_frame.empty:
            with open(file_name, "w") as f:
                # optional: write meta data
                if write_meta_data:
                    pass
                # write output
                data_frame.to_csv(f, sep = "\t", index=False)

    
    # initialize molecules from params file
    def initMolecules(self, file):
        self.molecules, self.molecule_names = self.parseMolecules(file)
        self.defined_molecules = True                               
            

    # get MDAnalysis object of specified residues at specified time slice
    def getChromophoreSnapshot(self, idx, molecule, molecule_name, conversion = None, cap = True):
        # (1) set time step
        self.trajectory_u.trajectory[idx]
        # (2) get positions of all residues specified in residue_ids
        for id in molecule:
            molecule_u = self.trajectory_u.select_atoms(f'resid {id}')
            # make sure selected residue name equals desired molecule_name
            selected_name = np.unique(self.trajectory_u.select_atoms(f'resid {id}').resnames)[0]
            assert(selected_name == molecule_name)
        # (3) need to cap residues with hydrogens (O3' and OP1)
        # TODO : might want to make this more general for other dyes
        molecule_u = self.capResiduesH(molecule_u) if cap else molecule_u
        # (4) define instance of Chromophore class 
        chromophore = structure.Chromophore(molecule_u)
        # (5) convert to other input format for processing of trajectory
        chromophore_conv = self.convertChromophore(chromophore, conversion) if conversion else None

        return chromophore, chromophore_conv
        
    # converts Chromophore instance into desired format for trajectory processing
    # TODO : might want to add this to Chromophore class
    def convertChromophore(self, chromophore, conversion = 'pyscf'):
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
                # add to output dict
                self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in coupling_out.keys()]] = list(coupling_out.values())

            # (b) get excitation energies
            if self.quant_info[0]["excited_energies"]:
                # get excited state energies based on QM (DFT/TDDFT) output
                energies_out = qm.getExcEnergies(output_qm['exc'], states, molecule_names=self.molecule_names, excitation_energy_type=self.quant_info[1]['excited_energies'])
                # add to output dict
                self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in energies_out.keys()]] = list(energies_out.values())


    # TODO : this function needs to be updated a lot and more functionalities implemeted
    def analyzeSnapshotClassical(self, time_idx):

         # (0) time (ps)
        self.output_class.loc[time_idx, "time"] = time_idx * self.dt

        # (1) compute distance metric:
        # TODO : add an actual function here and not just some kind of dummy
        if self.class_info[0]["distance"]:
            self.output_class.loc[time_idx, "distance"] = 4
        
                
    # analyze trajectory based on specific molecules of interest
    def loopTrajectory(self):

 
        # (1) time range of interest: time_slice = [idx_start, idx_end]
        if self.time_slice is None:                                             # study the whole trajectory
            self.time_slice = [0, self.num_frames - 1]
        else:                                                                   # study specified time-slice 
            pass

        print(f'*** We loop through {self.time_slice[1] + 1 - self.time_slice[0]} frames for the trajectory analysis!')


        # (2) check whether molecules have been defined and initialized
        if not self.defined_molecules:
            raise AttributeError("Molecules to study have not been defined!")
        
        # (3) initialize output DataFrames
        self.initOutput(self.time_slice[1]  - self.time_slice[0]) 


        # (3) analyze trajectory
        for idx in range(self.time_slice[0], self.time_slice[1] + 1):

            start_time = time.time()
            print(f"*** Running Time Step {idx} ...")

            # (1) get chromophores of interest 
            self.chromophores = []
            self.chromophores_conv = []
            for i, molecule in enumerate(self.molecules):
                chromophore, chromophore_conv = self.getChromophoreSnapshot(idx, molecule, self.molecule_names[i], conversion = 'pyscf')
                self.chromophores.append(chromophore)
                self.chromophores_conv.append(chromophore_conv)


            # (2) analyze with respect to QM quantities of interest
            # TODO : only execute this when we have quantum quantities to analyze
            # (2.1) run QM calculation
            output_qm = qm.doQM_gpu(self.chromophores_conv, self.qm_outs, verbosity=0)
            # (2.2) post-processing of QM output
            self.analyzeSnapshotQuantum(idx, output_qm)

            # (3) analyze with respect to classical quantities of interest
            # TODO : only do the following if we have classical aspects to study
            # self.analyzeSnapshotClassical(idx)
            
            # (4) take time per time step
            end_time = time.time()
            print(f"Elapsed time for step {idx}: {end_time- start_time} seconds")


        # (4) write output files
        # (4.1) quantum output
        self.writeOutputFiles(self.output_quant, self.quant_info[2])
        # (4.2) classical output
        self.writeOutputFiles(self.output_class, self.class_info[2])



    # get disatnce between chromophores
    # TODO : add various distance measure to compare them
    def getDistance(self, chromophore_1, chromophore_2):
        com_1, com_2 = chromophore_1.com, chromophore_2.com
        distance = np.linalg.norm(com_1 - com_2)
        return distance


    # TODO : might want to make this more general for atom types to cap etc.
    # especially when we want to move to differen molecules/chromophores
    # TODO : might want to link this to the type of molecule that is under study here, e.g. by adding molecule_name and referring to some bib file
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



    # convert *.nc file to NetCDF3 and overwrite it
    # TODO : might want to store convert_traj.sh in more global directory
    def convertTrajectory(self):

        # locate shell script for trajectory conversion
        convert_traj_script = os.path.join(config.PROJECT_HOME, 'bin', 'convert_traj.sh')

        # run shell script
        cpptraj_command = f"bash {convert_traj_script} {self.prmtop} {self.nc}"
        subprocess.run(cpptraj_command, shell = True, stdout = subprocess.DEVNULL)


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
    
    

