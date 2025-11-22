import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
import re
import time
import pandas as pd
import warnings

# from current package
from . import structure
from . import quanttools as qm
from . import geomtools as geom
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
        self.dt = self.md_params["dt"]                                      # in ps (MD time step)
        self.traj_dt = self.md_params["prod_ntwx"] * self.dt                # in ps (trajectory time step)
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
                        'eq1_ntpr'      :       5000,
                        'eq2_ntpr'      :       5000,
                        'eq1_ntwx'      :       5000,
                        'eq2_ntwx'      :       5000,
                        'eq1_ntwr'      :       5000,
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

    def __init__(self, MDsim, trajectory_data, traj_params_file = 'traj.params', qm_params_file = 'qm.params'):

        self.prmtop = trajectory_data[0]                                # load *.prmtop
        self.nc = trajectory_data[1]                                    # load *.nc from Amber MD simulation
        self.out = trajectory_data[2]                                   # load *.out file
        # make sure *.nc file is NetCDF3 (as required for MDAnalysis) and not NetCDF4 (as created by Amber)
        # self.convertTrajectory()
        # print('Trajectory file succesfully converted from NetCDF4 (Amber) to NetCDF3 (MDAnalysis)!')

        # load MDSimulation object which contains all information
        warnings.filterwarnings("ignore", message="Reader has no dt information")
        self.MD = MDsim                                                
        self.dt = self.MD.traj_dt

        # create MDAnalysis object for trajectory
        self.trajectory_u = mda.Universe(self.prmtop, self.nc)
        self.num_frames = self.trajectory_u.trajectory.n_frames         # number of frames in trajectory
        
        # parse output information for QM and MD simulations
        self.qm_outs, self.quant_info, self.class_info, self.time_slice = self.parseParameters(traj_params_file, parse_trajectory_out=True)
        # decide whether we perform quantum-mechanical and/or classical analysis
        self.do_quantum = bool(self.quant_info[0])
        self.do_classical = bool(self.class_info[0])
        # parse details on QM (DFT/TDDFT) calculations
        self.settings_dft, self.settings_tddft = self.setQMSettings(qm_params_file)
        print('Settings for DFT: ', self.settings_dft, flush=True)
        print('Settings for TDDFT: ', self.settings_tddft, flush=True)

        self.defined_molecules = False                                  # flag to track whether molecules have been defined
        # TODO : maybe delete this
        self.do_mulliken = False


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
            "singlet": True,
            "TDA": True,
            "gpu": True,
            "do_tddft": True,
            "optimize_cap": False
        }

        # read in user parameters from file
        user_params = fp.readParams(file)

        # update default parameters
        qm_settings.update(user_params)

        # split into dictionaries for keys related to DFT and TDDFT
        settings_dft = {key: qm_settings[key] for key in ["basis", "xc", "density_fit", "charge", "spin", "scf_cycles", "verbosity"]}
        settings_tddft = {key: qm_settings[key] for key in ["state_ids", "TDA", "do_tddft", "singlet"]}

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
                "orbit_enrgs" : False,
                "mol" :         True,
                "tdm" :         True,
                "dip" :         False,
                "osc" :         True,
                "idx" :         True,
                "mull_pops" :   False,
                "mull_chrgs" :  False,
                "OPA" :         False,
                "distance" :    False,
                'axis_angle' :  False,
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


        # (1) QM (DFT/TDDFT) outputs (NOTE : only boolean)
        qm_outs = {key: out.get(key) for key in ["exc", "mol", "tdm", "tdm_inter","mf", "occ", "virt", "orbit_enrgs", "dip", "osc", "idx", "mull_pops", "mull_chrgs", "OPA"]}               

        # (2) trajectory-based outputs per time steps
        qm_options =    [
                        "transitions",                                                                                          # transitions
                        "coupling", "coupling_type", "excited_energies", "dipole_moments", "osc_strengths",                     # quantities per transition
                        "abs_spec", "orbit_energies", "mulliken", "popanalysis"                                                 # quantities per molecule (transitions = None)
                        ]

        post_qm = {key: out.get(key) for key in qm_options}           
        qm_flags = {key: value for key, value in post_qm.items() if isinstance(value, bool) and value}                          # NOTE : only bool/True param
        # specify name of output file
        qm_out_file = out["file_qm"]


        # checkpoints: manually check if flags in out match with qm_flags:
        # TODO : maybe there is a better way to do this?
        qm_outs['exc'] = True if post_qm["abs_spec"] else qm_outs['exc']
        qm_outs['osc'] = True if post_qm["abs_spec"] else qm_outs['osc']
        qm_outs['mol'] = True if post_qm["orbit_energies"] else qm_outs['mol']
        qm_outs['orbit_enrgs'] = True if post_qm["orbit_energies"] else qm_outs['orbit_enrgs']
        qm_outs['exc'] = True if post_qm["excited_energies"] else qm_outs['exc']
        qm_outs['mull_pops'] = True if post_qm["mulliken"] else qm_outs['mull_pops']
        qm_outs['mull_chrgs'] = True if post_qm["mulliken"] else qm_outs['mull_chrgs']
        qm_outs['OPA'] = True if post_qm["popanalysis"] else qm_outs['OPA']
        qm_outs['dip'] = True if post_qm["dipole_moments"] else qm_outs['dip']
        qm_outs['osc'] = True if post_qm["osc_strengths"] else qm_outs['osc']
        qm_outs['mol'] = True if post_qm["coupling"] else qm_outs['mol']
        qm_outs['tdm'] = True if post_qm["coupling"] else qm_outs['tdm']
        # NOTE : only for intramolecular transfer studies, turn to True
        qm_outs['tdm_inter'] = False

        if "transitions" in out:
            qm_flags.update({"transitions": post_qm["transitions"]})

        # for each flag we either set specified methods_type or default
        qm_methods = {
                        key: out[f"{key}_type"] if f"{key}_type" in out else post_qm.get(f"{key}_type", "default")
                        for key in qm_flags
                        if isinstance(qm_flags[key], bool)
                    }

        # (3) classical parameters and methods
        class_options = [
                        "distance", "axis_angle"
                        ]

        post_class = {key: out.get(key) for key in class_options}# if key in out}                                                              
        class_flags = {key: value for key, value in post_class.items() if isinstance(value, bool) and value}                    # NOTE : only bool/True param
        # specify name of output file
        class_out_file = out["file_class"]
        # for each flag we either set specified methods_type or default
        class_methods = {
                            key: out[f"{key}_type"] if f"{key}_type" in out else post_class.get(f"{key}_type", "default")
                            for key in class_flags
                            if isinstance(class_flags[key], bool)
                        }
        

        if parse_trajectory_out:
            if verbose:
                # print parsed output for trajectory analysis
                print(" *** Parsed Output for Trajectory Analysis:")
                print(f"(1) classical parameters to evaluate at each time step: {', '.join(class_flags.keys())}")
                if "transitions" in qm_flags and qm_flags['transitions'] is not None:
                    print(f"(2) we study the following state transitions [stateA, stateB]: {', '.join(str(transition) for transition in qm_flags['transitions'])}")
                print(f"(2) quantum parameters to evaluate at each time step: {', '.join(key for key, value in qm_flags.items() if isinstance(value, bool))}")
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
                "name_1" :          None,
                "name_2" :          None,
                "constituents_1":   None,
                "constituents_2":   None,
                "charges":          None,
                "fragmentation":    None,
        }

        # read user parameters for molecules
        user_mols = fp.readParams(file)

        # update default settings
        mols.update(user_mols) 

        # store molecule IDs
        molecules = [value for key, value in mols.items() if key.startswith("molecule_") and value is not None]
        # store molecule names
        molecule_names = [value for key, value in mols.items() if key.startswith("name_") and value is not None]
        # store constituents of each molecules
        molecule_consituents = [value for key, value in mols.items() if key.startswith("constituents_") and value is not None]
        # store charges of each molecules
        molecule_charges = [value for key, value in mols.items() if key.startswith("charge_") and value is not None]
        # are we doing a fragment analysis for the specific molecule?
        molecule_do_fragments = [value for key, value in mols.items() if key.startswith("do_fragments_") and value is not None]

        # checkpoint
        assert(len(molecule_names) == len(molecules))
        if len(molecules) == 0:
            raise Warning("Need to specify at least one molecule in mols.params")
        elif len(molecules) > 2:
            raise NotImplementedError("More than 2 molecules (currently) not implemented!")

        return molecules, molecule_names, molecule_consituents, molecule_charges, molecule_do_fragments

    # read and parse DataFrame trajectory analysis output
    @staticmethod
    def readOutputFiles(file, output_type, output_info, molecule_names = ["D", "A"]):

        # (1) read file and parse output info 

        # (1.1) DataFrame with quantum information
        output = {}
        if output_type == 'quantum':
            df = pd.read_csv(file, sep='\t', header=[0,1])
            df.columns = [(col[0] if col[0] == "time" else col) for col in df.columns]
            # parse output information contained within data_frame
            _, qm_info, _, _  = Trajectory.parseParameters(output_info, parse_trajectory_out=True, verbose=False)
            # store df and output information
            output["df"] = df 
            output["qm_info"] = qm_info
            # (a) study quantities resolved by transition
            if qm_info[0]["transitions"]:
                # get names of the transitions under study
                transition_dict = {}
                for states in qm_info[0]["transitions"]:
                    key = Trajectory.generateTransitionString(states, molecule_names=molecule_names)
                    transition_dict[str(states)] = key
                # store transition dictionary
                output["transition_dict"] = transition_dict

        # (1.2) DataFrame with classical information
        elif output_type == 'classical':
            df = pd.read_csv(file, sep='\t', header=0)
            # parse output information contained within data_frame
            _, _, class_info, _  = Trajectory.parseParameters(output_info, parse_trajectory_out=True, verbose=False)
            # store df and output information
            output["df"] = df
            output["class_info"] = class_info
        
        else:
            raise TypeError("Output type does not exist!")
        
        return output


    # write a function that produces string for storing transition
    @staticmethod
    def generateTransitionString(states, molecule_names = ["D", "A"]):

        # check that states and molecule_names have same length
        assert(len(states) == len(molecule_names))

        # Case 1 (base case) : consider intermolecular transitions
        if len(states) == 2:
            # (a) consider transitions involving the largest oscillator strengths
            if states == ['strongest', 'strongest']:
                stateA, stateB = 's', 's'
            # (b) custom states
            else:
                stateA, stateB = states[0] + 1, states[1] + 1
            nameA, nameB = molecule_names[0], molecule_names[1]
            return f"[{nameA}({stateA}), {nameB}(0)] <--> [{nameA}(0), {nameB}({stateB})]"
        # Case 2 : consider intramolecular transitions
        if len(states) == 1:
            # (a) consider transitions involving the largest oscillator strengths
            if states == ['strongest']:
                state = 's'
            # (b) custom states
            else:
                state = states[0] + 1
            name = molecule_names[0]
            return f"[{name}(0)] <--> [{name}({state})]"



    # initialize output based on desired output parameters 
    def initOutput(self, output_length):


        # ##############      (1)   Classical analysis                       ###############

        if self.do_classical:
            # (1.1) classical MD output parameters:
            columns_class = [key for key, value in self.class_info[0].items() if isinstance(value, bool) and value]
            if not columns_class:
                self.output_class = pd.DataFrame()
            else:
                self.output_class = pd.DataFrame(index = range(output_length), columns = ["time"] + columns_class)

        
        # ##############      (2)   Quantum-mechanical analysis               ###############

        if self.do_quantum:

            # (2.1) define QM states we are interested in (0-indexed), i.e. (S_0^A , S_{stateB + 1}^B) <--> (S_{stateA + 1}^A, S_0^B)
            self.transitions = self.quant_info[0]["transitions"]
            # check that length of each transition in self.transitions agrees with number of molecules
            if self.transitions is not None:
                for transition in self.transitions:
                    assert(len(transition) == self.num_molecules)
        
            # (2.2) quantum output parameters if we want to get resolution per transition in self.transitions
            if self.transitions is not None:
                # (output the same quantities for every transition in self.transitions)
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

                # initialize columns for excitation energies
                if self.quant_info[0]["excited_energies"]:
                    if len(self.molecule_names) == 2:
                        columns_per_transitions += [f'energy {self.molecule_names[0]}', f'energy {self.molecule_names[1]}']
                    elif len(self.molecule_names) == 1:
                        columns_per_transitions += [f'energy {self.molecule_names[0]}']

                # intialize columns for oscillator strengths
                if self.quant_info[0]["osc_strengths"]:
                    if len(self.molecule_names) == 2:
                        columns_per_transitions += [f'osc_strength {self.molecule_names[0]}', f'osc_strength {self.molecule_names[1]}']
                    elif len(self.molecule_names) == 1:
                        columns_per_transitions += [f'osc_strength {self.molecule_names[0]}']

                # initialize columns for transition dipole moments
                if self.quant_info[0]["dipole_moments"]:
                    if len(self.molecule_names) == 2:
                        columns_per_transitions += [f'dip_moment {self.molecule_names[0]}', f'dip_moment {self.molecule_names[1]}']
                    elif len(self.molecule_names) == 1:
                        columns_per_transitions += [f'dip_moment {self.molecule_names[0]}']

                # TODO : add more as desired
                
                # construct output DataFrame
                if not columns_per_transitions:
                    self.output_quant = pd.DataFrame()
                else:
                    columns_quant = pd.MultiIndex.from_tuples(
                        [("time", "")] +
                        [(transition_name, value_name) for transition_name in self.transition_names for value_name in columns_per_transitions]
                    )
                    self.output_quant = pd.DataFrame(index = range(output_length), columns = columns_quant)
            
            # if we want to print direct DFT/TDDFT output per each time step (e.g. if abs_spec = True), do this here. We output these quantities
            # for every molecule specified in self.molecules.
            # NOTE : need to specify transitions = None for this!
            elif self.transitions is None:

                self.quant_info[0].pop("transitions")

                columns_per_molecule = []

                # initialize columns for plotting of absorption spectrum
                if "abs_spec" in self.quant_info[0]:
                    # which direct outputs of the TDDFT calculations do we need?
                    which_outs = ['exc', 'osc']
                    columns_per_molecule += [f"{which_out} {state_id}" for which_out in which_outs for state_id in self.settings_tddft["state_ids"]]

                # initialize columns for orbital energies
                if "orbit_energies" in self.quant_info[0]:
                    orbital_types = ['occ', 'virt']
                    columns_per_molecule += [f"{orbital_type}" for orbital_type in orbital_types]

                # initialize columns for excited energies
                if "excited_energies" in self.quant_info[0]:
                    columns_per_molecule += [f"exc_enrgs ({'singlets' if self.settings_tddft['singlet'] else 'triplets'}): {' ,'.join(str(state_id) for state_id in self.settings_tddft['state_ids'])}"]

                # initialize columns for Mulliken analysis of excited state populations
                # TODO : maybe parse information for Mulliken analysis in different functions
                if "mulliken" in self.quant_info[0]:

                    # parse Mulliken information
                    self.do_mulliken = True
                    self.fragment_type = self.quant_info[1]["mulliken"][0]
                    self.fragments = self.quant_info[1]["mulliken"][1]
                    if self.fragment_type == "molecule":
                        self.fragment_names = self.fragments
                    elif self.fragment_type == "atom_group":
                        self.fragment_names = [f'group {i}' for i in range(len(self.fragments))]

                    columns_per_molecule += [f"mulliken (state {state_id}) {fragment_name}" for state_id in self.settings_tddft["state_ids"] for fragment_name in self.fragment_names]
                    
                # TODO : maybe parse information for orbital population analysis (OPA) in different functions
                if "popanalysis" in self.quant_info[0]:

                    # parse Mulliken information
                    self.do_mulliken = True
                    self.fragment_type = self.quant_info[1]["popanalysis"][0]
                    self.fragments = self.quant_info[1]["popanalysis"][1]
                    if self.fragment_type == "molecule":
                        self.fragment_names = self.fragments
                    elif self.fragment_type == "atom_group":
                        self.fragment_names = [f'group {i}' for i in range(len(self.fragments))]

                    columns_per_molecule += [f"popanalysis (state {state_id})" for state_id in self.settings_tddft["state_ids"]]


                # construct output DataFrame
                if not columns_per_molecule:
                    self.output_quant = pd.DataFrame()
                else:
                    columns_quant = pd.MultiIndex.from_tuples(
                        [("time", "")] +
                        [(molecule_name, value_name) for molecule_name in self.molecule_names for value_name in columns_per_molecule]
                    )
                    self.output_quant = pd.DataFrame(index = range(output_length), columns = columns_quant)     


        

    # TODO : write simulation data into the header
    @staticmethod
    def writeOutputFiles(data_frame, file_name, write_meta_data = True, dir = None):
        # TODO : write meta data into header
        file_name = dir + file_name if dir else file_name
        # store DateFrame (classical or quantum) with meta data header (optional)
        if not data_frame.empty:
            with open(file_name, "w") as f:
                # optional: write meta data
                if write_meta_data:
                    pass
                # write output
                data_frame.to_csv(f, sep = "\t", index=False)
                

    
    # initialize molecules from params file
    def initMolecules(self, file, dye_path = None):

        # parse information of molecules attached and their consitutent builidng blocks
        self.molecules, self.molecule_names, self.molecule_constituents, self.molecule_charges, self.molecule_do_fragments = self.parseMolecules(file)
        self.defined_molecules = True 
        self.num_molecules = len(self.molecules)

        # find information of unique residues in list
        unique_dyes = np.unique(np.concatenate(self.molecule_constituents))
        if dye_path is None: 
            dye_base_dir = os.getenv("DYE_DIR")
        else:
            dye_base_dir = dye_path
        self.molecule_information = dict()

        for unique_dye in unique_dyes:

            dye_dir = os.path.join(dye_base_dir, unique_dye)
            self.molecule_information[unique_dye] = dict()
            # file names that we need to parse
            dye_atoms_file = os.path.join(dye_dir, "dye.info")
            capped_positions_file = os.path.join(dye_dir, "dye_cap.info")
            symmetry_info_file = os.path.join(dye_dir, "symm.info")

            # (1) read information about dye atoms
            with open(dye_atoms_file, 'r') as file:
                atom_list = [line.strip() for line in file if line.strip()]  
            self.molecule_information[unique_dye]["dye_atoms"] = " ".join(atom_list)

            # (2) read information about which positions act as the cap
            with open(capped_positions_file, 'r') as file:
                capped_list = [line.strip() for line in file if line.strip()]  
            self.molecule_information[unique_dye]["capped_atoms"] = " ".join(capped_list)

            # (3) read symmetry info
            self.molecule_information[unique_dye]["symm_info"] = fp.readParams(symmetry_info_file)
        
   
    # get MDAnalysis object of specified residues at specified time slice
    # TODO : delete this?
    def getChromophoreSnapshotOld(self, idx, molecule, molecule_name, conversion = None, cap = True):
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
        #molecule_u = self.capResiduesH(molecule_u) if cap else molecule_u
        molecule_u = self.capResiduesHNew(molecule_u) if cap else molecule_u
        # (4) define instance of Chromophore class 
        chromophore = structure.Chromophore(molecule_u)
        # (5) convert to other input format for processing of trajectory
        chromophore_conv = self.convertChromophore(chromophore, conversion) if conversion else None

        return chromophore, chromophore_conv
    

    # get MDAnalysis object of specified residues at specified time slice
    def getChromophoreSnapshot(self, molecule, molecule_constituents, 
                               fragments = None, enforce_symmetry = False, conversion = None, cap = True):

        # (1) (optional) load fragment information
        if fragments is not None:
            fragment_type = fragments[0]
            fragment_identifiers = fragments[1]
        else:
            fragment_type, fragment_identifiers = None, None

        # # # NOTE : only do this when trying to export .pdb file of the whole DNA
        # res_max = 46
        # residue_sel_string = f"resid 1:{res_max}"
        # dna_u = self.trajectory_u.select_atoms(residue_sel_string)
        # dna_u.atoms.write(f"dna_snapshot.pdb")

        # (2) get positions of all residues (constituents) specified in residue_ids
        molecules_u = []
        for i, id in enumerate(molecule):
            # select correct residue
            molecule_u = self.trajectory_u.select_atoms(f'resid {id}')
            # get information of dye/residue
            dye_atoms = self.molecule_information[molecule_constituents[i]]["dye_atoms"]
            capped_atoms = self.molecule_information[molecule_constituents[i]]["capped_atoms"]
             # get positions we want to cap with hydrogens
            capped_positions = molecule_u.atoms.select_atoms(f'name {capped_atoms}').positions
            molecule_u = molecule_u.atoms.select_atoms(f'name {dye_atoms}')
            # add hydrogen caps
            if cap:
                molecule_u = self.capWithHydrogens(molecule_u, capped_positions=capped_positions)

            # # NOTE : only do this when trying to export .pdb file of chromophores
            # molecule_u.atoms.write(f"snap_molecule_{id}.pdb")

            molecules_u.append(molecule_u)
            # make sure selected residue name equals desired molecule_name
            selected_name = np.unique(self.trajectory_u.select_atoms(f'resid {id}').resnames)[0]
            assert(selected_name == molecule_constituents[i])
        
        # (3) get fragment lengths in both fragments if fragments[0] = 'molecule'
        if fragment_type == 'molecule':
            fragments_length, fragment_names = [], []
            for i, constituent_name in enumerate(molecule_constituents):
                assert(fragment_identifiers[i] == constituent_name)
                fragment_names.append(fragment_identifiers[i])
                fragments_length.append(len(molecules_u[i].atoms.names))
        # TODO : implement this
        elif fragment_type == 'atom_group':
            pass
        elif fragment_type is None:
            pass
        
        
        # (4) check how many residues the molecule is composed of and allow for max of 2 
        if len(molecule) == 1:
            molecule_u = molecules_u[0]
            symmetry_info = self.molecule_information[molecule_constituents[0]]["symm_info"]
        elif len(molecule) == 2:
            molecule_u = mda.Merge(molecules_u[0].atoms, molecules_u[1].atoms)
        else:
            raise NotImplementedError('Only two neighboring residues currently implemented!')
        
        # (5) get fragment indices
        if fragment_type == 'molecule':
            fragment_indices = [[i for i in range(len(molecule_u.atoms[:fragments_length[0]]))], [i for i in range(len(molecule_u.atoms[:fragments_length[0]]), len(molecule_u.atoms))]]
        # TODO : implement this 
        elif fragment_type == 'atom_group':
            pass
        elif fragment_type is None:
            pass
        

        # (optional) enforce symmetry if they are composed of a single dye
        if enforce_symmetry and len(molecule) == 1:
            # shift center of geometry to (0,0,0) and align atoms in symmetry_axis with (0,0,1) vector
            if symmetry_info["point_group"] == 'Cs':
                molecule_u = geom.shiftAndAlign(molecule_u, symmetry_info["symmetry_axis"])
                molecule_u = geom.enforceSymmetry(molecule_u, symmetry_info["symmetry_axis"], symmetry_info["support_atom"])
            else:
                raise NotImplementedError('Only Cs point group symmetry implemented for DFT/TDDFT analysis')
        # (6) define instance of Chromophore class 
        chromophore = structure.Chromophore(molecule_u)

        # (7) convert to other input format for processing of trajectory
        chromophore_conv = self.convertChromophore(chromophore, conversion) if conversion else None

        # hash fragmentation info
        fragmentation_info = dict()
        fragmentation_info['fragment_indices'] = fragment_indices if fragments is not None else None
        fragmentation_info['fragment_names'] = fragment_names if fragments is not None else None

        print('TEST', fragmentation_info['fragment_indices'], fragmentation_info['fragment_names'], flush=True)

        return chromophore, chromophore_conv, fragmentation_info

        # if fragments is None:
        #     return chromophore, chromophore_conv
        # else:
        #     return chromophore, chromophore_conv, fragment_indices, fragment_names



    # converts Chromophore instance into desired format for trajectory processing
    # TODO ; might want to extend this to QChem input
    # TODO : might want to add this to Chromophore class
    @staticmethod
    def convertChromophore(chromophore, conversion = 'pyscf'):

        # can only convert to PySCF input
        if conversion not in ['pyscf']:
            raise NotImplementedError("Specify valid format to convert Chromophore object to.")
        
        # convert Chromophore object to PySCF input 
        if conversion == 'pyscf':

            xyz, names = chromophore.xyz, chromophore.names
            molecule_conv = []
            for i, coords in enumerate(xyz):
                # pyscf molecule input
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
        self.output_quant.loc[time_idx, ("time", "")] = (time_idx + 1) * self.dt

        # (1) loop over all specified transitions 
        if self.transitions is not None:
            for i, states in enumerate(self.transitions):


                # if we specify ['strongest', 'strongest'], then we consider the states with the largest oscillator strengths
                if states == ['strongest', 'strongest']:
                    states = [output_qm["idx"][0], output_qm["idx"][1]]
                elif states == ['strongest']:
                    states = [output_qm["idx"][0]]

                # (a) get Coulombic coupling information
                if self.quant_info[0]["coupling"]:
                    # compute coupling based on QM (DFT/TDDFT) output
                    # TODO : for intramolecular
                    if i == 0:
                        coupling_out = qm.getVCoulombic(output_qm['mol'], output_qm['tdm'],  [self.transitions[i], self.transitions[i+1]], coupling_type=self.quant_info[1]['coupling'])
                    if i == 1:
                        coupling_out = qm.getVCoulombic(output_qm['mol'], output_qm['tdm'],  [self.transitions[i-1], self.transitions[i]], coupling_type=self.quant_info[1]['coupling'])
                    # TODO : go back to this:
                    # TODO : for intermolecular
                    #coupling_out = qm.getVCoulombic(output_qm['mol'], output_qm['tdm'], states, coupling_type=self.quant_info[1]['coupling'])
                    # add to output df
                    self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in coupling_out.keys()]] = list(coupling_out.values())

                # (b) get excitation energies
                if self.quant_info[0]["excited_energies"]:
                    # get excited state energies based on QM (DFT/TDDFT) output
                    energies_out = qm.getExcEnergiesTransition(output_qm['exc'], states, molecule_names=self.molecule_names, excitation_energy_type=self.quant_info[1]['excited_energies'])
                    # add to output df
                    self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in energies_out.keys()]] = list(energies_out.values())
                
                # (c) get oscillator strengths
                if self.quant_info[0]["osc_strengths"]:
                    # get oscillator strengths based on QM (DFT/TDDFT) output
                    osc_out = qm.getOscillatorStrengths(output_qm['osc'], states, molecule_names=self.molecule_names,osc_strength_type=self.quant_info[1]['osc_strengths'])
                    # add to output df
                    self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in osc_out.keys()]] = list(osc_out.values())
                
                # (d) get transition dipoles
                if self.quant_info[0]["dipole_moments"]:
                    # get transition dipole moments based on QM (DFT/TDDFT) output
                    dipoles_out = qm.getTransitionDipoles(output_qm['dip'], states, molecule_names=self.molecule_names,dipole_moment_type=self.quant_info[1]['dipole_moments'])
                    # add to output df
                    self.output_quant.loc[time_idx, [(self.transition_names[i], key) for key in dipoles_out.keys()]] = list(dipoles_out.values())


        # (2) look at direct output quantities of QM (DFT/TDDFT) (if self.transitions = None)
        elif self.transitions is None:

            # (a) get quantities necessary to compute absorption spectrums
            if "abs_spec" in self.quant_info[0]:
                which_outs = ["exc", "osc"]
                # get desired TDDFT output
                tddft_out = qm.getTDDFToutput(output_qm, which_outs, self.settings_tddft["state_ids"], molecule_names = self.molecule_names)
                # add to output df
                for molecule_name in self.molecule_names:
                    for which_out in which_outs:
                        for state_id in self.settings_tddft["state_ids"]:
                            self.output_quant.loc[time_idx, (molecule_name, f"{which_out} {state_id}")] = tddft_out[f"{molecule_name} {which_out} {state_id}"]
            
            # (b) get orbital energies of occupied and virtual orbitals
            if "orbit_energies" in self.quant_info[0]:
                orbital_types = ["occ", "virt"]
                # get orbital energies from DFT output
                orbit_energies_out = qm.getOrbitalEnergies(output_qm, orbital_types, molecule_names = self.molecule_names)
                # add to output df
                for molecule_name in self.molecule_names:
                    for orbital_type in orbital_types:
                        self.output_quant.loc[time_idx, (molecule_name, f"{orbital_type}")] = orbit_energies_out[f"{molecule_name} {orbital_type}"]

            # (c) get excited state energies
            if "excited_energies" in self.quant_info[0]: 
                # get excited state energies from TDDFT output
                exc_energies_out = qm.getExcitedEnergies(output_qm, molecule_names = self.molecule_names)
                # add to output df
                for molecule_name in self.molecule_names:
                    self.output_quant.loc[time_idx, (molecule_name, f"exc_enrgs ({'singlets' if self.settings_tddft['singlet'] else 'triplets'}): {' ,'.join(str(state_id) for state_id in self.settings_tddft['state_ids'])}")] = exc_energies_out[molecule_name] 

            
            # (d) get Mulliken analysis on specified fragment
            if "mulliken" in self.quant_info[0]:
                # get Mulliken analysis on atom index group in self.chromophores_fragments
                mulliken_out = qm.getMullikenFragmentAnalysis(output_qm, self.settings_tddft['state_ids'], fragments=self.chromophores_fragments, fragment_names=self.chromophores_fragment_names, molecule_names=self.molecule_names)
                # add to output df
                for i, molecule_name in enumerate(self.molecule_names):
                    for state_id in self.settings_tddft['state_ids']:
                        # Mulliken analysis per molecule for each specified fragment
                        for fragment_name in self.chromophores_fragment_names[i]:
                            self.output_quant.loc[time_idx, (molecule_name, f"mulliken (state {state_id}) {fragment_name}")] = mulliken_out[f"{molecule_name} {state_id} {fragment_name}"]
            
            # (e) orbital population analysis (OPA) on specified fragment
            if "popanalysis" in self.quant_info[0]:
                # get orbital population analysis (OPA) on atom index group in self.chromophores_fragments
                # add to output df
                for i, molecule_name in enumerate(self.molecule_names):
                    for state_id in self.settings_tddft['state_ids']:
                        self.output_quant.loc[time_idx, (molecule_name, f"popanalysis (state {state_id})")] = output_qm['OPA'][i][state_id]
                        
            else:
                pass

    
    # TODO : this function needs to be updated a lot and more functionalities implemented
    def analyzeSnapshotClassical(self, time_idx):

         # (0) time (ps)
        self.output_class.loc[time_idx, "time"] = (time_idx + 1) * self.dt

        # (1) compute distance metric:
        if "distance" in self.class_info[0]:
            self.output_class.loc[time_idx, "distance"] = geom.getDistance(self.trajectory_u, self.class_info[1]["distance"])
        
        # (2) compute angle between two axes
        if "axis_angle" in self.class_info[0]:
            self.output_class.loc[time_idx, "axis_angle"] = geom.getAxisAngle(self.trajectory_u, self.class_info[1]["axis_angle"])
        
                
    # analyze trajectory based on specific molecules of interest
    def loopTrajectory(self, output_dir = None):

 
        # (1) time range of interest: time_slice = [idx_start, idx_end]
        if self.time_slice is None:                                             # study the whole trajectory
            self.time_slice = [0, self.num_frames - 1]
        else:                                                                   # study specified time-slice 
            pass

        print(f'*** Looping through {self.time_slice[1] + 1 - self.time_slice[0]} frames for the trajectory analysis!')


        # (2) check whether molecules have been defined and initialized
        if not self.defined_molecules:
            raise AttributeError("Molecules to study have not been defined!")
        
        # (3) initialize output DataFrames
        self.initOutput(self.time_slice[1]  - self.time_slice[0]) 

        print("*** Intialization of output done!")


        # (3) analyze trajectory
        for idx in range(self.time_slice[0], self.time_slice[1] + 1):

            start_time = time.time()

            print(f"*** Running Time Step {idx + 1} ...")

            # (0) set snapshot
            self.trajectory_u.trajectory[idx]

            # (1) get chromophores of interest 
            self.chromophores = []
            self.chromophores_conv = []

            # individual fragments per molecule; only necessary when doing a mulliken population analysis
            if self.do_quantum and self.do_mulliken:
                self.chromophores_fragments = [] if self.do_mulliken else None
                self.chromophores_fragment_names = [] if self.do_mulliken else None


            for i, molecule in enumerate(self.molecules):

                
                #chromophore, chromophore_conv = self.getChromophoreSnapshotOld(idx, molecule, self.molecule_names[i], conversion = 'pyscf')

                if self.do_quantum and self.do_mulliken and self.molecule_do_fragments[i]:
                    chromophore, chromophore_conv, fragmentation_info = self.getChromophoreSnapshot(
                                                                                molecule = molecule,
                                                                                molecule_constituents = self.molecule_constituents[i],
                                                                                fragments = [self.fragment_type, self.fragments],
                                                                                enforce_symmetry = False,
                                                                                conversion = 'pyscf'
                                                                                )
                    self.chromophores_fragments.append(fragmentation_info['fragment_indices'])
                    self.chromophores_fragment_names.append(fragmentation_info['fragment_names'])
                    
                else:
                    chromophore, chromophore_conv, _ = self.getChromophoreSnapshot(
                                                                                molecule = molecule,
                                                                                molecule_constituents = self.molecule_constituents[i],
                                                                                fragments = None, 
                                                                                enforce_symmetry = False,
                                                                                conversion = 'pyscf'
                                                                                )
                
                    # # TODO : this is only for debugging
                    # # optional : write test snapshot(s), typically we only need this for debugging
                    # chromophore.chromophore_u.atoms.write(f"snapshot_{i}.pdb")

                self.chromophores.append(chromophore)
                self.chromophores_conv.append(chromophore_conv)


            # (2) analyze with respect to QM quantities of interest
            if self.do_quantum:
                # (2.1) run QM calculation
                if self.do_mulliken:
                    output_qm = qm.doQM_gpu(self.chromophores_conv, self.qm_outs,
                                            do_fragments=self.molecule_do_fragments, 
                                            fragments=self.chromophores_fragments,
                                            charges=self.molecule_charges,
                                            verbosity = 3
                                            )
                else:
                    output_qm = qm.doQM_gpu(self.chromophores_conv, self.qm_outs,
                                            charges=self.molecule_charges, 
                                            verbosity = 3
                                            )
                # NOTE : set verbosity = 0 for production runs, and verbosity = 2 for debugging. 

                # (2.2) post-processing of QM output
                self.analyzeSnapshotQuantum(idx, output_qm)

            # (3) analyze with respect to classical quantities of interest
            if self.do_classical:
                self.analyzeSnapshotClassical(idx)
            
            # (4) take time per time step
            end_time = time.time()
            print(f"Elapsed time for step {idx + 1}: {end_time- start_time:.0f} seconds")


        print('*** Successfully looped through all trajectory snapshots!')
        
        
        # (4) write output files
        # (4.1) quantum output
        if self.do_quantum:
            self.writeOutputFiles(self.output_quant, self.quant_info[2], dir = output_dir)
        # (4.2) classical output
        if self.do_classical:
            self.writeOutputFiles(self.output_class, self.class_info[2], dir = output_dir)


        print('*** Finished writing oputput files!')



    # get disatnce between chromophores
    # TODO : add various distance measure to compare them
    def getDistance(self, chromophore_1, chromophore_2):
        com_1, com_2 = chromophore_1.com, chromophore_2.com
        distance = np.linalg.norm(com_1 - com_2)
        return distance

    # cap dye molecule with hydrogens at locations specified by capped_positions
    def capWithHydrogens(self, molecule, capped_positions):
         # (1) atom names, atom types, etc.
        max_H_idx = max([int(re.search(r'H(\d+)', name).group(1)) for name in molecule.atoms.names if re.match(r'H\d+', name)])
        new_atom_names = [f'H{max_H_idx + 1}', f'H{max_H_idx + 2}']
        new_atom_types = ['H', 'H']
        new_residue_name = np.unique(molecule.resnames)[0]
        new_residue_id = np.unique(molecule.resids)[0]
        new_elements = ['H', 'H']
        # (2) H positions
        new_atom_positions = np.array(capped_positions)
        # (3) initialize new hydrogen Universe
        Hs = mda.Universe.empty(n_atoms = len(capped_positions), trajectory=True)
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

        pass


    # TODO : might want to make this more general for atom types to cap etc.
    # especially when we want to move to differen molecules/chromophores
    # TODO : might want to link this to the type of molecule that is under study here, e.g. by adding molecule_name and referring to some bib file
    def capResiduesH(self, molecule, capped_atom_names = ["O3'", "OP1"]):

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
        for name in capped_atom_names:
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

    # NOTE : new function that is also removing the phosphate group from the dye molecule before capping with H's
    def capResiduesHNew(self, molecule):
        # (0) remove phosphate group first
        molecule = molecule.select_atoms("not name P OP1 OP2")
        # (1) now only cap O3' and O5'
        capped_atom_names = ["O3'", "O5'"]
        molecule = self.capResiduesH(molecule, capped_atom_names)
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
    
    

