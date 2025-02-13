import os
import argparse
import time
import subprocess
import torch
import json
import numpy as np
from pyscf import gto, lib
import io
import sys
from joblib import dump, load

# import custom modules
path_to_modules = '/home/hheelweg/Cy3Cy5/PyCY'
sys.path.append(path_to_modules)
import quantumTools, structure
import trajectory as traj
import const
import utils
import fileProcessing as fp

# Detect available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM allocation.")


# get pyscf mol object based on molecule index and time slice
# NOTE : this is just here for test purposes
def getMol(mol_idx, time_idx):
    MDsim = traj.MDSimulation([])                           # empty MDSimulation object

    path = '/home/hheelweg/Cy3Cy5/PyCY/test/prod/'          # specify relative path to MD ouput
    name_prmtop = 'dna_test.prmtop'
    name_nc = 'dna_test_prod.nc'                            
    name_out = 'dna_test_prod.out'
              

    data = [name_prmtop,name_nc, name_out]                  # trajectory data 
    test = traj.Trajectory(MDsim, path, data)               # initialize Trajectory object

    # (1) specify chromophore to perform DFT/TDDFT on
    molecule = [mol_idx]
    chromophore, chromophore_conv = test.getChromophoreSnapshot(time_idx, molecule, conversion = 'pyscf')

    return chromophore_conv


# NOTE : function that calls python ssubprocess to perform DFT/TDDFT on individual GPUs
def run_dft_tddft(molecule_id, gpu_id):
    """Launch a DFT/TDDFT calculation on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign GPU

    cmd = f"python DFT_gpu.py {molecule_id}"
    process = subprocess.Popen(cmd, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        

    return process

# do QM calculations on molecules for output quantities we are interested in (output_keys)
# TODO : this is written for GPU support only so far, but plan to extend this to CPU MPI
def doQM_gpu(molecules, time_idx, output_keys):

    # output dictionary
    out_dict = parseQMOutput('qm_out.params')
    print(out_dict, flush = True)
    output_keys = [key for key, value in out_dict.items() if value]
    output = {key: [] for key in output_keys}

    # run molecules on different GPUs in parallel
    procs = []
    for i, molecule_id in enumerate(molecules):
        # create pyscf input for subprocess and store in cache
        dump(getMol(molecule_id, time_idx), f"input_{molecule_id}.joblib")
        # run subprocess
        procs.append(run_dft_tddft(molecule_id, gpu_id = i))
    
    # wait for both subprocesses to finish
    for i, molecule_id in enumerate(molecules):
        procs[i].wait()

    # load and store relevant data from output of subprocesses
    # TODO : flexibilize this for quantities we are interested in
    exc, tdms, mols = [], [], []
    for i, molecule_id in enumerate(molecules):
        for key in output_keys:
            output[key].append(load(f"{key}_{molecule_id}.joblib"))

    # clean subprocess cache 
    utils.cleanCache()

    return output

# parse output information for QM calculations
def parseQMOutput(file):
    # output default parameters
    default_out = {
            "exc" : False,
            "mol" : True,
            "tdm" : True
    }
    # specify user parameters
    user_out = fp.readParams(file)

    # update default settings
    default_out.update(user_out)

    return default_out


def main(molecules, time_steps):

    # store couplings
    cJs, cKs = [], []
    # store excitation energies
    excs_A, excs_B = [], []

    # output quantities we are interested in
    output_keys = ['exc', 'tdm', 'mol']
    
    startT = time.time()
    for t in range(time_steps):
        print(f"\n Running Time Step {t}...", flush = True)
        start_time = time.time()

        # run QM on two molecules
        # TODO : only implemented for GPU support so far
        output = doQM_gpu(molecules, t, output_keys)

        # print output for debugging
        print(output['exc'])
        
        # # compute coupling information and excitation energies
        # stateA, stateB = 0, 0
        # cJ, cK = quantumTools.getV(mols[0], mols[1], tdms[0], tdms[1], stateA=stateA, stateB=stateB, coupling_type='both')
        # exc_A, exc_B = exc[0][stateA], exc[1][stateB]
        # print('cJ', cJ)
        # cJs.append(cJ)
        # cKs.append(cK)
        # excs_A.append(exc_A)
        # excs_B.append(exc_B)
        # print(excs_A, excs_B)


        end_time = time.time()  # End timing for this step
        elapsed_time = end_time - start_time
        print(f"Time Step {t} Completed in {elapsed_time:.2f} seconds", flush = True)



    endT = time.time()
    print(f"All DFT/TDDFT calculations completed in {endT -startT} sec!")




if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations")
    parser.add_argument("molecule_1_id", type=int, help="Molecule 1 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("molecule_2_id", type=int, help="Molecule 2 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("time_idx", type=int, help="Time index (integer)")                      # specifies time step upon we wish to analyze trajectory
    args = parser.parse_args()

    # run main
    molecules = [args.molecule_1_id, args.molecule_2_id]
    main(molecules, args.time_idx)

