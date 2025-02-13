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
import pickle

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


def main(molecules, time_steps):

    # store couplings
    cJs, cKs = [], []
    # store excitation energies
    excs_A, excs_B = [], []
    
    startT = time.time()
    for t in range(time_steps):
        print(f"\n Running Time Step {t}...", flush = True)
        start_time = time.time()

        # run molecules on different GPUs in parallel
        procs, mols = [], []
        for i, molecule_id in enumerate(molecules):
            # create pyscf input for subprocess and store in cache
            dump(getMol(molecule_id, t), f"input_{molecule_id}.joblib")
            # run subprocess
            #procs.append(run_dft_tddft(molecule_id, gpu_id = i))                

        # wait for both subprocesses to finish and capture their outputs
        outputs = []
        for i, molecule_id in enumerate(molecules):
            out, _= procs[i].communicate()
            outputs.append(out)

        # load and store relevant data from output of subprocesses
        exc, tdms, mols = [], [], []
        print('test')
        for i, molecule_id in enumerate(molecules):
            # array-type data
            data = np.load(io.BytesIO(outputs[i]))
            print(data["exc_energies"], data["tdms"])
            exc.append(data["exc_energies"])
            tdms.append(data["tdms"])
            # buffer = io.BytesIO(outputs[i])
            # exc_, tdm_ = pickle.load(buffer)
            # exc.append(exc_)
            # tdms.append(tdm_)
            # pyscf mol object
            #mols.append(load(f"mol_{molecule_id}.joblib"))

        # clean subprocess cache 
        utils.cleanCache()
        
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

