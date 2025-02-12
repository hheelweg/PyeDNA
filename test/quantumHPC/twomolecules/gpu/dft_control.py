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

    # (2) convert to pyscf mol object
    mol = gto.M(atom = chromophore_conv,
                basis = '6-31g',
                charge = 0,
                spin = 0)
    return mol, chromophore_conv


# NOTE : function that calls python ssubprocess to perform DFT/TDDFT on individual GPUs
def run_dft_tddft(molecule, time_idx, gpu_id, do_tddft):
    """Launch a DFT/TDDFT calculation on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign GPU

    cmd = f"python DFT_gpu.py {molecule} {time_idx}"
    if do_tddft:
        cmd += " --do-tddft"

    process = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        

    return process


def main(molecules, time_steps, do_tddft):

    # store couplings
    cJs, cKs = [], []
    
    startT = time.time()
    for t in range(time_steps):
        print(f"\n Running Time Step {t}...", flush = True)
        start_time = time.time()

        # TODO : do this as a for loop over molecules

        # run molecules on different GPUs in parallel
        procs, mols = [], []
        for i, molecule_id in enumerate(molecules):
             
            conv = getMol(molecule_id, t)                                                               # create pyscf input for subprocess 
            dump(conv, f"input_{molecule_id}.joblib")
            procs.append(run_dft_tddft(molecule_id, t, gpu_id = i, do_tddft=do_tddft))                  # run processes

        # wait for both processes to finish and capture their outputs
        outputs = []
        for i, molecule_id in enumerate(molecules):
            out, _ = procs[i].communicate()
            outputs.append(out)

        
        # load and store relevant data from outputs
        exc, tdms, mols = [], [], []
        for i, molecule_id in enumerate(molecules):
            data = np.load(io.BytesIO(outputs[i]))
            exc.append(data["exc_energies"])
            tdms.append(data["tdms"])
            print('testtttttttt')
            mol = load(f"mol_{molecule_id}.joblib")
            mols.append(mol)
            os.remove(f"mol_{molecule_id}.joblib")
        
        # debug output of DFT/TDDFT
        print(exc[0], exc[1])
        print(tdms[0].shape, tdms[1].shape)

        # compute coupling information
        cJ, cK = quantumTools.getV(mols[0], mols[1], tdms[0], tdms[1], coupling_type='both')
        print('cJ', cJ)
        cJs.append(cJ)
        cKs.append(cK)
        print(cJs)


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
    parser.add_argument("--do-tddft", action="store_true", help="Enable TDDFT calculation")     # boolean : run TDDFT
    args = parser.parse_args()

    # run main
    molecules = [args.molecule_1_id, args.molecule_2_id]
    main(molecules, args.time_idx, args.do_tddft)

