import os
import argparse
import time
import subprocess
import torch
import json
import numpy as np
import sys

# Detect available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM allocation.")


# NOTE : old version calling subprocess routin
def run_dft_tddft(molecule, time_idx, gpu_id, do_tddft):
    """Launch a DFT/TDDFT calculation on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign GPU

    cmd = f"python DFT_gpu.py {molecule} {time_idx}"
    if do_tddft:
        cmd += " --do-tddft"

    process = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=sys.stderr, text=True )        

    return process


# # NOTE : (old) function that parallel executes 
def main(mol_1, mol_2, time_steps, do_tddft):
    
    startT = time.time()
    for t in range(time_steps):
        print(f"\n Running Time Step {t}...", flush = True)
        start_time = time.time()

        # run molecule_1 on GPU 0 and molecule_2 on GPU 1
        proc1 = run_dft_tddft(mol_1, t, gpu_id=0, do_tddft=do_tddft)
        proc2 = run_dft_tddft(mol_2, t, gpu_id=1, do_tddft=do_tddft)

        # wait for both processes to finish and capture their outputs
        output1, error1 = proc1.communicate()
        output2, error2 = proc2.communicate()
        print('output1', output1, flush = True)

        # read in inputs
        output1_json = json.loads(output1.strip())
        exc_energies_1 = np.array(output1_json["exc_energies"])
        tdms_1 = np.array(output1_json["tdms"])
        print(exc_energies_1)
        print(tdms_1.shape)


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
    main(args.molecule_1_id, args.molecule_2_id, args.time_idx, args.do_tddft)

