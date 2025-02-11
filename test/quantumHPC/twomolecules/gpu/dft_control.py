import os
import argparse
import time
import subprocess
import torch
import DFT_gpu
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Detect available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM allocation.")

# Set the multiprocessing method to "spawn" to avoid CUDA errors
mp.set_start_method('spawn', force=True)

# NOTE : old version calling subprocess routin
# def run_dft_tddft(molecule, time_idx, gpu_id, do_tddft):
#     """Launch a DFT/TDDFT calculation on a specific GPU."""
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign GPU

#     cmd = f"python DFT_gpu.py {molecule} {time_idx}"
#     if do_tddft:
#         cmd += " --do-tddft"
    
#     result = subprocess.run(cmd, shell=True, env=env, capture_output=True)
#     output = result.stdout.strip().split("\n")                                  # Split into lines
#     exc_energies = [float(x) for x in output[0].split()]  
#     tdms = [float(x) for x in output[1].split()]          

#     return exc_energies, tdms


def run_dft_tddft(molecule, time_idx, gpu_id, do_tddft):

    # Set GPU device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running DFT/TDDFT on molecule {molecule} at time step {time_idx} using GPU {gpu_id}")

    # Run DFT/TDDFT
    exc_energies, tdms = DFT_gpu.main(molecule, time_idx, do_tddft)
    return exc_energies, tdms


# # NOTE : (old) function that parallel executes 
# def main(mol_1, mol_2, time_steps, do_tddft):
    
#     startT = time.time()
#     for t in range(time_steps):
#         print(f"\n Running Time Step {t}...", flush = True)
#         start_time = time.time()
#         # Run molecule_1 on GPU 0 and molecule_2 on GPU 1
#         exc_energies_1, tdms_1 = run_dft_tddft(mol_1, t, gpu_id=0, do_tddft=do_tddft)
#         exc_energies_2, tdms_2 = run_dft_tddft(mol_2, t, gpu_id=1, do_tddft=do_tddft)

#         # print test-wise
#         print(exc_energies_1)
#         print(exc_energies_2)
#         print(tdms_1.shape)

#         end_time = time.time()  # End timing for this step
#         elapsed_time = end_time - start_time
#         print(f"Time Step {t} Completed in {elapsed_time:.2f} seconds", flush = True)

#     endT = time.time()
#     print(f"All DFT/TDDFT calculations completed in {endT -startT} sec!")

def main(mol_1, mol_2, time_steps, do_tddft):
    startT = time.time()
    results = []

    with ProcessPoolExecutor(max_workers = 2) as executor:  # Use 2 parallel processes
        for t in range(time_steps):
            print(f"\nRunning Time Step {t}...", flush=True)
            start_time = time.time()

            # Run both calculations in parallel
            future1 = executor.submit(run_dft_tddft, mol_1, t, 0, do_tddft)
            future2 = executor.submit(run_dft_tddft, mol_2, t, 1, do_tddft)

            # Retrieve results when both processes complete
            exc_energies_1, tdms_1 = future1.result()
            exc_energies_2, tdms_2 = future2.result()

            print(exc_energies_1)
            print(exc_energies_2)
            print(tdms_1)

            #results.append((t, exc_energies_1, tdms_1, exc_energies_2, tdms_2))

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time Step {t} Completed in {elapsed_time:.2f} seconds", flush=True)

    print(f"All DFT/TDDFT calculations completed in {time.time() - startT} sec!")



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

