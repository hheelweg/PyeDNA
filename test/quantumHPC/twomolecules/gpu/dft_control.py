import os
import argparse
import time
import subprocess
import torch

# Detect available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM allocation.")


def run_dft_tddft(molecule, time_idx, gpu_id, do_tddft):
    """Launch a DFT/TDDFT calculation on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign GPU

    cmd = f"python DFT_gpu.py {molecule} {time_idx}"
    if do_tddft:
        cmd += " --do-tddft"
    
    print(f"Running: {cmd} on GPU {gpu_id}")
    return subprocess.Popen(cmd, shell=True, env=env)

# function that parallel executes 
def main(mol_1, mol_2, time_steps, do_tddft):
    
    start = time.time()
    for t in range(time_steps):
        print(f"\n Running Time Step {t}...")

        # Run molecule_1 on GPU 0 and molecule_2 on GPU 1
        proc1 = run_dft_tddft(mol_1, t, gpu_id=0, do_tddft=do_tddft)
        proc2 = run_dft_tddft(mol_2, t, gpu_id=1, do_tddft=do_tddft)

        # Wait for both processes to complete before moving to next time step
        proc1.wait()
        proc2.wait()

        # print total time for time step:
        print(f"Total time after time step {t}: {time.time()} sec!")

    end = time.time()
    print(f"All DFT/TDDFT calculations completed in {end -start} sec!")



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

