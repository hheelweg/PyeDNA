import os
import argparse
import time



# function that parallel executes 
def main(mol_1, mol_2, time_idx, do_tddft):
    
    start_time_global = time.time()
    print("to_dft:", do_tddft)
    # revaluate trajectory:
    for t in range(time_idx):
        # parallel execute molecule 1 and 2
        os.system(f"python DFT_gpu.py {mol_1} {t} {do_tddft} & python DFT_gpu.py {mol_2} {t} {do_tddft} & wait")

    print("All DFT/TDDFT calculations completed!")
    end_time_global = time.time()
    print(f"Total elapsed time: {end_time_global - start_time_global} sec.")



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

