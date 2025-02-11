import numpy as np
import os
from pyscf import gto, lib
import argparse
import sys

# import custom modules
path_to_modules = '/home/hheelweg/Cy3Cy5/PyCY'
sys.path.append(path_to_modules)
import quantumTools, structure
import trajectory as traj
import const


def main(molecules, time_idx):
    # NOTE : this script only serves the purpose of debugging 
    print('** Debug script to compute the coupling from the TDM of two molecules')

    exc = []
    tdm = []
    for mol in molecules:
        # load molecule data from DFT/TDDFT
        with np.load(f"output_{mol}.npz") as data:
            exc_energies = data["exc_energies"]
            tdms = data["tdms"]

        exc.append(exc_energies)
        tdm.append(tdms)
        
        print(tdms.shape)




if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations")
    parser.add_argument("molecule_1_id", type=int, help="Molecule 1 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("molecule_2_id", type=int, help="Molecule 2 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("time_idx", type=int, help="Time index (integer)")                      # specifies time step upon we wish to analyze trajectory
    args = parser.parse_args()

    molecules = [args.molecule_1_id, args.molecule_2_id]
    main(molecules, args.time_idx)