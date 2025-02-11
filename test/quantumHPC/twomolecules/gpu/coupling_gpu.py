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


def main(mol_1, mol_2):

    # load molecule 1
    with np.load(f"output_{mol_1}.npz") as data:
        exc_energies_1 = data["exc_energies"]
        tdms_1 = data["tdms"]

    # load molecule 2
    with np.load(f"output_{mol_2}.npz") as data:
        exc_energies_2 = data["exc_energies"]
        tdms_2 = data["tdms"]

    print(tdms_1.shape, tdms_2.shape)




if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations")
    parser.add_argument("molecule_1_id", type=int, help="Molecule 1 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("molecule_2_id", type=int, help="Molecule 2 ID (integer)")              # specifies residue name of molecule 1
    args = parser.parse_args()

    main(args.molecule_1_id, args.molecule_2_id)