import numpy as np
import os
import argparse
import sys
from joblib import load, dump

# import custom modules
path_to_modules = '/home/hheelweg/Cy3Cy5/PyCY'
sys.path.append(path_to_modules)
import quantumTools, structure
import trajectory as traj
import const


def main(molecule_id, do_tddft):

    # (1) load chromophore pyscf input from cache
    chromophore_conv = load(f"input_{molecule_id}.joblib")
    #os.remove(f"input_{molecule_id}.joblib")

    # (2) perform DFT calculation
    mol, mf, occ, virt = quantumTools.doDFT_gpu(chromophore_conv, density_fit=False, verbosity=0)

    # (3) dump mol object to cache
    # print('testtt')
    dump(mol, f"mol_{molecule_id}.joblib")

    # (3) optional: do TDDFT calculation based on that result:
    if do_tddft:
        state_ids = [0, 1, 2]                               # might want to add more states
        exc_energies, tdms = quantumTools.doTDDFT_gpu(mf, occ, virt, state_ids, TDA=True)
        return exc_energies, tdms


if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations on molecule")
    parser.add_argument("molecule_id", type=int, help="Molecule 1 ID (integer)")                # specifies residue name of molecule
    parser.add_argument("--do-tddft", action="store_true", help="Enable TDDFT calculation")
    args = parser.parse_args()

    exc_energies, tdms = main(args.molecule_id, args.do_tddft)

    # write array output to binary stream
    np.savez(sys.stdout.buffer, exc_energies = exc_energies, tdms = tdms)
    sys.stdout.flush()

    # TODO : we only have this for debugging purposes where we actually need the TDMs so that 
    # we don't have to run DFT/TDDFT over and over again
    # save arrays to file for debugging
    filename = f"output_{args.molecule_id}.npz"
    np.savez(filename, exc_energies = exc_energies, tdms = tdms)


