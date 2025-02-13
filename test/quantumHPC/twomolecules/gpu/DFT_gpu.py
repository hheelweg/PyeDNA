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
# NOTE : only add this temporarily for debugging purposes
#import dft_control


def main(molecule_id):

    # (0) set settings for QM (DFT/TDDFT) calculation
    settings_dft, settings_tddft = quantumTools.setQMSettings('qm.params')

    # (0) load desired outputs
    #qm_outs = dft_control.parseQMOutput('qm_out.params')

    # (1) load chromophore pyscf input from cache
    chromophore_conv = load(f"input_{molecule_id}.joblib")

    # (2) perform DFT calculation
    mol, mf, occ, virt = quantumTools.doDFT_gpu(chromophore_conv, **settings_dft)

    # (3) dump mol object to cache
    dump(mol, f"mol_{molecule_id}.joblib")

    # (4) optional: do TDDFT calculation based on that result:
    if settings_tddft.pop("do_tddft", False):
        exc_energies, tdms = quantumTools.doTDDFT_gpu(mf, occ, virt, **settings_tddft)

        # output TDDFT quantities of interest
        dump(exc_energies, f"exc_{args.molecule_id}.joblib")
        dump(tdms, f"tdm_{args.molecule_id}.joblib")


if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations on molecule")
    parser.add_argument("molecule_id", type=int, help="Molecule ID (integer)")                              # specifies residue name of molecule
    args = parser.parse_args()

    # run main
    main(args.molecule_id)
    
    # # NOTE : write array output to binary stream
    # np.savez(sys.stdout.buffer, exc_energies = exc_energies, tdms = tdms)
    # sys.stdout.flush()


    # # TODO : we only have this for debugging purposes where we actually need the TDMs so that 
    # # we don't have to run DFT/TDDFT over and over again
    # # save arrays to file for debugging
    # filename = f"output_{args.molecule_id}.npz"
    # np.savez(filename, exc_energies = exc_energies, tdms = tdms)


