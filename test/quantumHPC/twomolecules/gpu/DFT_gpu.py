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


def main(molecule_id):

    # (0) set settings for QM (DFT/TDDFT) calculation
    settings = quantumTools.setQMSettings('qm.params')
    settings_dft = {key: settings[key] for key in ["basis", "xc", "density_fit", "charge", "spin", "scf_cycles", "verbosity"]}
    settings_tddft = {key: settings[key] for key in ["state_ids", "TDA"]}

    # (1) load chromophore pyscf input from cache
    chromophore_conv = load(f"input_{molecule_id}.joblib")

    # (2) perform DFT calculation
    mol, mf, occ, virt = quantumTools.doDFT_gpu(chromophore_conv, **settings_dft)

    # (3) dump mol object to cache
    dump(mol, f"mol_{molecule_id}.joblib")

    # (3) optional: do TDDFT calculation based on that result:
    print(settings['do_tddft'])
    if settings['do_tddft']:
        exc_energies, tdms = quantumTools.doTDDFT_gpu(mf, occ, virt, **settings_tddft)
        return exc_energies, tdms


if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations on molecule")
    parser.add_argument("molecule_id", type=int, help="Molecule ID (integer)")                              # specifies residue name of molecule
    args = parser.parse_args()

    exc_energies, tdms = main(args.molecule_id)

    # # write array output to binary stream
    # np.savez(sys.stdout.buffer, exc_energies = exc_energies, tdms = tdms)
    # sys.stdout.flush()
    import io
    output = io.BytesIO()
    np.savez(output, exc_energies = exc_energies, tdms = tdms)
    # Write pure binary data to stdout
    sys.stdout.buffer.write(output.getvalue())
    sys.stdout.flush()


    # # TODO : we only have this for debugging purposes where we actually need the TDMs so that 
    # # we don't have to run DFT/TDDFT over and over again
    # # save arrays to file for debugging
    # filename = f"output_{args.molecule_id}.npz"
    # np.savez(filename, exc_energies = exc_energies, tdms = tdms)


