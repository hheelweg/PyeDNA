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

# TODO : we manually set the path to the input scrips here, but ideally
# we want to use the path from where this script is being executed
path_to_files = '/home/hheelweg/Cy3Cy5/PyCY/test/prod'
sys.path.append(path_to_files)


def main(molecule_id):

    # (0) set settings for QM (DFT/TDDFT) calculation
    settings_dft, settings_tddft = quantumTools.setQMSettings('qm.params')

    # (0) load output information
    output_keys = quantumTools.parseQMOutput('qm_out.params')
    # intialize value dict for storing outputs
    values = {key: None for key in output_keys}

    # (1) load chromophore pyscf input from cache as DFT/TDDFT input
    chromophore_conv = load(f"input_{molecule_id}.joblib")

    # (2) perform DFT/TDDFT calculation and store outputs
    values['mol'], values['mf'], values['occ'], values['virt'] = quantumTools.doDFT_gpu(chromophore_conv, **settings_dft)
    if settings_tddft.pop("do_tddft", False):
        values['exc'], values['tdm'], values['dip'], values['osc'], values['idx'] = quantumTools.doTDDFT_gpu(values['mf'], values['occ'], values['virt'], **settings_tddft)

    # (3) output quantities of interest
    # TODO : might want to add that specific outputs are only possible if do_tddft is set to True
    output = {key: values[key] for key, value in output_keys.items() if value}
    for key in output:
        dump(values[key], f"{key}_{molecule_id}.joblib")



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


