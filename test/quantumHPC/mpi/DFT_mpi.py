import numpy as np
import os
from pyscf import lib
import argparse
import sys
import time

# import custom modules
path_to_modules = '/home/hheelweg/Cy3Cy5/PyCY'
sys.path.append(path_to_modules)
import quantumTools, structure
import trajectory as traj


def main(molecule_id, time_idx, do_tddft):

    MDsim = traj.MDSimulation([])                           # empty MDSimulation object

    path = '../../prod/'                                    # specify relative path to MD ouput
    name_prmtop = 'dna_test.prmtop'
    name_nc = 'dna_test_prod.nc'                            
    name_out = 'dna_test_prod.out'

    # enable PySCF multi-threading
    lib.num_threads()
    print(f"Numer of PySCF Threads: {lib.num_threads()}")              

    data = [name_prmtop,name_nc, name_out]                  # trajectory data 
    test = traj.Trajectory(MDsim, path, data)               # initialize Trajectory object

    # (1) specify chromophore to perform DFT/TDDFT on
    molecule = [molecule_id]
    chromophore, chromophore_conv = test.getChromophoreSnapshot(time_idx, molecule, conversion = 'pyscf')

    # (2) perform DFT calculation
    start_time = time.time()
    mf, occ, virt = quantumTools.doDFT(chromophore_conv)
    end_time = time.time()
    # (2.1) elapsed time after DFT
    print(f"Elapsed time (after DFT): {end_time - start_time} sec")

    # (3) optional: do TDDFT calculation based on that result:
    if do_tddft:
        state_ids = [0, 1, 2]                               # might want to add more states
        exc_energies, trans_dipoles, osc_strengths, tdms, osc_idx = quantumTools.doTDDFT(mf, occ, virt, state_ids)
        end_time = time.time()
         # (3.1) elapsed time after TDDFT
        print(f"Elapsed time (after DFT + TDDFT): {end_time - start_time} sec")


if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations")
    parser.add_argument("molecule_id", type=int, help="Molecule ID (integer)")                  # specifies residue name of one molecule
    parser.add_argument("time_idx", type=int, help="Time index (integer)")                      # specifies time slice of trajectory
    parser.add_argument("--do-tddft", action="store_true", help="Enable TDDFT calculation")
    args = parser.parse_args()

    # run main
    main(args.molecule_id, args.time_idx, args.do_tddft)