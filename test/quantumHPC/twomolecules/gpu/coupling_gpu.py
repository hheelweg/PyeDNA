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



def getMol(mol_idx, time_idx):
    MDsim = traj.MDSimulation([])                           # empty MDSimulation object

    path = '/home/hheelweg/Cy3Cy5/PyCY/test/prod/'          # specify relative path to MD ouput
    name_prmtop = 'dna_test.prmtop'
    name_nc = 'dna_test_prod.nc'                            
    name_out = 'dna_test_prod.out'
              

    data = [name_prmtop,name_nc, name_out]                  # trajectory data 
    test = traj.Trajectory(MDsim, path, data)               # initialize Trajectory object

    # (1) specify chromophore to perform DFT/TDDFT on
    molecule = [mol_idx]
    chromophore, chromophore_conv = test.getChromophoreSnapshot(time_idx, molecule, conversion = 'pyscf')

    # (2) convert to pyscf mol object
    mol = gto.M(atom = chromophore_conv,
                basis = '6-31g',
                charge = 0,
                spin = 0)
    return mol



def main(molecules, time_idx):
    # NOTE : this script only serves the purpose of debugging 
    print('** Debug script to compute the coupling from the TDM of two molecules')

    exc = []
    tdm = []
    mols = []
    for mol in molecules:
        # load molecule data from DFT/TDDFT
        with np.load(f"output_{mol}.npz") as data:
            exc_energies = data["exc_energies"]
            tdms = data["tdms"]

        exc.append(exc_energies)
        tdm.append(tdms)
        mols.append(getMol(mol, time_idx))
        
    # having loaded the pyscf mol object as well 




if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations")
    parser.add_argument("molecule_1_id", type=int, help="Molecule 1 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("molecule_2_id", type=int, help="Molecule 2 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("time_idx", type=int, help="Time index (integer)")                      # specifies time step upon we wish to analyze trajectory
    args = parser.parse_args()

    molecules = [args.molecule_1_id, args.molecule_2_id]
    main(molecules, args.time_idx)