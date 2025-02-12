import numpy as np
import os
from pyscf import gto, lib
import argparse
import sys
import scipy
import time

# import custom modules
path_to_modules = '/home/hheelweg/Cy3Cy5/PyCY'
sys.path.append(path_to_modules)
import quantumTools, structure
import trajectory as traj
import const


# get pyscf mol object
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


# NOTE : we here try to compute the coupling
# compute coupling test-wise
def getCoupling(molA, molB, tdmA, tdmB, calcK = False):

    from pyscf.scf import jk, _vhf

    # (0) check that dimensions are correct
    assert(tdmA.shape == (molA.nao, molA.nao))
    assert(tdmB.shape == (molB.nao, molB.nao))

    # (1) merge separate molecules together and set up joint density matrix
    mol_AB = molA + molB
    dm_AB = scipy.linalg.block_diag(tdmA, tdmB) 
    
    # (2) set of HF infrastucture for fast integral evaluation
    # 'int2e' specifies two-electron integrals, 'CVHFnrs8_prescreen' use prescreen options to reduce computational time
    # key idea : instead of directly computing all (ij∣kl) integrals, PySCF uses prescreening techniques to skip irrelevant terms.
    vhfopt = _vhf.VHFOpt(mol_AB, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf',
                         'CVHFsetnr_direct_scf_dm')                    
    vhfopt.set_dm(dm_AB, mol_AB._atm, mol_AB._bas, mol_AB._env)             # enables denisity-based prescreening
    vhfopt._dmcondname = None

    # (3) compute Coulomb integrals
    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0        , molA.nbas , 0        , molA.nbas,
                      molA.nbas, mol_AB.nbas, molA.nbas, mol_AB.nbas)  # AABB
        vJ = jk.get_jk(mol_AB, tdmB, 'ijkl,lk->s2ij', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, tdmA)
    
    # (4) compute Exchange integrals
    if calcK == True:
        with lib.temporary_env(vhfopt._this.contents,
                               fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0        , molA.nbas , molA.nbas, mol_AB.nbas,
                          molA.nbas, mol_AB.nbas, 0        , molA.nbas)  # ABBA
            vK = jk.get_jk(mol_AB, tdmB, 'ijkl,jk->il', shls_slice=shls_slice,
                           vhfopt=vhfopt, aosym='s1', hermi=0)
            cK = np.einsum('ia,ia->', vK, tdmA)
            
        return cJ, cK
    
    else: 
        return cJ, 0

# NOTE : brute force way to compute the Coulomb coupling
def getCouplingBF(molA, molB, tdmA, tdmB):
    # (1) compute the Coulomb interaction matrix
    eri = molB.intor('int2e', aosym=1)                                              # (ij|kl) integrals
    J = eri.reshape(molB.nao, molB.nao, molA.nao, molA.nao).sum(axis=(2, 3))        # Sum over k, l, should have shape (naoA, naoA)
    # (2) compute electronic coupling
    V = np.einsum('ij,ij->', tdmB, J @ tdmA)
    return V


def main(molecules, time_idx):
    # NOTE : this script only serves the purpose of debugging 
    print('** Debug script to compute the coupling from the TDM of two molecules')

    exc = []
    tdm = []
    mols = []
    for molecule_id in molecules:
        # load molecule data from DFT/TDDFT
        with np.load(f"output_{molecule_id}.npz") as data:
            exc_energies = data["exc_energies"]
            tdms = data["tdms"]

        exc.append(exc_energies)
        tdm.append(tdms)
        mols.append(getMol(molecule_id, time_idx))
        
    # NOTE : since we have run TDDFT based on three excited states state_ids = [0, 1, 2],
    # each dimension in the TDM correspond to one of the states specified
    # we here only want to use the TDM for the first excited state, i.e. state_id = 0 and therefore use tdm[molecule_id][0]
    # to compute the coupling

    # compute coupling (Ardy's function)
    start_time = time.time()
    a, b = getCoupling(mols[0], mols[1], tdm[0][0], tdm[1][0])
    end_time = time.time()
    print(f"Elapsed time for computing the coupling: {end_time - start_time} seconds")
    print(a, b)

    # compare to my coupling function
    start_time = time.time()
    a = getCouplingBF(mols[0], mols[1], tdm[0][0], tdm[1][0])
    end_time = time.time()
    print(f"Elapsed time for computing the BF coupling: {end_time - start_time} seconds")
    print(a )




if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations")
    parser.add_argument("molecule_1_id", type=int, help="Molecule 1 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("molecule_2_id", type=int, help="Molecule 2 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("time_idx", type=int, help="Time index (integer)")                      # specifies time step upon we wish to analyze trajectory
    args = parser.parse_args()

    molecules = [args.molecule_1_id, args.molecule_2_id]
    main(molecules, args.time_idx)