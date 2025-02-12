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


# NOTE : fast function to compute coupling terms cJ and cK
def getCJCK(molA, molB, tdmA, tdmB, get_cK = False):

    from pyscf.scf import jk, _vhf

    # (0) check that dimensions are correct
    assert(tdmA.shape == (molA.nao, molA.nao))
    assert(tdmB.shape == (molB.nao, molB.nao))

    # (1) merge separate molecules together and set up joint density matrix
    molAB = molA + molB
    dm_AB = scipy.linalg.block_diag(tdmA, tdmB) 
    
    # (2) set of HF infrastucture for fast integral evaluation
    # 'int2e' specifies two-electron integrals, 'CVHFnrs8_prescreen' use prescreen options to reduce computational time
    # key idea : instead of directly computing all (ij∣kl) integrals, PySCF uses prescreening techniques to skip irrelevant terms.
    vhfopt = _vhf.VHFOpt(molAB, 'int2e', 'CVHFnrs8_prescreen', 'CVHFsetnr_direct_scf', 'CVHFsetnr_direct_scf_dm')                    
    vhfopt.set_dm(dm_AB, molAB._atm, molAB._bas, molAB._env)             # enables density-based prescreening
    vhfopt._dmcondname = None

    # (3) compute Coulomb integrals
    with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0, molA.nbas, 0, molA.nbas, molA.nbas, molAB.nbas, molA.nbas, molAB.nbas) 
        vJ = jk.get_jk(molAB, tdmB, 'ijkl,lk->s2ij', shls_slice=shls_slice, vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, tdmA)
    
    # (4) compute Exchange integrals
    if get_cK == True:
        with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0, molA.nbas , molA.nbas, molAB.nbas, molA.nbas, molAB.nbas, 0, molA.nbas)  
            vK = jk.get_jk(molAB, tdmB, 'ijkl,jk->il', shls_slice=shls_slice, vhfopt=vhfopt, aosym='s1', hermi=0)
            cK = np.einsum('ia,ia->', vK, tdmA)
        return cJ, cK
    else: 
        return cJ, 0


# NOTE : brute force way to compute the electronic coupling terms cJ and cK
# [with explanation of the individual terms]
def getCJCK_BF(molA, molB, tdmA, tdmB, get_cK = False):
    from pyscf.scf import jk
    """ Efficiently computes the Coulomb interaction between molA and molB
        according to the formula cJ = \sum_{i,j, k, l} P_A(i,j)* J_ijkl * P_B(k,l)  
        with J_ijkl = \int AO_i(r)AO_j(r) * (1/|r-r'|) * AO_k(r')AO_l(r') dr dr' as the Coulomb matrix 
        which is computed internally with jk.get_jk()
    """
    
    # (0) Merge donor and acceptor into one system
    molAB = molA + molB 

    # (1) Pad TDM to match the full molecule size
    def pad_tdm_in_molAB(molA, molB, tdmB):
        """Embeds tdmB into the full-sized molecule molAB to match dimensions for `jk.get_jk()`."""
        naoA = molA.nao
        naoB = molB.nao
        naoAB = naoA + naoB 
        
        # Create a zero matrix of the full dimension
        tdmB_padded = np.zeros((naoAB, naoAB))
        
        # Insert tdmB into the correct block
        tdmB_padded[naoA:naoA+naoB, naoA:naoA+naoB] = tdmB
        
        return tdmB_padded
    
    tdmB = pad_tdm_in_molAB(molA, molB, tdmB)
    
    # (2) Compute Coulomb interaction matrix J_ijkl using HF potential optimization
    # pre-multiplies J_ijkl with P_B(k,l) (TDM of molecule B), i.e. \sum_{k,l} J_ijkl * P_B(k,l) 
    vJ = jk.get_jk(molAB, tdmB, 'ijkl,lk->s2ij', aosym='s4', hermi=1)
    
    # (3) Contract with P_A(i,j) (TDM of molecule A) to obtain Coulombic coupling
    # i.e. cJ = \sum_{i,j} P_A(i,j)* [\sum_{k,l} J_ijkl * P_B(k,l)]
    cJ = np.einsum('ij,ij->', tdmA, vJ[:molA.nao, :molA.nao])
    
    # (4) Compute Exchange potential matrix (cK) if requested
    if get_cK:
        vK = jk.get_jk(molAB, tdmB, 'ijkl,jk->il', aosym='s1', hermi=0)
        cK = np.einsum('ij,ij->', tdmA, vK[:molA.nao, :molA.nao])
        return cJ, cK  
    else:
        return cJ, 0 


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

    # accelerated computation of coupling
    start_time = time.time()
    #cJ, cK = getCJCK(mols[0], mols[1], tdm[0][0], tdm[1][0], get_cK = False)
    cJ, cK = getCJCK(mols[1], mols[0], tdm[1][0], tdm[0][0], get_cK = False)
    end_time = time.time()
    print(f"Elapsed time for computing the coupling: {end_time - start_time} seconds")
    print(cJ, cK)

    # compare to brute-force coupling function
    start_time = time.time()
    #cJ, cK = getCJCK_BF(mols[0], mols[1], tdm[0][0], tdm[1][0], get_cK = False)
    cJ, cK = getCJCK_BF(mols[1], mols[0], tdm[1][0], tdm[0][0], get_cK = False)
    end_time = time.time()
    print(f"Elapsed time for computing the coupling (brute force): {end_time - start_time} seconds")
    print(cJ, cK)

    # NOTE : both ways of computing the couplings lead to the same results
    # but getCJCK() is much fast than getCJCK_BF(), so we prefer that computational advantage


if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations")
    parser.add_argument("molecule_1_id", type=int, help="Molecule 1 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("molecule_2_id", type=int, help="Molecule 2 ID (integer)")              # specifies residue name of molecule 1
    parser.add_argument("time_idx", type=int, help="Time index (integer)")                      # specifies time step upon we wish to analyze trajectory
    args = parser.parse_args()

    molecules = [args.molecule_1_id, args.molecule_2_id]
    main(molecules, args.time_idx)