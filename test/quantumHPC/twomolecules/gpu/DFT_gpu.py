import numpy as np
import os
from pyscf import gto, lib
from gpu4pyscf import scf, scf, solvent, tdscf
from gpu4pyscf.dft import rks
import argparse
import sys
import time
import cupy as cp

# import custom modules
path_to_modules = '/home/hheelweg/Cy3Cy5/PyCY'
sys.path.append(path_to_modules)
import quantumTools, structure
import trajectory as traj


# GPU-supported DFT
def doDFT_gpu(molecule, basis = '6-31g', xc = 'b3lyp', density_fit = False, charge = 0, spin = 0, scf_cycles = 200, verbosity = 4):

    # (1) make PySCF molecular structure 
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin)
    mol.verbose = verbosity

    # (2) initialize SCF object
    mf = rks.RKS(mol)
    mf.xc = xc
    mf.max_cycle = scf_cycles               
    mf.conv_tol = 1e-5
    mf = mf.SMD()                           # TODO : look up this model
    mf.with_solvent.method = 'COSMO'        # COSMO implicit solvent model 
    if density_fit:                         # optional: use density fit for accelerating computation
        mf.density_fit()

    # (3) run DFT
    mf.kernel()       

    # (4) output
    mo = mf.mo_coeff                        # MO Coefficients
    occ = mo[:, mf.mo_occ != 0]             # occupied orbitals
    virt = mo[:, mf.mo_occ == 0]            # virtual orbitals

    return mf, occ, virt

# GPU-supported TDDFT
def doTDDFT_gpu(molecule_mf, occ_orbits, virt_orbits, state_ids = [0], TDA = True):

    # (1) number of states
    nstates = len(state_ids)

    # (2) run TDDFT with or without TDA (Tamm-Dancoff approximation)
    td = molecule_mf.TDA().run(nstates = nstates) if TDA else molecule_mf.TDDFT().run(nstates = nstates)

    # (3) extract excitation energies and transition dipole moments
    exc_energies = [td.e[id] for id in state_ids]
    trans_dipoles = [td.transition_dipole()[id] for id in state_ids]

    # (4) compute oscillator strengths
    # (4.1) for all possible transitions
    osc_strengths = [2/3 * exc_energies[i] * np.linalg.norm(trans_dipoles[i])**2 for i in range(len(exc_energies))]
    # (4.2) find strongest transition
    osc_idx = np.argmax(osc_strengths) if not any(np.array(osc_strengths) > 0.1) else np.argwhere(np.array(osc_strengths) > 0.1)[0][0]

    # (5) compute TDM (Tarnsition Density Matrix) for all states
    # td.xy[i] is tuple (X_i, Y_i) with X_i contains the expansion coefficients for the excitation part of the i-th excited state
    # and Y_1 the expansion coefficients for the de-excitation part; in TDDFT td.xy[i][0] quantifies how the virtual orbitals mix
    # with the occupied orbital in the i-th excitation
    tdms = [cp.sqrt(2) * cp.asarray(occ_orbits).dot(cp.asarray(td.xy[id][0])).dot(cp.asarray(virt_orbits).T) for id in state_ids]

    return exc_energies, trans_dipoles, osc_strengths, tdms, osc_idx


def main(molecule_id, time_idx, do_tddft):

    MDsim = traj.MDSimulation([])                           # empty MDSimulation object

    path = '/home/hheelweg/Cy3Cy5/PyCY/test/prod/'          # specify relative path to MD ouput
    name_prmtop = 'dna_test.prmtop'
    name_nc = 'dna_test_prod.nc'                            
    name_out = 'dna_test_prod.out'
              

    data = [name_prmtop,name_nc, name_out]                  # trajectory data 
    test = traj.Trajectory(MDsim, path, data)               # initialize Trajectory object

    # (1) specify chromophore to perform DFT/TDDFT on
    molecule = [molecule_id]
    chromophore, chromophore_conv = test.getChromophoreSnapshot(time_idx, molecule, conversion = 'pyscf')

    # (2) perform DFT calculation
    start_time = time.time()
    mf, occ, virt = doDFT_gpu(chromophore_conv, density_fit=False)
    end_time = time.time()
    # (2.1) elapsed time after DFT
    print(f"Elapsed time (after DFT): {end_time - start_time} sec")

    # (3) optional: do TDDFT calculation based on that result:
    if do_tddft:
        state_ids = [0, 1, 2]                                     # might want to add more states
        exc_energies, trans_dipoles, osc_strengths, tdms, osc_idx = doTDDFT_gpu(mf, occ, virt, state_ids, TDA=True)
        end_time = time.time()
         # (3.1) elapsed time after TDDFT
        print(f"Elapsed time (after DFT + TDDFT) in step {time_idx}: {end_time - start_time} sec")


if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations on molecule")
    parser.add_argument("molecule_id", type=int, help="Molecule 1 ID (integer)")                # specifies residue name of molecule
    parser.add_argument("time_idx", type=int, help="Time index (integer)")                      # specifies time idx of trajectory
    parser.add_argument("--do-tddft", action="store_true", help="Enable TDDFT calculation")
    args = parser.parse_args()

    # run main
    main(args.molecule_id, args.time_idx, args.do_tddft)