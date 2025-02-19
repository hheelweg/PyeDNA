import numpy as np 
from pyscf import gto, scf, geomopt, tdscf, lib, dft, lo, solvent
from MDAnalysis.coordinates.XYZ import XYZReader
from MDAnalysis.coordinates.PDB import PDBWriter
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
import subprocess
import scipy

from joblib import dump, load
import os

# from current package
from . import utils
from . import fileproc as fp
from . import const


# optimize molecular structure from *.xyz file into optimized structure in *.pdb file
def optimizeStructureQM(molecule, basis = 'sto-3g'):

    from pyscf.geomopt.berny_solver import optimize

    # load molecule
    mol = gto.M(
        atom = molecule,            # Path to the XYZ file
        basis = '3-21G',            # Choose a basis set (e.g., sto-3g, 6-31G, cc-pVDZ)
        charge = 1,
        symmetry = True
    )
    # Perform SCF calculation
    # mf = scf.RHF(mol)  # Restricted Hartree-Fock

    mf = scf.KS(mol)  # Use DFT instead of HF
    mf.xc = 'b3lyp'   # Specify the B3LYP functional

    # Optimize the geometry
    optimizedMol = optimize(mf, maxsteps = 2)
    optimizedMol.kernel()

    # Step 2: Define a function to infer bonds based on distances
    def infer_bonds(coords, elements, bond_threshold = 1.6):
        bonds = []
        num_atoms = len(coords)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                # Use covalent radii to determine bonding
                if dist < bond_threshold:
                    bonds.append((i + 1, j + 1))  # PDB uses 1-based indexing
        return bonds

    # Step 3: Write the PDB file
    def write_pdb(filename, coords, elements, bonds=None):
        with open(filename, 'w') as pdb_file:
            pdb_file.write("HEADER    Optimized structure from PySCF\n")
            pdb_file.write("TITLE     PySCF Optimization Output\n")
            
            # Write atom coordinates
            for idx, (element, coord) in enumerate(zip(elements, coords), start=1):
                pdb_file.write(
                    f"ATOM  {idx:5d}  {element:<2}  MOL     1    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n"
                )
            
            # Write bonds if available
            if len(bonds) > 0:
                pdb_file.write("CONECT\n")
                for bond in bonds:
                    pdb_file.write(f"CONECT{bond[0]:5d}{bond[1]:5d}\n")
            
            pdb_file.write("END\n")

    # Step 4: Get coordinates and elements from the optimized molecule
    coords = optimizedMol.atom_coords(unit="angstrom")
    elements = [mol.atom_symbol(i) for i in range(mol.natm)]

    # Step 5: Infer bonds 
    bonds = infer_bonds(coords, elements)
    print(bonds)

    # Step 6: Write to PDB
    write_pdb("optimized_structure.pdb", coords, elements, bonds)

# convert and optimize molecule in *.cdx (ChemDraw) format into *.pdb file (unconstrained pre-optimization)
def optimizeStructureFF(path, moleculeName, stepsNo = 50000, econv = 1e-12, FF = 'UFF'):
    from openbabel import openbabel
    # (1) convert *.cdx into *.smi (SMILES string)
    command = f'obabel -icdx {path + moleculeName}.cdx -osmi -O {path + moleculeName}.smi'
    subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    
    # (2) read smiles string:
    with open(path + moleculeName + '.smi', "r") as file:
        smiles = file.readline().strip()
    smiles = fr"{smiles}" # convert into raw string
    
    # (3) read SMILES string into OpenBabbel
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "pdb")
    # Create an OBMol object
    mol = openbabel.OBMol()
    # Read the SMILES string into the OBMol object
    obConversion.ReadString(mol, smiles)
    # Add explicit hydrogens to the molecule
    mol.AddHydrogens()
    # Generate 3D coordinates (preserves connectivity)
    builder = openbabel.OBBuilder()
    builder.Build(mol)  # Generate 3D geometry without changing connectivity


    # (4) perform forcefield optimization
    # econv defines the convergence tolerance for the energy, stepsNo the step number of the optimization
    forcefield = openbabel.OBForceField.FindForceField(FF)
    # NOTE : it seems like constrained optimization is not activated in the Pythin API of OpenBabbel
    forcefield.Setup(mol)
    forcefield.ConjugateGradients(stepsNo, econv)  
    forcefield.GetCoordinates(mol)
    
    # Save the molecule as an PDB file
    output_file = path + moleculeName + "_preopt.pdb"
    obConversion.WriteFile(mol, output_file)

# finer geometry optimization incorporating C2 symmetry of chromophore molecules and disance constraint between adjacent phosphor atoms
def optimizeStructureSymmetryFF(path, moleculeNamePDB, stepsNo = 50000, econv = 1e-12, FF = 'MMFF94'):

    from openbabel import openbabel

    # (0) load symmetry information
    symm_file = './createStructure/cy5/cy5_symm_info.txt'
    def readSymmetry(symm_file):
        axis_pair = None
        symmetry_pairs = []
        with open(symm_file, 'r') as file:
            for i, line in enumerate(file):
                # Parse the line as two integers
                columns = line.strip().split()
                pair = list(map(int, columns))
                # First row defines the two atoms that make the rotation axis
                if i == 0:
                    axis_pair = pair
                # every other rows defines 
                else:
                    symmetry_pairs.append(pair)
        symmetry_pairss = np.array(symmetry_pairs).T
        return axis_pair, symmetry_pairss
    # read indices (0-indexed !) for rotation axis and symmetry pairs
    axis_pair, symm_pairs = readSymmetry(symm_file)

    # (1) import .pdb file and load molecule
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdb")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, path + moleculeNamePDB + "_preopt.pdb")

    # (2) CONSTRAINED geometry optimization
    # (2.1) function to enforce C2 symmetry of molecule
    def enforceC2Symmetry(mol):
        # (1) get rotation axis vector and point in axis
        def getAxisInfo(mol, axis_pair):
            axis_atom1 = mol.GetAtom(axis_pair[0] + 1)
            axis_pos1 = np.array([axis_atom1.GetX(), axis_atom1.GetY(), axis_atom1.GetZ()])
            axis_atom2 = mol.GetAtom(axis_pair[1] + 1)
            axis_pos2 = np.array([axis_atom2.GetX(), axis_atom2.GetY(), axis_atom2.GetZ()])
            # get point in axis and rotation axis vector (normalized)
            axis_point = axis_pos1
            axis_vec = (axis_pos2 - axis_pos1) / np.linalg.norm(axis_pos2 - axis_pos1)
            return axis_vec, axis_point
        
        axis_vec, axis_point = getAxisInfo(mol, axis_pair)

        # (2) rotate atoms in first half of the molecule
        for i, atom_idx in enumerate(symm_pairs[0]):
            atom = mol.GetAtom(int(atom_idx) + 1)
            atom_pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
            # translate atom 
            atom_pos_trans = atom_pos - axis_point
            # Decompose into parallel and perpendicular components
            parallel_component = np.dot(atom_pos_trans, axis_vec) * axis_vec
            perpendicular_component = atom_pos_trans - parallel_component
            # Calculate the 180° rotated position
            atom_pos_rot= axis_point + parallel_component - perpendicular_component
            # overwrite position of symm_pairs[1] symmetry partner
            atom_symm = mol.GetAtom(int(symm_pairs[1][i]) + 1)
            atom_symm.SetVector(*atom_pos_rot)

    # (2.2) find phosphorus atoms to constrain them
    P_atoms = [atom for atom in openbabel.OBMolAtomIter(mol) if atom.GetAtomicNum() == 15]   # indices of phosphorus atoms
    # need to get 1-indexed indices
    P1_idx = P_atoms[0].GetIndex() + 1
    P2_idx = P_atoms[1].GetIndex() + 1

    # (3) optimize with C2 symmetry constraint and distance constraint on distance between P-atoms
    forcefield = openbabel.OBForceField.FindForceField(FF)
    # add distance constraint directly 
    constraint = openbabel.OBFFConstraints() 
    constraint.AddDistanceConstraint(P1_idx, P2_idx, 6.49)
    forcefield.SetConstraints(constraint)
    # NOTE : might want to play around with FastRotorSearch versus WeightedRotorSearch etc.
    # the current implementation seems to make the distance between the P-atoms smmaller, so one could choose a more hand-wavy
    # approach and aritficially make the distance in  constraint.AddDistanceConstraint() a little bit bigger than desired
    for _ in range(100):
        forcefield.Setup(mol)                           # need to feed back C2-coorected coordinates into forcefield
        forcefield.FastRotorSearch(True)
        forcefield.ConjugateGradients(1000, econv)      # conjugate gradient optimization
        enforceC2Symmetry(mol)                          # enforce C2 symmetry of molecule 
    forcefield.GetCoordinates(mol)
    enforceC2Symmetry(mol)                              # ensure output molecule has C2 symmetry

    # Save the molecule as an PDB file
    output_file = path + moleculeNamePDB + "_opt.pdb"
    obConversion.WriteFile(mol, output_file)


#--------------------------------------------------------------------------------------------------------------------------------------------
# functions e.g. for analyzing MD trajectories


# perform DFT calculation on molecule
def doDFT(molecule, basis = '6-31g', xc = 'b3lyp', 
          density_fit = False, charge = 0, spin = 0, scf_cycles = 200, verbosity = 4):

    # (1) make PySCF molecular structure object
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin)
    mol.verbose = verbosity

    # # TODO: Marias code here (What is this for?)
    # # optimize cap
    # if opt_cap is not None:
    #     mf = scf.RHF(mol)
    #     mol = contrained_opt(mf, opt_cap)

    # (2) initialize SCF object
    # mf = scf.RKS(mol)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.max_cycle = scf_cycles               
    mf.conv_tol = 1e-5
    if density_fit:                         # optional: use density fit for accelerating computation
        mf.density_fit(auxbasis="weigend")

    # (3) run with with SMD/ddCOSMO implicit solvent model
    mf = mf.SMD()
    mf.with_solvent.method = 'DDCOSMO'
    mf.kernel()       

    # (4) output quantities of interest:
    mo = mf.mo_coeff                        # MO Coefficients
    occ = mo[:, mf.mo_occ != 0]             # occupied orbitals
    virt = mo[:, mf.mo_occ == 0]            # virtual orbitals

    return mf, occ, virt


# perform TDDFT calculation on pyscf mf object
def doTDDFT(molecule_mf, occ_orbits, virt_orbits, state_ids = [0], TDA = True):

    # (1) number of states
    nstates = len(state_ids)

    # (2) run TDDFT with or without TDA (Tamm-Dancoff approximation)
    td = molecule_mf.TDA().run(nstates = nstates) if TDA else molecule_mf.TDDFT(nstates = nstates).run()

    # (3) extract excitation energies and transition dipole moments
    exc_energies = [td.e[id] for id in state_ids]
    trans_dipoles = [td.transition_dipole()[id] for id in state_ids]

    # (4) compute oscillator strengths
    # (4.1) for all possible transitions
    # TODO : Maria does not have the 2/3 pre-factor (find reason for this!)
    osc_strengths = [2/3 * exc_energies[i] * np.linalg.norm(trans_dipoles[i])**2 for i in range(len(exc_energies))]
    # (4.2) find strongest transition
    osc_idx = np.argmax(osc_strengths) if not any(np.array(osc_strengths) > 0.1) else np.argwhere(np.array(osc_strengths) > 0.1)[0][0]

    # (5) compute TDM (Tarnsition Density Matrix) for all states
    # td.xy[i] is tuple (X_i, Y_i) with X_i contains the expansion coefficients for the excitation part of the i-th excited state
    # and Y_1 the expansion coefficients for the de-excitation part; in TDDFT td.xy[i][0] quantifies how the virtual orbitals mix
    # with the occupied orbital in the i-th excitation
    tdms = [np.sqrt(2) * occ_orbits.dot(td.xy[id][0]).dot(virt_orbits.T) for id in state_ids]

    return exc_energies, trans_dipoles, osc_strengths, tdms, osc_idx

# do DFT with GPU support
# TODO : merge with doDFT()
def doDFT_gpu(molecule, basis = '6-31g', xc = 'b3lyp', 
              density_fit = False, charge = 0, spin = 0, scf_cycles = 200, verbosity = 4):
    
    # (0) import gou4pyscf and GPU support
    from gpu4pyscf import scf, solvent, tdscf
    from gpu4pyscf.dft import rks
    import cupy as cp


    # (1) make PySCF molecular structure object 
    mol = gto.M(atom = molecule,
                basis = basis,
                charge = charge,
                spin = spin)
    mol.verbose = verbosity


    # (2) initialize SCF object
    mf = rks.RKS(mol)
    mf.xc = xc
    mf.max_cycle = scf_cycles               
    mf.conv_tol = 1e-5                      # TODO : only did this for debugging
    mf = mf.SMD()                           # TODO : look up this model
    mf.with_solvent.method = 'DDCOSMO'      # COSMO implicit solvent model 
    if density_fit:                         # optional: use density fit for accelerating computation
        mf.density_fit()

    # (3) run DFT
    mf.kernel()       

    # (4) output
    mo = mf.mo_coeff                        # MO Coefficients
    occ = mo[:, mf.mo_occ != 0]             # occupied orbitals
    virt = mo[:, mf.mo_occ == 0]            # virtual orbitals

    return mol, mf, occ, virt


# do TDDFT with GPU support
# TODO : merge with doTDDFT()
def doTDDFT_gpu(molecule_mf, occ_orbits, virt_orbits, state_ids = [0], TDA = True):

    # (0) import gou4pyscf and GPU support
    from gpu4pyscf import scf, solvent, tdscf
    from gpu4pyscf.dft import rks
    import cupy as cp

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

    # (5) compute TDM (Transition Density Matrix) for all states
    # td.xy[i] is tuple (X_i, Y_i) with X_i contains the expansion coefficients for the excitation part of the i-th excited state
    # and Y_1 the expansion coefficients for the de-excitation part; in TDDFT td.xy[i][0] quantifies how the virtual orbitals mix
    # with the occupied orbital in the i-th excitation
    tdms = [cp.sqrt(2) * cp.asarray(occ_orbits).dot(cp.asarray(td.xy[id][0])).dot(cp.asarray(virt_orbits).T) for id in state_ids]

    # return numpy arrays
    return np.array(exc_energies), np.array([tdm.get() for tdm in tdms]), np.array(trans_dipoles), np.array(osc_strengths), osc_idx


# NOTE : function that calls python ssubprocess to perform DFT/TDDFT on individual GPUs with PySCF
# TODO : make this more flexible with regards to the path where the launcher (DFT_gpu.py) is
def launchQMdriver(molecule_no, gpu_id):
    """Launch a DFT/TDDFT calculation on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign GPU
    print('testdebug', molecule_no, flush=True)
    # path+file_name for execution of qm_driver.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qm_driver_path = os.path.join(script_dir, "qm_driver.py")

    cmd = f"python {qm_driver_path} {molecule_no}"
    process = subprocess.Popen(cmd, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        

    return process


# do PySCF on molecules = [mol1, mol2] where mol are the nuclear coordinates for PySCF calculations
# TODO : make this also without GPU-support depending on the available resources
def doQM_gpu(molecules, output_keys):

    # (0) initialize output dictionary for quantities of interest
    # [] stores data for both molecules in a list-type fashion
    output = {key: [] for key, value in output_keys.items() if value}


    # (1)run molecules on different GPUs in parallel
    procs = []
    for i, molecule in enumerate(molecules):
        # create pyscf input for subprocess and store in cache
        dump(molecule, f"input_{i}.joblib")
        # run subprocess
        procs.append(launchQMdriver(i, gpu_id = i))
    
    # wait for both subprocesses to finish
    for i, molecule in enumerate(molecules):
        procs[i].wait()

    # (2) load and store relevant data from output of subprocesses
    # TODO : flexibilize this for quantities we are interested in
    for i, molecule in enumerate(molecules):
        for key in output:
            output[key].append(load(f"{key}_{i}.joblib"))

    # (3) clean subprocess cache 
    utils.cleanCache()

    # returns output dictionary with keys specified in output_keys
    return output


# coupling terms for the computation cJ and cK 
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


# compute coupling terms ('cJ', 'cK', 'electronic', 'both') for the states (S_0^A , S_{stateB + 1}^B) <--> (S_{stateA + 1}^A, S_0^B)
# 'cJ' only returns the electrostatic interaction, 'cK' only the exchange interaction, 'electronic' returns 2 * cJ - cK
# NOTE : stateA and stateB are zero-indexed here so stateA = 0 corresponds to the first excited state of molecule A etc.
# stateA and stateB default to 0 to for the transition (S_0^A , S_1^B) <--> (S_1^A, S_0^B)
def getVCoulombic(mols, tdms, states, coupling_type = 'electronic'):

    stateA, stateB = states[0], states[1]
    molA, molB = mols[0], mols[1]
    tdmA, tdmB = tdms[0][stateA], tdms[1][stateB]

    if coupling_type in ['electronic', 'cK']:
        cJ, cK = getCJCK(molA, molB, tdmA, tdmB, get_cK=True)
    elif coupling_type in ['cJ']:
        cJ, _ = getCJCK(molA, molB, tdmA, tdmB, get_cK=False)
    else:
        raise NotImplementedError("Invalid coupling type specified!")
    
    results = {'coupling cJ': cJ, 'coupling cK': cK}
    if coupling_type == 'electronic':                                     
        results['coupling V_C'] = 2 * cJ - cK                               # total electronic coupling
    elif coupling_type == 'cK':
        results['couplingV_C'] = - cK                                       # exchange-interaction part of the electronic coupling
    elif coupling_type == 'cJ':
        results['coupling V_C'] = 2 * cJ                                    # electrostatic-interaction part of the electronic coupling
    return results

# get excitation energies for specified states
def getExcEnergies(excs, states, molecule_names = ['D', 'A'], excitation_energy_type = 'default'):

    stateA, stateB = states[0], states[1]
    excA, excB = excs[0], excs[1]

    results = {}
    results[f'energy {molecule_names[0]}'] = excA[stateA]
    results[f'energy {molecule_names[1]}'] = excB[stateB]
    return results


# compute absorption spectrum from oscillator strength and excitation energies
def getAbsorptionSpectrum(osc_strengths, exc_energies, sigma = 0.1, energy_units = 'eV'):

    # (0) convert excitation energies and oscillator strength to appropriate units
    # TODO : confirm that this is correct
    conv = energyConversion(energy_units)
    exc_energies *= conv
    osc_strengths *= conv
    print(exc_energies, osc_strengths)

    # (1) define energy grid for the plot
    energy_grid = np.linspace(0, max(exc_energies) + 1, 1000)
    spectrum = np.zeros_like(energy_grid)

    # (2) compute spectrum
    for i, exc_energy in enumerate(exc_energies):
        spectrum += osc_strengths[i] * np.exp(-((energy_grid - exc_energy) ** 2) / (2 * sigma ** 2))

    return energy_grid, spectrum


# function that handles energy conversion from Hartree to energy_units
def energyConversion(out_unit):
    if out_unit not in ['cm-1', 'E_h', 'eV']:
        raise ValueError("Specify valid energy unit!")
    # load conversion factors
    if out_unit == 'cm-1':
        factor = const.EH2CM
    elif out_unit == 'eV':
        factor = const.EH2EV
    elif out_unit == 'E_h':
        factor = 1.0
    return factor