"""PySCF quantum-analysis backend."""

import subprocess

import numpy as np
from pyscf import gto

from .base import DFTResult, QuantumBackend


class PySCFBackend(QuantumBackend):
    name = "pyscf"

    def run_dft(self, molecule_input, molecule_id, settings):
        mol, mf, occ, virt, orbital_energies = run_dft_gpu(
            molecule_input,
            molecule_id=molecule_id,
            **settings,
        )
        return DFTResult(
            molecule=mol,
            mean_field=mf,
            occupied_orbitals=occ,
            virtual_orbitals=virt,
            orbital_energies=orbital_energies,
        )

    def run_tddft(self, dft_result, output_flags, settings):
        return run_tddft_gpu(
            dft_result.molecule,
            dft_result.mean_field,
            dft_result.occupied_orbitals,
            dft_result.virtual_orbitals,
            output_flags,
            fragments=settings.get("fragments"),
            state_ids=settings.get("state_ids"),
            tda=settings.get("tda", True),
            singlet=settings.get("singlet", True),
        )


def run_dft_gpu(molecule_input, molecule_id, **settings):
    return do_dft_gpu(molecule_input, molecule_id=molecule_id, **settings)


def run_tddft_gpu(
    molecule_mol,
    molecule_mf,
    occupied_orbitals,
    virtual_orbitals,
    output_flags,
    *,
    fragments=None,
    state_ids=None,
    tda=True,
    singlet=True,
):
    return do_tddft_gpu(
        molecule_mol,
        molecule_mf,
        occupied_orbitals,
        virtual_orbitals,
        output_flags,
        fragments=fragments,
        state_ids=[0] if state_ids is None else state_ids,
        TDA=tda,
        singlet=singlet,
    )


def do_dft_gpu(
    molecule,
    molecule_id,
    basis="6-31g",
    xc="b3lyp",
    density_fit=False,
    charge=0,
    spin=0,
    scf_cycles=200,
    verbosity=4,
    optimize_cap=False,
):
    from gpu4pyscf.dft import rks

    mol = gto.M(atom=molecule, basis=basis, charge=charge, spin=spin)
    mol.verbose = verbosity

    if optimize_cap:
        mf_opt = rks.RKS(mol, xc=xc).density_fit()
        mf_opt.verbose = 0
        freeze_atom_string = f"1-{len(molecule) - 2}"
        mol = constrained_optimization(mf_opt, molecule_id, freeze_atom_string)

    mf = rks.RKS(mol)
    mf.xc = xc
    mf.max_cycle = scf_cycles
    mf.conv_tol = 1e-6
    mf = mf.PCM()
    mf.with_solvent.method = "COSMO"
    if density_fit:
        mf.density_fit()

    mf.kernel()

    mo = mf.mo_coeff
    occ = mo[:, mf.mo_occ != 0]
    virt = mo[:, mf.mo_occ == 0]
    orbital_energies = mf.mo_energy

    return mol, mf, occ, virt, orbital_energies


def do_tddft_gpu(
    molecule_mol,
    molecule_mf,
    occ_orbits,
    virt_orbits,
    quantum_dict,
    fragments=None,
    state_ids=None,
    TDA=False,
    singlet=True,
):
    import cupy as cp

    state_ids = [0] if state_ids is None else state_ids
    nstates = len(state_ids)

    molecule_td = molecule_mf.TDA().run(nstates=nstates) if TDA else molecule_mf.TDDFT().run(nstates=nstates)

    exc_energies = [molecule_td.e[state_id] for state_id in state_ids]
    trans_dipoles = [molecule_td.transition_dipole()[state_id] for state_id in state_ids]
    trans_quadpoles = [molecule_td.transition_quadrupole()[state_id] for state_id in state_ids]

    osc_strengths = [
        2 / 3 * exc_energies[i] * np.linalg.norm(trans_dipoles[i]) ** 2
        for i in range(len(exc_energies))
    ]
    osc_idx = (
        np.argmax(osc_strengths)
        if not any(np.array(osc_strengths) > 0.1)
        else np.argwhere(np.array(osc_strengths) > 0.1)[0][0]
    )

    tdms = [
        cp.sqrt(2)
        * cp.asarray(occ_orbits).dot(cp.asarray(molecule_td.xy[state_id][0])).dot(cp.asarray(virt_orbits).T)
        for state_id in state_ids
    ]

    tddft_output = {}
    if quantum_dict["exc"]:
        tddft_output["exc"] = np.array(exc_energies)
    if quantum_dict["tdm"]:
        tddft_output["tdm"] = np.array([tdm.get() for tdm in tdms])
    if quantum_dict["dip"]:
        tddft_output["dip"] = np.array(trans_dipoles)
    if quantum_dict["quad"]:
        tddft_output["quad"] = np.array(trans_quadpoles)
    if quantum_dict["osc"]:
        tddft_output["osc"] = np.array(osc_strengths)
    if quantum_dict["idx"]:
        tddft_output["idx"] = osc_idx

    if quantum_dict["mull_pops"] or quantum_dict["mull_chrgs"]:
        tddft_output["mull_pops"], tddft_output["mull_chrgs"] = do_mulliken_analysis(
            molecule_mf,
            molecule_mol,
            tdms,
            state_ids=state_ids,
        )

    if quantum_dict["OPA"] and fragments is not None:
        tddft_output["OPA"] = do_orbital_participation_analysis(
            molecule_mol,
            molecule_td,
            fragments,
            state_ids=state_ids,
            TDA=TDA,
        )

    return tddft_output


def do_mulliken_analysis(molecule_mf, molecule_mol, molecule_tdms, state_ids=None):
    state_ids = [0] if state_ids is None else state_ids
    atom_pops, atom_charges = [], []
    overlap = molecule_mf.get_ovlp()

    for index in range(len(state_ids)):
        tdm = molecule_tdms[index]
        assert tdm.shape == (molecule_mol.nao, molecule_mol.nao)
        pop, charges = mulliken_pop(molecule_mol, overlap, tdm)
        atom_pops.append(pop)
        atom_charges.append(charges)

    return atom_pops, atom_charges


def mulliken_pop(mol, overlap, density_matrix):
    ao_pops = np.einsum("ij,ji->i", density_matrix, overlap).real
    ao_to_atom = np.array([label[0] for label in mol.ao_labels(fmt=None)])

    atom_pops = np.zeros(mol.natm)
    for ao_idx in range(mol.nao):
        atom_pops[ao_to_atom[ao_idx]] += ao_pops[ao_idx]

    atom_charges = mol.atom_charges() - atom_pops
    return atom_pops, atom_charges


def do_orbital_participation_analysis(molecule_mol, molecule_td, fragments, state_ids=None, TDA=False):
    state_ids = [0] if state_ids is None else state_ids
    ao2atom = np.array([label[0] for label in molecule_mol.ao_labels(fmt=None)])

    assert len(fragments) == 2
    fragment_map = np.full(len(ao2atom), fill_value=-1)
    for frag_id, atom_indices in enumerate(fragments):
        for atom in atom_indices:
            fragment_map[ao2atom == atom] = frag_id

    mo_coeff = molecule_td._scf.mo_coeff
    nmo = mo_coeff.shape[1]
    mo_weights = np.zeros((nmo, len(fragments)))

    overlap = molecule_td._scf.get_ovlp()
    for mo_idx in range(nmo):
        coeff = mo_coeff[:, mo_idx]
        for frag_id in range(len(fragments)):
            frag_mask = fragment_map == frag_id
            coeff_frag = coeff[frag_mask]
            overlap_frag = overlap[np.ix_(frag_mask, frag_mask)]
            mo_weights[mo_idx, frag_id] = coeff_frag @ overlap_frag @ coeff_frag

    result = []
    for state_id in state_ids:
        x_amplitudes, y_amplitudes = molecule_td.xy[state_id]

        nocc = molecule_td._scf.mol.nelec[0]
        nvirt = mo_coeff.shape[1] - nocc
        transitions = [(i, j) for i in range(nocc) for j in range(nocc, nocc + nvirt)]

        if TDA:
            amplitudes = x_amplitudes.flatten()
        else:
            amplitudes = (x_amplitudes + y_amplitudes).flatten()

        frag_contributions = np.zeros((len(fragments), len(fragments)))
        for amplitude, (i_occ, i_virt) in zip(amplitudes, transitions):
            for frag_h in range(len(fragments)):
                for frag_p in range(len(fragments)):
                    weight = mo_weights[i_occ, frag_h] * mo_weights[i_virt, frag_p]
                    frag_contributions[frag_h, frag_p] += (amplitude**2) * weight

        total = frag_contributions.sum()
        if total > 0:
            frag_contributions /= total
        result.append(frag_contributions)

    return result


def constrained_optimization(mf, molecule_idx, freeze_atom_string):
    from pyscf.geomopt.geometric_solver import optimize

    constraints_file = f"constraints_{molecule_idx}.txt"
    with open(constraints_file, "w") as f:
        f.write("$freeze\n")
        f.write("xyz " + freeze_atom_string)

    params = {
        "constraints": constraints_file,
        "verbose": 0,
    }

    gradients = []

    def callback(envs):
        gradients.append(envs["gradients"])

    molecule_eq = optimize(mf, maxsteps=10, callback=callback, **params)
    subprocess.run(f"rm -f {constraints_file}", shell=True)

    return molecule_eq
