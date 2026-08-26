"""Transition-density-matrix coupling helpers."""

import numpy as np
import scipy.linalg
from pyscf import gto, lib


def tdm_coupling(mols, tdms, states, coupling_type="electronic"):
    return get_v_coulombic(mols, tdms, states, coupling_type=coupling_type)


def rebuild_pyscf_mol(molecule_input, dft_settings):
    return gto.M(
        atom=molecule_input,
        basis=dft_settings["basis"],
        charge=dft_settings["charge"],
        spin=dft_settings["spin"],
        unit="Angstrom",
    )


def get_inter_cj_ck(mol_a, mol_b, tdm_a, tdm_b, get_cK=False):
    from pyscf.scf import _vhf, jk

    assert tdm_a.shape == (mol_a.nao, mol_a.nao)
    assert tdm_b.shape == (mol_b.nao, mol_b.nao)

    mol_ab = mol_a + mol_b
    dm_ab = scipy.linalg.block_diag(tdm_a, tdm_b)

    vhfopt = _vhf.VHFOpt(
        mol_ab,
        "int2e",
        "CVHFnrs8_prescreen",
        "CVHFsetnr_direct_scf",
        "CVHFsetnr_direct_scf_dm",
    )
    vhfopt.set_dm(dm_ab, mol_ab._atm, mol_ab._bas, mol_ab._env)
    vhfopt._dmcondname = None

    with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer("CVHFnrs8_vj_prescreen")):
        shls_slice = (0, mol_a.nbas, 0, mol_a.nbas, mol_a.nbas, mol_ab.nbas, mol_a.nbas, mol_ab.nbas)
        vj = jk.get_jk(
            mol_ab,
            tdm_b,
            "ijkl,lk->s2ij",
            shls_slice=shls_slice,
            vhfopt=vhfopt,
            aosym="s4",
            hermi=1,
        )
        cJ = np.einsum("ia,ia->", vj, tdm_a)

    if get_cK:
        with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer("CVHFnrs8_vk_prescreen")):
            shls_slice = (0, mol_a.nbas, mol_a.nbas, mol_ab.nbas, mol_a.nbas, mol_ab.nbas, 0, mol_a.nbas)
            vk = jk.get_jk(
                mol_ab,
                tdm_b,
                "ijkl,jk->il",
                shls_slice=shls_slice,
                vhfopt=vhfopt,
                aosym="s1",
                hermi=0,
            )
            cK = np.einsum("ia,ia->", vk, tdm_a)
        return cJ, cK

    return cJ, 0


def get_intra_cj_ck(mol, tdm_a, tdm_b, get_cK=False):
    from pyscf.scf import _vhf, jk

    assert tdm_a.shape == (mol.nao, mol.nao)
    assert tdm_b.shape == (mol.nao, mol.nao)

    vhfopt = _vhf.VHFOpt(
        mol,
        "int2e",
        "CVHFnrs8_prescreen",
        "CVHFsetnr_direct_scf",
        "CVHFsetnr_direct_scf_dm",
    )
    vhfopt.set_dm(tdm_b, mol._atm, mol._bas, mol._env)
    vhfopt._dmcondname = None

    with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer("CVHFnrs8_vj_prescreen")):
        shls_slice = (0, mol.nbas, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
        vj = jk.get_jk(
            mol,
            tdm_b,
            "ijkl,lk->s2ij",
            shls_slice=shls_slice,
            vhfopt=vhfopt,
            aosym="s4",
            hermi=1,
        )
        cJ = np.einsum("ij,ij->", vj, tdm_a)

    if get_cK:
        with lib.temporary_env(vhfopt._this.contents, fprescreen=_vhf._fpointer("CVHFnrs8_vk_prescreen")):
            shls_slice = (0, mol.nbas, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            vk = jk.get_jk(
                mol,
                tdm_b,
                "ijkl,jk->il",
                shls_slice=shls_slice,
                vhfopt=vhfopt,
                aosym="s1",
                hermi=0,
            )
            cK = np.einsum("ij,ij->", vk, tdm_a)
    else:
        cK = 0.0

    return cJ, cK


def get_v_coulombic(mols, tdms, states, coupling_type="electronic"):
    state_a, state_b = states[0], states[1]
    mol_a, mol_b = mols[0], mols[1]
    tdm_a, tdm_b = tdms[0][state_a], tdms[1][state_b]

    if coupling_type in ("electronic", "cK"):
        cJ, cK = get_inter_cj_ck(mol_a, mol_b, tdm_a, tdm_b, get_cK=True)
    elif coupling_type == "cJ":
        cJ, _ = get_inter_cj_ck(mol_a, mol_b, tdm_a, tdm_b, get_cK=False)
        cK = 0
    else:
        raise NotImplementedError("Invalid coupling type specified")

    results = {"coupling cJ": cJ, "coupling cK": cK}
    if coupling_type == "electronic":
        results["coupling V_C"] = 2 * cJ - cK
    elif coupling_type == "cK":
        results["coupling V_C"] = -cK
    elif coupling_type == "cJ":
        results["coupling V_C"] = 2 * cJ
    return results
