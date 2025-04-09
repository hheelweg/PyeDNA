import numpy as np


# mulliken population analysis (populations/charges) per atom in mol
def mulliken_pop(mol, s, dm):
    nao = mol.nao
    natm = mol.natm

    # (1) AO populations (contract density with overlap)
    ao_pops = np.einsum('ij,ji->i', dm, s).real   # shape: (nao,)

    # (2) Map AO index → atom index
    ao2atom = np.array([label[0] for label in mol.ao_labels(fmt=None)])

    # (3) Sum AO populations per atom
    atom_pops = np.zeros(natm)
    for ao_idx in range(nao):
        atom_idx = ao2atom[ao_idx]
        atom_pops[atom_idx] += ao_pops[ao_idx]

    # (4) Mulliken charges: Z - N
    atom_charges = mol.atom_charges() - atom_pops

    return atom_pops, atom_charges