"""Trajectory-derived molecular structure assembly."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from pyscf import dft, gto
from pyscf.geomopt.geometric_solver import optimize


@dataclass(frozen=True)
class Fragment:
    name: str
    residue: int
    atom_indices: list


def combine_molecules(molecules, basis="6-31g", spin=0):
    atoms = []
    for mol in molecules:
        atoms.extend(
            (mol.atom_symbol(i), mol.atom_coord(i, unit="Angstrom"))
            for i in range(mol.natm)
        )

    return gto.M(
        atom=atoms,
        basis=basis,
        charge=sum(mol.charge for mol in molecules),
        spin=spin,
        unit="Angstrom",
    )


def build_groups(config, attachment_mols, basis="6-31g"):
    groups = {}

    for group in config.get("groups", []):
        name = group["name"]
        residues = group["attachments"]

        missing = [r for r in residues if r not in attachment_mols]
        if missing:
            raise ValueError(f"Group '{name}' references undefined attachment residues: {missing}")

        groups[name] = combine_molecules(
            [attachment_mols[r] for r in residues],
            basis=basis,
        )

    return groups


def build_group_fragments(config, attachment_mols):
    fragments = {}

    for group in config.get("groups", []):
        name = group["name"]
        residues = group["attachments"]
        group_fragments = []
        start = 0

        for residue in residues:
            mol = attachment_mols[residue]
            stop = start + mol.natm
            group_fragments.append(
                Fragment(
                    name=str(residue),
                    residue=residue,
                    atom_indices=list(range(start, stop)),
                )
            )
            start = stop

        fragments[name] = group_fragments

    return fragments


def load_attachment_info(initial_residue, mapping_file="resid_mapping.json"):
    with open(mapping_file) as f:
        mapping = json.load(f)

    for item in mapping["attachments"]:
        if item["dna_residue"] == initial_residue:
            dye = item["dye"]
            return {
                "dye": dye,
                "amber_residue": item["amber_residue"],
                "attach_atoms": load_attach_atoms(dye),
            }

    raise ValueError(f"No mapping found for initial residue {initial_residue}")


def infer_dye_charge(dye):
    dye_dir = os.environ.get("DYE_DIR")
    if not dye_dir:
        raise EnvironmentError("DYE_DIR is not set")

    path = Path(dye_dir) / dye / "gaff2" / f"{dye}.mol2"
    if not path.exists():
        raise FileNotFoundError(f"MOL2 file not found: {path}")

    charges, in_atoms = [], False
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if in_atoms and line.strip():
            fields = line.split()
            if len(fields) >= 9:
                charges.append(float(fields[8]))

    if not charges:
        raise ValueError(f"No atomic charges found in {path}")

    total = sum(charges)
    charge = int(round(total))
    if abs(total - charge) > 0.1:
        raise ValueError(f"{dye} MOL2 charges sum to {total:.4f}, not close to an integer")

    return charge


def load_attach_atoms(dye):
    dye_dir = os.environ.get("DYE_DIR")
    if not dye_dir:
        raise EnvironmentError("DYE_DIR is not set")

    path = Path(dye_dir) / dye / f"{dye}.attach"
    if not path.exists():
        raise FileNotFoundError(f"Attach file not found: {path}")

    atoms = [
        line.split()[1]
        for line in path.read_text().splitlines()
        if len(line.split()) == 2 and line.split()[0].upper() == "LINKER"
    ]

    if not atoms:
        raise ValueError(f"No LINKER atoms found in {path}")

    return atoms


def get_external_neighbor(atom):
    external = []

    for bond in atom.bonds:
        other = bond.atoms[0] if bond.atoms[1].index == atom.index else bond.atoms[1]
        if other.resid != atom.resid:
            external.append(other)

    if len(external) != 1:
        raise ValueError(
            f"Expected one external neighbor for {atom.name}, found {len(external)}"
        )

    return external[0]


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector")

    return vector / norm


def build_cap(attachment, external, cap_type):
    direction = unit(np.asarray(external) - np.asarray(attachment))
    cap_type = cap_type.upper()

    if cap_type == "H":
        return [("H", np.asarray(attachment) + 1.01 * direction)]

    if cap_type != "CH3":
        raise ValueError("cap_type must be 'H' or 'CH3'")

    carbon = np.asarray(attachment) + 1.47 * direction
    axis = unit(np.asarray(attachment) - carbon)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)

    hydrogens = []
    for phi in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        hdir = -axis / 3 + np.sqrt(8 / 9) * (np.cos(phi) * e1 + np.sin(phi) * e2)
        hydrogens.append(("H", carbon + 1.09 * hdir))

    return [("C", carbon), *hydrogens]


def optimize_cap_geometry(mol, cap_indices, xc="b3lyp", maxsteps=25):
    if not cap_indices:
        return mol

    first_cap = min(cap_indices)

    if cap_indices != list(range(first_cap, mol.natm)):
        raise ValueError("Cap atoms must be appended after the original dye atoms")

    mf = (dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)).density_fit()
    mf.xc = xc
    mf = mf.to_gpu()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        f.write(f"$freeze\nxyz 1-{first_cap}\n")
        f.flush()
        return optimize(mf, constraints=f.name, maxsteps=maxsteps)
