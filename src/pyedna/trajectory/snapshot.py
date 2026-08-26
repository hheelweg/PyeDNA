"""Trajectory snapshot extraction and PySCF molecule construction."""

from pathlib import Path

import MDAnalysis as mda
from pyscf import gto

from .structure import (
    build_cap,
    get_external_neighbor,
    infer_dye_charge,
    load_attachment_info,
    optimize_cap_geometry,
)


class Trajectory:
    def __init__(self, topology_file, trajectory_file):
        self.topology_file, self.trajectory_file = Path(topology_file), Path(trajectory_file)
        if not self.topology_file.exists():
            raise FileNotFoundError(f"Topology file not found: {self.topology_file}")
        if not self.trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_file}")

        self.universe = mda.Universe(self.topology_file, self.trajectory_file)
        self.num_frames = len(self.universe.trajectory)

    def get_attachment_info(self, initial_residue, mapping_file="resid_mapping.json"):
        return load_attachment_info(initial_residue, mapping_file=mapping_file)

    def extract_residue(self, frame, amber_residue):
        if not 0 <= frame < self.num_frames:
            raise IndexError(f"Frame {frame} outside trajectory range 0-{self.num_frames - 1}")

        self.universe.trajectory[frame]
        atoms = self.universe.select_atoms(f"resid {amber_residue}")

        if not len(atoms):
            raise ValueError(f"No atoms found for Amber residue {amber_residue}")

        return atoms

    def get_capped_snapshot(self, frame, initial_residue, dye=None, cap_type="H",
                        optimize_caps=False, basis="6-31g", spin=0):
        info = self.get_attachment_info(initial_residue)

        if dye is not None and dye != info["dye"]:
            raise ValueError(
                f"Dye mismatch at residue {initial_residue}: "
                f"traj.toml={dye}, resid_mapping.json={info['dye']}"
            )

        atoms = self.extract_residue(frame, info["amber_residue"])
        missing = [name for name in info["attach_atoms"] if name not in atoms.names]
        if missing:
            raise ValueError(f"Attachment atoms not found in residue: {missing}")

        mol_atoms = [(atom.element, atom.position.copy()) for atom in atoms]
        cap_indices = []

        for name in info["attach_atoms"]:
            selected = atoms.select_atoms(f"name {name}")
            if len(selected) != 1:
                raise ValueError(f"Expected one atom named {name}, found {len(selected)}")

            attachment = selected[0]
            external = get_external_neighbor(attachment)

            for element, position in build_cap(attachment.position, external.position, cap_type):
                cap_indices.append(len(mol_atoms))
                mol_atoms.append((element, position))

        mol = gto.M(
            atom=mol_atoms,
            basis=basis,
            charge=infer_dye_charge(info["dye"]),
            spin=spin,
            unit="Angstrom",
        )

        if optimize_caps:
            mol = optimize_cap_geometry(mol, cap_indices)

        return mol
