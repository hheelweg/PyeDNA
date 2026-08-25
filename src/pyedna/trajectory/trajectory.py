from pathlib import Path
import json, os, tempfile
import numpy as np
import MDAnalysis as mda
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


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
        with open(mapping_file) as f:
            mapping = json.load(f)

        for item in mapping["attachments"]:
            if item["dna_residue"] == initial_residue:
                dye = item["dye"]
                return {
                    "dye": dye,
                    "amber_residue": item["amber_residue"],
                    "attach_atoms": load_attach_atoms(dye)
                }

        raise ValueError(f"No mapping found for initial residue {initial_residue}")

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
            unit="Angstrom"
        )

        if optimize_caps:
            mol = optimize_cap_geometry(mol, cap_indices)

        return mol


def validate_frame_interval(frame_interval, num_frames):
    if not isinstance(frame_interval, (list, tuple)) or len(frame_interval) != 2:
        raise ValueError("frame_interval must be [initial_frame, final_frame]")

    start, stop = frame_interval

    if not isinstance(start, int) or not isinstance(stop, int):
        raise TypeError("frame_interval values must be integers")
    if start < 0:
        raise ValueError("Initial frame cannot be negative")
    if stop < start:
        raise ValueError("Final frame must be >= initial frame")
    if stop >= num_frames:
        raise ValueError(
            f"Final frame {stop} exceeds trajectory range 0-{num_frames - 1}"
        )

    return start, stop

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
    ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)

    hydrogens = []
    for phi in (0, 2*np.pi/3, 4*np.pi/3):
        hdir = -axis/3 + np.sqrt(8/9) * (np.cos(phi)*e1 + np.sin(phi)*e2)
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


def load_config(filename):
    if tomllib is None:
        raise ImportError("tomllib/tomli is required")

    with open(filename, "rb") as f:
        return tomllib.load(f)

def load_analysis_attachments(config):
    attachments = []

    for item in config.get("attachments", []):
        if "dye" not in item or "residue" not in item:
            raise ValueError("Each [[attachments]] block requires dye and residue")

        cap = item.get("cap", "H").upper()

        if cap not in ("H", "CH3"):
            raise ValueError(
                f"Unsupported cap '{cap}' for residue {item['residue']}; use H or CH3"
            )

        attachments.append({
            "dye": item["dye"],
            "residue": item["residue"],
            "cap": cap
        })

    if not attachments:
        raise ValueError("No [[attachments]] blocks found in traj.toml")

    return attachments

def analyze_trajectory(config_file):
    cfg = load_config(config_file)
    traj_cfg = cfg["trajectory"]
    cwd = Path.cwd()

    traj = Trajectory(
        cwd / traj_cfg["topology_file"],
        cwd / traj_cfg["run_directory"] / traj_cfg["trajectory_file"]
    )

    print(f"Trajectory loaded: {traj.num_frames} frames")
    validate_frame_interval(traj_cfg["frame_interval"], traj.num_frames)

    return traj, cfg


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
        unit="Angstrom"
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
            basis=basis
        )

    return groups