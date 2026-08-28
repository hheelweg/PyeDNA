"""Load, validate and assemble dye/linker templates."""

import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from pyedna.config import amber_environment, amber_executable, get_config

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _read_attach(path):
    """Read attachment labels and atom names from an .attach file."""
    data = {}

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}: invalid attachment line: {line}")

        label, atom = fields
        data.setdefault(label, []).append(atom)

    return data


def _mol2_resname(path):
    """Return the unique residue name stored in a MOL2 template."""
    u = _load_mol2(path)
    names = set(u.atoms.resnames)

    if len(names) != 1:
        raise ValueError(
            f"{path}: expected exactly one residue name, found {sorted(names)}"
        )

    return next(iter(names))


def _mol2_frcmod(path):
    """Return corresponding frcmod file."""
    frcmod = Path(path).with_suffix(".frcmod")

    if not frcmod.exists():
        raise FileNotFoundError(f"Missing frcmod parameter file: {frcmod}")

    return frcmod


def forcefield_id(forcefield):
    """Return the library-directory identifier for an Amber forcefield."""
    value = str(forcefield)
    if value.startswith("leaprc.DNA."):
        return value.removeprefix("leaprc.DNA.")
    if value.startswith("leaprc."):
        return value.removeprefix("leaprc.")
    return value


def tleap_source(forcefield, family):
    """Return the tleap source file for a compact forcefield identifier."""
    value = str(forcefield)
    if value.startswith("leaprc."):
        return value
    if family == "dna":
        return f"leaprc.DNA.{value}"
    if family == "water":
        return f"leaprc.water.{value}"
    return f"leaprc.{value}"


def resolve_connect_frcmod(lnk_root, linker_forcefield, dna_forcefield):
    """Return shared DNA-linker compatibility parameters for a forcefield pair."""
    connect_dir = Path(lnk_root) / "connect" / linker_forcefield / dna_forcefield
    canonical = connect_dir / "connectparams.frcmod"
    legacy = connect_dir / "connectparms.frcmod"

    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "Missing DNA-linker compatibility parameters.\n\n"
        "The requested structure requires:\n"
        f"linker forcefield: {linker_forcefield}\n"
        f"DNA forcefield: {dna_forcefield}\n\n"
        f"Expected compatibility file:\n{canonical}\n\n"
        "This file contains manually curated Amber parameters connecting "
        f"DNA {dna_forcefield} residues with {linker_forcefield} linker residues.\n\n"
        "Please add the required bond/angle/dihedral terms manually to this "
        "connectparams.frcmod file before proceeding."
    )


def _element_from_gaff(atom_type):
    """Infer chemical element from an Amber/GAFF atom type."""
    atom_type = atom_type.lower()

    if atom_type.startswith("cl"):
        return "Cl"
    if atom_type.startswith("br"):
        return "Br"

    return {
        "c": "C",
        "h": "H",
        "n": "N",
        "o": "O",
        "p": "P",
        "s": "S",
        "f": "F",
        "i": "I",
    }.get(atom_type[0], "")


def _load_mol2(path):
    """Load a MOL2 template and normalize element metadata."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown elements found for some atoms.*")
        warnings.filterwarnings("ignore", message="Failed to guess the mass.*")
        u = mda.Universe(str(path))

    elements = [_element_from_gaff(atom_type) for atom_type in u.atoms.types]
    u.add_TopologyAttr("elements", elements)

    return u


def _mol2_atom_names(path):
    """Return atom names defined in a MOL2 file."""
    return set(_load_mol2(path).atoms.names)


def _validate_attach(mol2, attach, expected):
    """Validate attachment labels and referenced MOL2 atoms."""
    data = _read_attach(attach)

    if set(data) != set(expected):
        raise ValueError(
            f"{attach}: expected attachment labels {sorted(expected)}, "
            f"found {sorted(data)}"
        )

    for label, count in expected.items():
        if len(data[label]) != count:
            raise ValueError(
                f"{attach}: expected {count} {label} entry/entries, "
                f"found {len(data[label])}"
            )

    names = _mol2_atom_names(mol2)
    missing = [
        atom
        for atoms in data.values()
        for atom in atoms
        if atom not in names
    ]

    if missing:
        raise ValueError(
            f"{attach}: attachment atoms not found in {mol2}: {missing}"
        )

    return data


def _atom(universe, name):
    """Return one uniquely named atom."""
    atoms = universe.select_atoms(f"name {name}")

    if len(atoms) != 1:
        raise ValueError(
            f"{universe.filename}: expected one atom named {name}, "
            f"found {len(atoms)}"
        )

    return atoms[0]


def _unit(vector):
    """Return normalized vector."""
    norm = np.linalg.norm(vector)

    if norm < 1e-10:
        raise ValueError("Cannot normalize zero-length vector")

    return vector / norm


def _attachment_direction(universe, atom_name, coords=None):
    """Estimate outward direction from an attachment atom and its neighbors."""
    atom = _atom(universe, atom_name)
    coords = universe.atoms.positions if coords is None else coords
    neighbors = atom.bonded_atoms.indices

    if len(neighbors) == 0:
        raise ValueError(f"Attachment atom {atom.name} has no bonded neighbors")

    return _unit(coords[atom.index] - coords[neighbors].mean(axis=0))


def _rotate_about_axis(coords, origin, axis, angle_deg):
    """Rotate coordinates around an arbitrary axis."""
    rotation = Rotation.from_rotvec(np.deg2rad(angle_deg) * _unit(axis))
    return rotation.apply(coords - origin) + origin


def _clash_score(moving, fixed, moving_coords=None, fixed_coords=None,
                 exclude_pair=None, cutoff=2.0):
    """Return heavy-atom intermolecular overlap penalty."""
    moving_coords = (
        moving.positions if moving_coords is None else np.asarray(moving_coords)
    )
    fixed_coords = (
        fixed.positions if fixed_coords is None else np.asarray(fixed_coords)
    )

    distances = cdist(moving_coords, fixed_coords)

    moving_heavy = np.asarray(moving.elements) != "H"
    fixed_heavy = np.asarray(fixed.elements) != "H"
    mask = moving_heavy[:, None] & fixed_heavy[None, :]

    # The intended new covalent bond is ~1.5 Å and must not count as a clash.
    if exclude_pair is not None:
        moving_idx, fixed_idx = exclude_pair
        mask[moving_idx, fixed_idx] = False

    overlap = np.clip(cutoff - distances, 0.0, None)
    return float(np.sum((overlap[mask]) ** 2))



def _generate_linker_conformers(linker, n_conformers=20, seed=7):
    """
    Generate linker conformers using RDKit.

    RDKit is only used for geometry generation.
    The Amber/GAFF template remains the source of truth.

    Only coordinates of the original Amber atoms are returned.
    """

    rw = Chem.RWMol()

    # Store the number of Amber atoms before adding temporary H
    n_atoms = len(linker.atoms)

    # Preserve original Amber atom ordering
    for atom in linker.atoms:
        rw.AddAtom(
            Chem.Atom(atom.element)
        )

    # Preserve connectivity
    for bond in linker.bonds:
        rw.AddBond(
            int(bond.indices[0]),
            int(bond.indices[1]),
            Chem.BondType.SINGLE,
        )

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    # Add temporary hydrogens for MMFF
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5

    conf_ids = AllChem.EmbedMultipleConfs(
        mol_h,
        numConfs=n_conformers,
        params=params,
    )

    if not conf_ids:
        raise RuntimeError(
            "RDKit failed to generate linker conformers"
        )

    try:
        energies = AllChem.MMFFOptimizeMoleculeConfs(
            mol_h,
            maxIters=200,
        )
    except Exception:
        energies = [(0, 0)] * len(conf_ids)

    conformers = []

    for conf_id, (_, energy) in zip(conf_ids, energies):

        conf = mol_h.GetConformer(conf_id)

        # Only keep coordinates belonging to original Amber atoms.
        # RDKit appends added hydrogens after these atoms.
        coords = np.array(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(n_atoms)
            ]
        )

        if coords.shape != (n_atoms, 3):
            raise RuntimeError(
                "RDKit coordinate mapping failed"
            )

        conformers.append(
            (
                coords,
                float(energy),
            )
        )

    return conformers


def _place_conformer(linker, coords, linker_atom_name, dye, dye_atom_name,
                     fixed, bond_length=1.50, angle_step=10.0):
    """Place one linker conformer and find its best axial orientation."""
    linker_atom = _atom(linker, linker_atom_name)
    dye_atom = _atom(dye, dye_atom_name)

    linker_idx = linker_atom.index
    dye_idx = dye_atom.index

    dye_direction = _attachment_direction(dye, dye_atom_name)
    linker_direction = _attachment_direction(
        linker, linker_atom_name, coords=coords
    )

    # Outward directions must oppose one another across the new bond.
    rotation, _ = Rotation.align_vectors(
        np.array([-dye_direction]),
        np.array([linker_direction]),
    )

    placed = rotation.apply(coords - coords[linker_idx])

    target = dye_atom.position + bond_length * dye_direction
    placed += target

    best_coords = None
    best_score = np.inf

    for angle in np.arange(0.0, 360.0, angle_step):
        candidate = _rotate_about_axis(
            placed,
            target,
            dye_direction,
            angle,
        )

        score = _clash_score(
            linker.atoms,
            fixed,
            moving_coords=candidate,
            exclude_pair=(linker_idx, dye_idx),
        )

        if score < best_score:
            best_score = score
            best_coords = candidate.copy()

    return best_coords, best_score


def _select_linker_pair(
    dye,
    linker3,
    linker5,
    dye3_atom,
    dye5_atom,
    linker3_atom,
    linker5_atom,
    n_conformers=20,
    bond_length=1.50,
    angle_step=10.0,
):
    """Select the globally lowest-clash DE3/DE5 conformer combination."""
    conformers3 = _generate_linker_conformers(
        linker3,
        n_conformers=n_conformers,
        seed=7,
    )
    conformers5 = _generate_linker_conformers(
        linker5,
        n_conformers=n_conformers,
        seed=17,
    )

    best = None
    best_key = (np.inf, np.inf)

    for coords3, energy3 in conformers3:
        placed3, clash3 = _place_conformer(
            linker3,
            coords3,
            linker3_atom,
            dye,
            dye3_atom,
            dye.atoms,
            bond_length=bond_length,
            angle_step=angle_step,
        )

        linker3.atoms.positions = placed3

        # Dye is deliberately first so its atom indices remain unchanged.
        fixed = mda.Merge(dye.atoms, linker3.atoms)

        for coords5, energy5 in conformers5:
            placed5, clash5 = _place_conformer(
                linker5,
                coords5,
                linker5_atom,
                dye,
                dye5_atom,
                fixed.atoms,
                bond_length=bond_length,
                angle_step=angle_step,
            )

            # clash5 contains both:
            #   DE5 <-> dye
            #   DE5 <-> DE3
            # while clash3 contains DE3 <-> dye.
            total_clash = clash3 + clash5
            total_energy = energy3 + energy5

            # Sterics dominate. Conformer MMFF/UFF energy only breaks ties.
            key = (total_clash, total_energy)

            if key < best_key:
                best_key = key
                best = (placed3.copy(), placed5.copy())

    if best is None:
        raise RuntimeError("Could not find a valid linker conformer pair")

    linker3.atoms.positions = best[0]
    linker5.atoms.positions = best[1]

    return linker3, linker5


def _write_combined_pdb(dye, linker3, linker5, output_file):
    """Write assembled structure in the PDB format expected by tleap."""
    output_file = Path(output_file)

    # Molecular order in the finished construct:
    # 5' linker -> dye -> 3' linker
    components = [
        (linker5, 1),
        (dye, 2),
        (linker3, 3),
    ]

    lines = []
    serial = 1

    for universe, resid in components:
        atoms = universe.atoms
        resname = atoms.residues[0].resname

        for atom in atoms:
            x, y, z = atom.position
            element = (
                atom.element
                if atom.element
                else _element_from_gaff(atom.type)
            )

            lines.append(
                f"HETATM{serial:5d} {atom.name:>4s} "
                f"{resname:>3s} B{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}  "
            )

            serial += 1

    lines.extend(["TER", "END"])
    output_file.write_text("\n".join(lines) + "\n")

    return output_file


@dataclass(frozen=True)
class DyeLinkerConfig:
    dye: str
    linker: str
    dye_forcefield: str
    dna_forcefield: str

    dye_mol2: Path
    linker3_mol2: Path
    linker5_mol2: Path
    linker_connect_frcmod: Path

    dye_attach: Path
    linker3_attach: Path
    linker5_attach: Path

    dye_linker_atoms: tuple[str, str]

    linker3_3connect: str
    linker3_5connect: str

    linker5_3connect: str
    linker5_5connect: str

    @classmethod
    def from_file(cls, path):
        """Load dye/linker templates and final MOL2 attachment atoms."""
        path = Path(path)

        with path.open("rb") as handle:
            config = tomllib.load(handle)

        try:
            dyelnk = config["dyelnk"]
            dye = dyelnk["dye"]
            linker = dyelnk["linker"]
        except KeyError as exc:
            raise ValueError(
                f"{path}: missing configuration field {exc}"
            ) from exc

        return cls.from_names(
            dye,
            linker,
            dye_forcefield=dyelnk.get("dye_forcefield", "gaff2"),
            dna_forcefield=dyelnk.get("dna_forcefield", "OL15"),
        )

    @classmethod
    def from_names(
        cls,
        dye,
        linker,
        dye_forcefield="gaff2",
        dna_forcefield="OL15",
        dye_dir=None,
        lnk_dir=None,
    ):
        """Load dye/linker templates by name from DYE_DIR and LNK_DIR."""
        dye_ff = forcefield_id(dye_forcefield)
        dna_ff = forcefield_id(dna_forcefield)
        config = get_config()
        dye_root = dye_dir or config.libraries.dye_dir
        lnk_root = lnk_dir or config.libraries.linker_dir

        dye_directory = Path(dye_root) / dye
        dye_dir = dye_directory / dye_ff
        lnk_dir = Path(lnk_root) / linker / dye_ff / dna_ff

        dye_mol2 = dye_dir / f"{dye}.mol2"
        linker3_mol2 = lnk_dir / f"{linker}3.mol2"
        linker5_mol2 = lnk_dir / f"{linker}5.mol2"
        linker_connect_frcmod = resolve_connect_frcmod(lnk_root, dye_ff, dna_ff)

        dye_attach = dye_directory / f"{dye}.attach"
        linker3_attach = lnk_dir / f"{linker}3.attach"
        linker5_attach = lnk_dir / f"{linker}5.attach"

        files = (
            dye_mol2,
            linker3_mol2,
            linker5_mol2,
            linker_connect_frcmod,
            dye_attach,
            linker3_attach,
            linker5_attach,
        )

        missing = [str(file) for file in files if not file.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing dye-linker inputs:\n  " + "\n  ".join(missing)
            )

        dye_data = _validate_attach(
            dye_mol2,
            dye_attach,
            {"LINKER": 2},
        )

        linker3_data = _validate_attach(
            linker3_mol2,
            linker3_attach,
            {"3CONNECT": 1, "5CONNECT": 1},
        )

        linker5_data = _validate_attach(
            linker5_mol2,
            linker5_attach,
            {"3CONNECT": 1, "5CONNECT": 1},
        )

        return cls(
            dye=dye,
            linker=linker,
            dye_forcefield=dye_ff,
            dna_forcefield=dna_ff,
            dye_mol2=dye_mol2,
            linker3_mol2=linker3_mol2,
            linker5_mol2=linker5_mol2,
            linker_connect_frcmod=linker_connect_frcmod,
            dye_attach=dye_attach,
            linker3_attach=linker3_attach,
            linker5_attach=linker5_attach,
            dye_linker_atoms=tuple(dye_data["LINKER"]),
            linker3_3connect=linker3_data["3CONNECT"][0],
            linker3_5connect=linker3_data["5CONNECT"][0],
            linker5_3connect=linker5_data["3CONNECT"][0],
            linker5_5connect=linker5_data["5CONNECT"][0],
        )

    @property
    def dye_bonds(self):
        """Return the two bonds joining linker3-dye-linker5."""
        dye3, dye5 = self.dye_linker_atoms

        return (
            (
                self.linker3_mol2,
                self.linker3_5connect,
                self.dye_mol2,
                dye3,
            ),
            (
                self.dye_mol2,
                dye5,
                self.linker5_mol2,
                self.linker5_3connect,
            ),
        )

    @property
    def exposed_connections(self):
        """Return ports retained for later DNA attachment."""
        return {
            "3CONNECT": (
                self.linker3_mol2,
                self.linker3_3connect,
            ),
            "5CONNECT": (
                self.linker5_mol2,
                self.linker5_5connect,
            ),
        }

    def structure_attachment_records(self):
        """Return DNA-facing attachment atoms from linker metadata."""
        res5 = _mol2_resname(self.linker5_mol2)
        res3 = _mol2_resname(self.linker3_mol2)

        # The 5'/3' keys here follow the legacy structure workflow convention:
        # 5' connects to the previous DNA O3', and 3' connects to the next DNA P.
        # They are not the same as the linker's XCONNECT labels.
        return {
            "5'": (res3, 3, self.linker3_3connect),
            "3'": (res5, 1, self.linker5_5connect),
        }

    def build_linked_mol2(self, workdir=".", name=None, n_conformers=20):
        """Run the existing assembly and tleap workflow for one dye-linker."""
        workdir = Path(workdir)
        default_name = f"{self.dye}_{self.linker}"
        tleap_input = workdir / (
            f"tleap_{name}.in" if name is not None else "tleap_dyelnk.in"
        )
        name = name or default_name

        assembled_pdb = self.assemble(
            workdir / f"{name}_assembled.pdb",
            n_conformers=n_conformers,
        )
        mol2_output = workdir / f"{name}_linked.mol2"

        self.write_tleap_input(
            assembled_pdb,
            output_file=tleap_input,
            mol2_output=mol2_output,
        )
        mol2_output = self.run_tleap(
            tleap_input,
            mol2_output,
            assembled_pdb=assembled_pdb,
            workdir=workdir,
        )
        self.build_linked_frcmod(mol2_output, workdir=workdir)
        return mol2_output

    def build_linked_frcmod(self, mol2_file, output_file=None, workdir=None):
        """Generate missing GAFF2 parameters for the combined dye-linker MOL2."""
        mol2_file = Path(mol2_file).resolve()
        output_file = Path(output_file or mol2_file.with_suffix(".frcmod")).resolve()
        workdir = Path(workdir or output_file.parent)

        if not mol2_file.exists():
            raise FileNotFoundError(f"Linked dye-linker MOL2 not found: {mol2_file}")

        result = subprocess.run(
            [
                str(amber_executable("parmchk2")),
                "-i", str(mol2_file),
                "-f", "mol2",
                "-o", str(output_file),
                "-s", self.dye_forcefield,
            ],
            cwd=workdir,
            text=True,
            capture_output=True,
            env=amber_environment(),
        )

        output = result.stdout + result.stderr
        log_file = output_file.with_suffix(".parmchk2.log")
        log_file.write_text(output)

        if result.returncode != 0 or not output_file.exists():
            raise RuntimeError(
                f"parmchk2 failed. See:\n"
                f"  input: {mol2_file}\n"
                f"  log:   {log_file}"
            )

        return output_file

    def assemble(
        self,
        output_file=None,
        n_conformers=20,
        bond_length=1.50,
        angle_step=10.0,
    ):
        """Generate linker conformers, place both around dye, and write PDB."""
        dye = _load_mol2(self.dye_mol2)
        linker3 = _load_mol2(self.linker3_mol2)
        linker5 = _load_mol2(self.linker5_mol2)

        dye3_atom, dye5_atom = self.dye_linker_atoms

        linker3, linker5 = _select_linker_pair(
            dye=dye,
            linker3=linker3,
            linker5=linker5,
            dye3_atom=dye3_atom,
            dye5_atom=dye5_atom,
            linker3_atom=self.linker3_5connect,
            linker5_atom=self.linker5_3connect,
            n_conformers=n_conformers,
            bond_length=bond_length,
            angle_step=angle_step,
        )

        output_file = Path(
            output_file
            or f"{self.dye}_{self.linker}_assembled.pdb"
        )

        return _write_combined_pdb(
            dye,
            linker3,
            linker5,
            output_file,
        )

    def write_tleap_input(
        self,
        assembled_pdb,
        output_file=None,
        mol2_output=None,
    ):
        """Write tleap input for bonding assembled dye/linker residues."""
        assembled_pdb = Path(assembled_pdb).resolve()
        output_file = Path(output_file or "tleap_dyelnk.in")
        mol2_output = Path(
            mol2_output or f"{self.dye}_{self.linker}_linked.mol2"
        )

        if not assembled_pdb.exists():
            raise FileNotFoundError(
                f"Assembled dye-linker PDB not found: {assembled_pdb}"
            )

        res3 = _mol2_resname(self.linker3_mol2)
        res5 = _mol2_resname(self.linker5_mol2)
        resd = _mol2_resname(self.dye_mol2)

        frc3 = _mol2_frcmod(self.linker3_mol2)
        frc5 = _mol2_frcmod(self.linker5_mol2)
        frcd = _mol2_frcmod(self.dye_mol2)
        frc_connect = self.linker_connect_frcmod

        dye3_atom, dye5_atom = self.dye_linker_atoms

        text = f"""source {tleap_source(self.dye_forcefield, "small")}

loadAmberParams "{frc3}"
loadAmberParams "{frc5}"
loadAmberParams "{frcd}"
loadAmberParams "{frc_connect}"

{res3} = loadMol2 "{self.linker3_mol2}"
{res5} = loadMol2 "{self.linker5_mol2}"
{resd} = loadMol2 "{self.dye_mol2}"

dyelnk = loadPdb "{assembled_pdb}"

bond dyelnk.1.{self.linker5_3connect} dyelnk.2.{dye5_atom}
bond dyelnk.2.{dye3_atom} dyelnk.3.{self.linker3_5connect}

check dyelnk
charge dyelnk

saveMol2 dyelnk "{mol2_output}" 1

quit
"""

        output_file.write_text(text)
        return output_file

    def run_tleap(
        self,
        tleap_input,
        mol2_output,
        assembled_pdb=None,
        workdir=None,
    ):
        """Run tleap, validate success, and remove temporary assembly files."""
        tleap_input = Path(tleap_input).resolve()
        mol2_output = Path(mol2_output).resolve()
        workdir = Path(workdir or tleap_input.parent)

        if not tleap_input.exists():
            raise FileNotFoundError(
                f"tleap input not found: {tleap_input}"
            )

        result = subprocess.run(
            [str(amber_executable("tleap")), "-f", str(tleap_input)],
            cwd=workdir,
            text=True,
            capture_output=True,
            env=amber_environment(),
        )

        output = result.stdout + result.stderr
        log_file = workdir / "tleap_dyelnk.log"
        log_file.write_text(output)

        leap_failed = (
            result.returncode != 0
            or "FATAL:" in output
            or "Exiting LEaP: Errors = 0" not in output
            or not mol2_output.exists()
        )

        if leap_failed:
            raise RuntimeError(
                f"tleap failed. See:\n"
                f"  input: {tleap_input}\n"
                f"  log:   {log_file}"
            )

        temporary_files = [
            tleap_input,
            log_file,
            Path(assembled_pdb) if assembled_pdb else None,
            workdir / "leap.log",
        ]

        for path in temporary_files:
            if path and path.exists():
                path.unlink()

        return mol2_output
