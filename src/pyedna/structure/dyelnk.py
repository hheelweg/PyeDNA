"""Load, validate and assemble dye/linker templates."""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import subprocess

import MDAnalysis as mda
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

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
    """Load a MOL2 template and normalize its element metadata."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown elements found for some atoms.*",
        )
        warnings.filterwarnings(
            "ignore",
            message="Failed to guess the mass.*",
        )

        u = mda.Universe(str(path))

    elements = [
        _element_from_gaff(atom_type)
        for atom_type in u.atoms.types
    ]

    # Topology attributes must be added through the Universe rather than
    # assigned directly to u.atoms.
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
    """Return a normalized vector."""
    norm = np.linalg.norm(vector)

    if norm < 1e-10:
        raise ValueError("Cannot normalize zero-length vector")

    return vector / norm


def _attachment_direction(atom):
    """Estimate outward attachment direction from bonded neighbors."""
    neighbors = atom.bonded_atoms

    if len(neighbors) == 0:
        raise ValueError(
            f"Attachment atom {atom.name} has no bonded neighbors"
        )

    return _unit(
        atom.position - neighbors.positions.mean(axis=0)
    )


def _rotate_about_axis(coords, origin, axis, angle_deg):
    """Rotate coordinates about an arbitrary axis."""
    rotation = Rotation.from_rotvec(
        np.deg2rad(angle_deg) * _unit(axis)
    )

    return rotation.apply(coords - origin) + origin


def _heavy_positions(atoms):
    """Return positions of non-hydrogen atoms."""
    mask = np.asarray([
        element != "H"
        for element in atoms.elements
    ])

    return atoms.positions[mask]


def _clash_score(moving, fixed, cutoff=2.0):
    """Return a simple heavy-atom steric overlap penalty."""
    moving_coords = _heavy_positions(moving)
    fixed_coords = _heavy_positions(fixed)

    distances = cdist(moving_coords, fixed_coords)
    overlap = np.clip(cutoff - distances, 0.0, None)

    return np.sum(overlap**2)


def _place_linker(
    linker,
    linker_atom_name,
    dye,
    dye_atom_name,
    fixed,
    bond_length=1.50,
    angle_step=10.0,
):
    """Rigidly place one linker and minimize clashes around the new bond."""
    linker_atom = _atom(linker, linker_atom_name)
    dye_atom = _atom(dye, dye_atom_name)

    dye_direction = _attachment_direction(dye_atom)
    linker_direction = _attachment_direction(linker_atom)

    # Align the linker outward direction opposite to that of the dye.
    rotation, _ = Rotation.align_vectors(
        np.array([-dye_direction]),
        np.array([linker_direction]),
    )

    coords = rotation.apply(
        linker.atoms.positions - linker_atom.position
    )

    # Position the linker attachment atom at the desired bond distance.
    target = (
        dye_atom.position
        + bond_length * dye_direction
    )

    coords += target
    linker.atoms.positions = coords

    best_coords = coords.copy()
    best_score = np.inf

    # Remaining rigid-body freedom is rotation around the prospective bond.
    for angle in np.arange(0.0, 360.0, angle_step):
        candidate = _rotate_about_axis(
            coords,
            target,
            dye_direction,
            angle,
        )

        linker.atoms.positions = candidate
        score = _clash_score(linker.atoms, fixed)

        if score < best_score:
            best_score = score
            best_coords = candidate.copy()

    linker.atoms.positions = best_coords

    return linker


def _write_combined_pdb(dye, linker3, linker5, output_file):
    """Write assembled structure in the PDB format expected by tleap."""
    output_file = Path(output_file)

    # Match the desired molecular order:
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
            element = atom.element if atom.element else _element_from_gaff(atom.type)

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

    dye_mol2: Path
    linker3_mol2: Path
    linker5_mol2: Path

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
            dye = config["dyelnk"]["dye"]
            linker = config["dyelnk"]["linker"]
        except KeyError as exc:
            raise ValueError(
                f"{path}: missing configuration field {exc}"
            ) from exc

        dye_dir = Path(os.environ["DYE_DIR"]) / dye
        lnk_dir = Path(os.environ["LNK_DIR"]) / linker

        dye_mol2 = dye_dir / f"{dye}.mol2"
        linker3_mol2 = lnk_dir / f"{linker}3.mol2"
        linker5_mol2 = lnk_dir / f"{linker}5.mol2"

        dye_attach = dye_dir / f"{dye}.attach"
        linker3_attach = lnk_dir / f"{linker}3.attach"
        linker5_attach = lnk_dir / f"{linker}5.attach"

        files = (
            dye_mol2,
            linker3_mol2,
            linker5_mol2,
            dye_attach,
            linker3_attach,
            linker5_attach,
        )

        for file in files:
            if not file.exists():
                raise FileNotFoundError(
                    f"Missing dye-linker input: {file}"
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
            dye_mol2=dye_mol2,
            linker3_mol2=linker3_mol2,
            linker5_mol2=linker5_mol2,
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

    def assemble(
        self,
        output_file=None,
        bond_length=1.50,
        angle_step=10.0,
    ):
        """Place both linkers around the dye and write a combined PDB."""
        dye = _load_mol2(self.dye_mol2)
        linker3 = _load_mol2(self.linker3_mol2)
        linker5 = _load_mol2(self.linker5_mol2)

        dye3_atom, dye5_atom = self.dye_linker_atoms

        linker3 = _place_linker(
            linker3,
            self.linker3_5connect,
            dye,
            dye3_atom,
            dye.atoms,
            bond_length=bond_length,
            angle_step=angle_step,
        )

        fixed = mda.Merge(
            dye.atoms,
            linker3.atoms,
        )

        linker5 = _place_linker(
            linker5,
            self.linker5_3connect,
            dye,
            dye5_atom,
            fixed.atoms,
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
        pdb_output=None,
    ):
        """Write tleap input for bonding the assembled dye/linker residues."""
        assembled_pdb = Path(assembled_pdb).resolve()
        output_file = Path(output_file or "tleap_dyelnk.in")
        mol2_output = Path(
            mol2_output or f"{self.dye}_{self.linker}_linked.mol2"
        )
        pdb_output = Path(
            pdb_output or f"{self.dye}_{self.linker}_linked.pdb"
        )

        if not assembled_pdb.exists():
            raise FileNotFoundError(
                f"Assembled dye-linker PDB not found: {assembled_pdb}"
            )

        res3 = _mol2_resname(self.linker3_mol2)
        res5 = _mol2_resname(self.linker5_mol2)
        resd = _mol2_resname(self.dye_mol2)

        dye3_atom, dye5_atom = self.dye_linker_atoms

        text = f"""source leaprc.gaff2

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


    def run_tleap(self, tleap_input, mol2_output, assembled_pdb=None, workdir=None):
        """Run tleap, validate success, and remove temporary assembly files."""
        tleap_input = Path(tleap_input).resolve()
        mol2_output = Path(mol2_output).resolve()
        workdir = Path(workdir or tleap_input.parent)

        if not tleap_input.exists():
            raise FileNotFoundError(f"tleap input not found: {tleap_input}")

        result = subprocess.run(
            ["tleap", "-f", str(tleap_input)],
            cwd=workdir,
            text=True,
            capture_output=True,
        )

        output = result.stdout + result.stderr
        log_file = workdir / "tleap_dyelnk.log"
        log_file.write_text(output)

        # tleap can occasionally return shell status 0 despite LEaP errors,
        # so inspect both the return code and LEaP's own summary.
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

        # Successful run: keep only the final linked MOL2.
        temporary_files = [
            tleap_input,
            log_file,
            Path(assembled_pdb) if assembled_pdb else None,
            workdir / f"{self.dye}_{self.linker}_linked.pdb",
            workdir / "leap.log",
        ]

        for path in temporary_files:
            if path and path.exists():
                path.unlink()

        return mol2_output