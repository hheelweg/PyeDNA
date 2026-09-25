"""Classical geometry calculations for trajectory analysis."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyedna.config import get_config

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True)
class DyeGeometry:
    dye: str
    path: Path
    axis_atoms: tuple[str, str] | None = None
    plane_atoms: tuple[str, ...] | None = None


def atom_coords(mol):
    return np.asarray(
        [mol.atom_coord(i, unit="Angstrom") for i in range(mol.natm)],
        dtype=float,
    )


def center_of_geometry(coords):
    return np.mean(coords, axis=0)


def center_of_mass(mol, coords):
    masses = atom_masses(mol)
    return np.average(coords, axis=0, weights=masses)


def radius_of_gyration(coords):
    center = center_of_geometry(coords)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def distance_between_groups(group_a, group_b, method="center_of_geometry"):
    coords_a = atom_coords(group_a)
    coords_b = atom_coords(group_b)

    if method == "center_of_geometry":
        point_a = center_of_geometry(coords_a)
        point_b = center_of_geometry(coords_b)
    elif method == "center_of_mass":
        point_a = center_of_mass(group_a, coords_a)
        point_b = center_of_mass(group_b, coords_b)
    else:
        raise ValueError("Distance method must be center_of_geometry or center_of_mass")

    return float(np.linalg.norm(point_a - point_b))


def axis_from_named_atoms(atom_names, coords, axis_atoms, context="axis"):
    selected = coordinates_for_named_atoms(atom_names, coords, axis_atoms, context=context)
    return unit_vector(selected[1] - selected[0])


def plane_normal_from_named_atoms(atom_names, coords, plane_atoms, context="plane"):
    selected = coordinates_for_named_atoms(atom_names, coords, plane_atoms, context=context)
    return best_fit_plane_normal(selected)


def axis_angle(axis_a, axis_b):
    dot = float(np.dot(unit_vector(axis_a), unit_vector(axis_b)))
    dot = np.clip(abs(dot), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def orientation_factor(mu_donor, mu_acceptor, donor_to_acceptor, tolerance=1e-10):
    mu_donor = unit_vector(mu_donor)
    mu_acceptor = unit_vector(mu_acceptor)
    r_hat = unit_vector(donor_to_acceptor)

    kappa = float(
        np.dot(mu_donor, mu_acceptor)
        - 3.0 * np.dot(mu_donor, r_hat) * np.dot(mu_acceptor, r_hat)
    )
    kappa_squared = float(kappa * kappa)

    if kappa_squared < -tolerance or kappa_squared > 4.0 + tolerance:
        raise ValueError(
            "Squared orientation factor is outside the physical range "
            f"[0, 4]: {kappa_squared}"
        )

    return {"kappa": kappa, "kappa_squared": kappa_squared}


def best_fit_plane_normal(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 3:
        raise ValueError("Plane construction requires at least three atoms")

    centered = coords - np.mean(coords, axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("Cannot construct a plane normal from collinear coordinates")

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return unit_vector(vh[-1])


def coordinates_for_named_atoms(atom_names, coords, requested_names, context="geometry"):
    atom_names = list(atom_names)
    coords = np.asarray(coords, dtype=float)
    selected = []

    for name in requested_names:
        matches = [index for index, atom_name in enumerate(atom_names) if atom_name == name]
        if len(matches) != 1:
            raise ValueError(
                f"{context} requires atom '{name}' to exist exactly once; "
                f"found {len(matches)}"
            )
        selected.append(coords[matches[0]])

    return np.asarray(selected, dtype=float)


def load_dye_geometry(dye, dye_dir=None, require_axis=False, analysis="axis_angle"):
    dye_dir = Path(get_config().libraries.dye_dir if dye_dir is None else dye_dir)
    path = dye_dir / dye / "geometry.toml"

    if not path.exists():
        if require_axis:
            raise FileNotFoundError(
                f"{analysis} requires geometry metadata for dye {dye}, but "
                f"{path} was not found.\n\n"
                "geometry.toml is optional for normal PyeDNA workflows, but "
                "required for this requested analysis. Add geometry.toml manually "
                "to the dye-library entry and define:\n\n"
                "[axis]\n"
                'atoms = ["ATOM1", "ATOM2"]'
            )
        return DyeGeometry(dye=dye, path=path)

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    axis_atoms = _optional_atom_list(data, "axis", exact=2, path=path)
    plane_atoms = _optional_atom_list(data, "plane", minimum=3, path=path)

    if require_axis and axis_atoms is None:
        raise ValueError(
            f"{analysis} requires a valid [axis] block for dye {dye}, but "
            f"{path} does not define one.\n\n"
            "Add:\n\n"
            "[axis]\n"
            'atoms = ["ATOM1", "ATOM2"]'
        )

    return DyeGeometry(
        dye=dye,
        path=path,
        axis_atoms=tuple(axis_atoms) if axis_atoms is not None else None,
        plane_atoms=tuple(plane_atoms) if plane_atoms is not None else None,
    )


def atom_masses(mol):
    if hasattr(mol, "atom_mass_list"):
        masses = mol.atom_mass_list()
    else:
        masses = [1.0 for _ in range(mol.natm)]
    return np.asarray(masses, dtype=float)


def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


def _optional_atom_list(data, section, exact=None, minimum=None, path=None):
    table = data.get(section)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise TypeError(f"[{section}] in {path} must be a table")

    atoms = table.get("atoms")
    if not isinstance(atoms, list) or not all(
        isinstance(atom, str) and atom for atom in atoms
    ):
        raise TypeError(f"[{section}].atoms in {path} must be a list of non-empty strings")

    if exact is not None and len(atoms) != exact:
        raise ValueError(f"[{section}].atoms in {path} must contain exactly {exact} atoms")
    if minimum is not None and len(atoms) < minimum:
        raise ValueError(f"[{section}].atoms in {path} must contain at least {minimum} atoms")

    return atoms
