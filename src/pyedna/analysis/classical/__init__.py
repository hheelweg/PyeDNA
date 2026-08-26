"""Classical analysis quantities."""

from dataclasses import dataclass

import numpy as np


DEFAULT_CLASSICAL_OUTPUTS = ["center_of_geometry"]


@dataclass(frozen=True)
class ClassicalResult:
    frame: int
    group: str
    values: dict


def run_classical_jobs(config, groups, frame):
    results = []

    for index, job in enumerate(config.get("classical", []), start=1):
        group_name = job["group"]
        if group_name not in groups:
            raise ValueError(f"[[classical]] block {index} references undefined group '{group_name}'")

        outputs = job.get("outputs", DEFAULT_CLASSICAL_OUTPUTS)
        results.append(
            ClassicalResult(
                frame=frame,
                group=group_name,
                values=classical_observables(groups[group_name], outputs),
            )
        )

    return results


def summarize_classical_result(result):
    values = ", ".join(
        f"{key}={_format_value(value)}"
        for key, value in result.values.items()
    )
    return f"Frame {result.frame}: classical group {result.group}, {values}"


def classical_observables(mol, outputs):
    values = {}
    coords = atom_coords(mol)

    for output in outputs:
        if output == "center_of_geometry":
            values[output] = center_of_geometry(coords)
        elif output == "center_of_mass":
            values[output] = center_of_mass(mol, coords)
        elif output == "radius_of_gyration":
            values[output] = radius_of_gyration(coords)
        elif output == "axis_angle":
            values[output] = axis_angle()
        else:
            raise ValueError(f"Unsupported classical output '{output}'")

    return values


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


def axis_angle(*args, **kwargs):
    raise NotImplementedError("axis_angle is reserved for a future classical orientation analysis")


def atom_masses(mol):
    if hasattr(mol, "atom_mass_list"):
        masses = mol.atom_mass_list()
    else:
        masses = [1.0 for _ in range(mol.natm)]
    return np.asarray(masses, dtype=float)


def _format_value(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{float(array):.6g}"
    return "[" + ", ".join(f"{float(item):.6g}" for item in array.ravel()) + "]"
