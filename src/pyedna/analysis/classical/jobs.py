"""Classical analysis job dispatch."""

from dataclasses import dataclass

import numpy as np

from pyedna.analysis.classical.geometry import (
    atom_coords,
    center_of_geometry,
    center_of_mass,
    radius_of_gyration,
)


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
        else:
            raise ValueError(f"Unsupported classical output '{output}'")

    return values


def _format_value(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{float(array):.6g}"
    return "[" + ", ".join(f"{float(item):.6g}" for item in array.ravel()) + "]"
