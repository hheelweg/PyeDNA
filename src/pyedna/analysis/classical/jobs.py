"""Classical analysis job dispatch."""

from dataclasses import dataclass

import numpy as np

from pyedna.analysis.classical.geometry import (
    atom_coords,
    center_of_geometry,
    center_of_mass,
    load_dye_geometry,
    plane_deviation_from_named_atoms,
    radius_of_gyration,
)


DEFAULT_CLASSICAL_OUTPUTS = ["center_of_geometry"]


@dataclass(frozen=True)
class ClassicalResult:
    frame: int
    group: str
    values: dict


def run_classical_jobs(config, groups, frame, attachment_snapshots=None):
    results = []
    attachment_snapshots = attachment_snapshots or {}

    for index, job in enumerate(config.get("classical", []), start=1):
        group_name = job["group"]
        if group_name not in groups:
            raise ValueError(f"[[classical]] block {index} references undefined group '{group_name}'")

        outputs = job.get("outputs", DEFAULT_CLASSICAL_OUTPUTS)
        results.append(
            ClassicalResult(
                frame=frame,
                group=group_name,
                values=classical_observables(
                    groups[group_name],
                    outputs,
                    group_name=group_name,
                    config=config,
                    attachment_snapshots=attachment_snapshots,
                    context=f"[[classical]] block {index}",
                ),
            )
        )

    return results


def summarize_classical_result(result):
    values = ", ".join(
        f"{key}={_format_value(value)}"
        for key, value in result.values.items()
    )
    return f"Frame {result.frame}: classical group {result.group}, {values}"


def classical_observables(
    mol,
    outputs,
    group_name=None,
    config=None,
    attachment_snapshots=None,
    context="[[classical]]",
):
    values = {}
    coords = atom_coords(mol)

    for output in outputs:
        if output == "center_of_geometry":
            values[output] = center_of_geometry(coords)
        elif output == "center_of_mass":
            values[output] = center_of_mass(mol, coords)
        elif output == "radius_of_gyration":
            values[output] = radius_of_gyration(coords)
        elif output == "plane_deviation":
            values.update(
                plane_deviation_observable(
                    group_name,
                    config,
                    attachment_snapshots or {},
                    context=context,
                )
            )
        else:
            raise ValueError(f"Unsupported classical output '{output}'")

    return values


def plane_deviation_observable(group_name, config, attachment_snapshots, context="[[classical]]"):
    snapshot = _single_attachment_snapshot(
        group_name,
        config,
        attachment_snapshots,
        "plane_deviation",
        context,
    )
    geometry = load_dye_geometry(
        snapshot.dye,
        require_plane=True,
        analysis="plane_deviation",
    )
    return plane_deviation_from_named_atoms(
        snapshot.atom_names,
        snapshot.coordinates,
        geometry.plane_atoms,
        context=f"plane_deviation for dye {snapshot.dye}",
    )


def _single_attachment_snapshot(group_name, config, attachment_snapshots, analysis, context):
    if config is None or group_name is None:
        raise ValueError(f"{context} {analysis} requires group attachment metadata")

    group_attachments = {
        group["name"]: list(group["attachments"])
        for group in config.get("groups", [])
    }
    if group_name not in group_attachments:
        raise ValueError(f"{context} references undefined group '{group_name}' for {analysis}")

    residues = group_attachments[group_name]
    if len(residues) != 1:
        raise ValueError(
            f"{context} {analysis} requires group '{group_name}' to contain exactly "
            f"one attachment; found {len(residues)}"
        )

    residue = residues[0]
    if residue not in attachment_snapshots:
        raise ValueError(
            f"{context} {analysis} could not find attachment snapshot for "
            f"residue {residue} in group '{group_name}'"
        )
    return attachment_snapshots[residue]


def _format_value(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{float(array):.6g}"
    return "[" + ", ".join(f"{float(item):.6g}" for item in array.ravel()) + "]"
