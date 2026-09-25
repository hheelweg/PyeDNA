"""Interaction analysis between analysis groups."""

from dataclasses import dataclass

import numpy as np

from pyedna.analysis.classical import (
    atom_coords,
    axis_angle,
    axis_from_named_atoms,
    center_of_geometry,
    center_of_mass,
    distance_between_groups,
    load_dye_geometry,
    orientation_factor,
)


@dataclass(frozen=True)
class InteractionResult:
    frame: int
    type: str
    method: str
    groups: list
    state_pair: list
    values: dict


def run_quantum_interactions(config, quantum_results):
    interactions = config.get("quantum_interactions", [])
    if not interactions:
        return []

    quantum_by_group = {result.group: result for result in quantum_results}
    results = []

    for index, interaction in enumerate(interactions, start=1):
        if interaction["type"] != "coupling":
            continue

        results.extend(
            run_coupling_interaction(
                interaction,
                quantum_by_group,
                context=f"[[quantum_interactions]] block {index}",
            )
        )

    return results


def run_classical_interactions(config, groups, frame, attachment_snapshots=None):
    interactions = config.get("classical_interactions", [])
    if not interactions:
        return []

    attachment_snapshots = attachment_snapshots or {}
    results = []
    for index, interaction in enumerate(interactions, start=1):
        context = f"[[classical_interactions]] block {index}"
        if interaction["type"] == "distance":
            results.append(
                run_distance_interaction(
                    interaction,
                    groups,
                    frame,
                    context=context,
                )
            )
        elif interaction["type"] == "axis_angle":
            results.append(
                run_axis_angle_interaction(
                    interaction,
                    config,
                    attachment_snapshots,
                    frame,
                    context=context,
                )
            )
        elif interaction["type"] == "orientation_factor":
            results.append(
                run_orientation_factor_interaction(
                    interaction,
                    config,
                    groups,
                    attachment_snapshots,
                    frame,
                    context=context,
                )
            )

    return results


def run_interactions(config, quantum_results, groups=None, frame=None, attachment_snapshots=None):
    return [
        *run_quantum_interactions(config, quantum_results),
        *run_classical_interactions(
            config,
            groups or {},
            frame,
            attachment_snapshots=attachment_snapshots,
        ),
    ]


def run_distance_interaction(interaction, groups, frame, context="[[interactions]]"):
    group_names = interaction["groups"]
    missing = [group for group in group_names if group not in groups]
    if missing:
        raise ValueError(f"{context} references undefined groups for distance: {missing}")

    method = interaction.get("method", "center_of_geometry")
    value = distance_between_groups(groups[group_names[0]], groups[group_names[1]], method=method)

    return InteractionResult(
        frame=frame,
        type=interaction["type"],
        method=method,
        groups=group_names,
        state_pair=None,
        values={"distance": value},
    )


def run_axis_angle_interaction(
    interaction,
    config,
    attachment_snapshots,
    frame,
    context="[[interactions]]",
):
    group_names = interaction["groups"]
    snapshots = _single_attachment_snapshots(
        config,
        group_names,
        attachment_snapshots,
        "axis_angle",
        context,
    )
    axes = _snapshot_axes(snapshots, "axis_angle")

    return InteractionResult(
        frame=frame,
        type=interaction["type"],
        method=interaction.get("method", "axis"),
        groups=group_names,
        state_pair=None,
        values={"axis_angle": axis_angle(axes[0], axes[1])},
    )


def run_orientation_factor_interaction(
    interaction,
    config,
    groups,
    attachment_snapshots,
    frame,
    context="[[interactions]]",
):
    group_names = interaction["groups"]
    missing = [group for group in group_names if group not in groups]
    if missing:
        raise ValueError(f"{context} references undefined groups for orientation_factor: {missing}")

    snapshots = _single_attachment_snapshots(
        config,
        group_names,
        attachment_snapshots,
        "orientation_factor",
        context,
    )
    axes = _snapshot_axes(snapshots, "orientation_factor")
    method = interaction.get("method", "center_of_geometry")
    centers = [_group_center(groups[group], method) for group in group_names]
    donor_to_acceptor = centers[1] - centers[0]

    if np.linalg.norm(donor_to_acceptor) == 0:
        raise ValueError(
            f"{context} orientation_factor requires distinct donor and acceptor "
            "centers; the selected group centers coincide"
        )

    return InteractionResult(
        frame=frame,
        type=interaction["type"],
        method=method,
        groups=group_names,
        state_pair=None,
        values=orientation_factor(axes[0], axes[1], donor_to_acceptor),
    )


def run_coupling_interaction(interaction, quantum_by_group, context="[[interactions]]"):
    from pyedna.analysis.quantum.couplings import tdm_coupling

    groups = interaction["groups"]
    missing = [group for group in groups if group not in quantum_by_group]
    if missing:
        raise ValueError(f"{context} references groups without quantum results: {missing}")

    result_a = quantum_by_group[groups[0]]
    result_b = quantum_by_group[groups[1]]
    _require_tdm(result_a, context)
    _require_tdm(result_b, context)

    mols = [_rebuild_mol(result_a), _rebuild_mol(result_b)]
    tdms = [result_a.tddft["tdm"], result_b.tddft["tdm"]]
    coupling_type = interaction.get("coupling_type", "electronic")

    output = []
    for state_pair in interaction.get("state_pairs", [[0, 0]]):
        states = [
            _resolve_state(state_pair[0], result_a),
            _resolve_state(state_pair[1], result_b),
        ]
        values = tdm_coupling(mols, tdms, states, coupling_type=coupling_type)
        output.append(
            InteractionResult(
                frame=result_a.frame,
                type=interaction["type"],
                method=interaction.get("method", "tdm"),
                groups=groups,
                state_pair=state_pair,
                values=values,
            )
        )

    return output


def summarize_interaction_result(result):
    values = ", ".join(
        f"{key}={float(value):.6g}"
        for key, value in result.values.items()
    )
    return (
        f"Frame {result.frame}: interaction {result.type} "
        f"{result.groups} state_pair={result.state_pair}, {values}"
    )


def _group_center(group, method):
    coords = atom_coords(group)
    if method == "center_of_geometry":
        return center_of_geometry(coords)
    if method == "center_of_mass":
        return center_of_mass(group, coords)
    raise ValueError("Center method must be center_of_geometry or center_of_mass")


def _single_attachment_snapshots(config, group_names, attachment_snapshots, analysis, context):
    group_attachments = _group_attachments(config)
    missing = [group for group in group_names if group not in group_attachments]
    if missing:
        raise ValueError(f"{context} references undefined groups for {analysis}: {missing}")

    snapshots = []
    for group in group_names:
        residues = group_attachments[group]
        if len(residues) != 1:
            raise ValueError(
                f"{context} {analysis} requires group '{group}' to contain exactly "
                f"one attachment; found {len(residues)}"
            )
        residue = residues[0]
        if residue not in attachment_snapshots:
            raise ValueError(
                f"{context} {analysis} could not find attachment snapshot for "
                f"residue {residue} in group '{group}'"
            )
        snapshots.append(attachment_snapshots[residue])

    return snapshots


def _snapshot_axes(snapshots, analysis):
    axes = []
    for snapshot in snapshots:
        geometry = load_dye_geometry(snapshot.dye, require_axis=True, analysis=analysis)
        axes.append(
            axis_from_named_atoms(
                snapshot.atom_names,
                snapshot.coordinates,
                geometry.axis_atoms,
                context=f"{analysis} for dye {snapshot.dye}",
            )
        )
    return axes


def _group_attachments(config):
    return {
        group["name"]: list(group["attachments"])
        for group in config.get("groups", [])
    }


def _require_tdm(result, context):
    if "tdm" not in result.tddft:
        raise ValueError(
            f"{context} requires TDDFT transition density matrices for group "
            f"'{result.group}'. Add 'tdm' to that [[quantum]].outputs list."
        )


def _resolve_state(state, result):
    if state == "strongest":
        if "idx" not in result.tddft:
            raise ValueError(
                f"State 'strongest' for group '{result.group}' requires "
                "'strongest_state' in [[quantum]].outputs"
            )
        return int(result.tddft["idx"])
    return state


def _rebuild_mol(result):
    from pyedna.analysis.quantum.couplings import rebuild_pyscf_mol

    return rebuild_pyscf_mol(result.molecule_input, result.dft_settings)
