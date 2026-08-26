"""Interaction analysis between analysis groups."""

from dataclasses import dataclass

from pyedna.analysis.classical import distance_between_groups


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


def run_classical_interactions(config, groups, frame):
    interactions = config.get("classical_interactions", [])
    if not interactions:
        return []

    results = []
    for index, interaction in enumerate(interactions, start=1):
        if interaction["type"] != "distance":
            continue

        results.append(
            run_distance_interaction(
                interaction,
                groups,
                frame,
                context=f"[[classical_interactions]] block {index}",
            )
        )

    return results


def run_interactions(config, quantum_results, groups=None, frame=None):
    return [
        *run_quantum_interactions(config, quantum_results),
        *run_classical_interactions(config, groups or {}, frame),
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
