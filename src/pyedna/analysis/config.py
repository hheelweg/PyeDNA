"""Validation for the trajectory-analysis TOML schema."""

from dataclasses import dataclass

from .units import DEFAULT_ANALYSIS_UNITS, SUPPORTED_DISTANCE_UNITS, SUPPORTED_ENERGY_UNITS

SUPPORTED_CAPS = {"H", "CH3"}
SUPPORTED_QUANTUM_METHODS = {"dft", "tddft"}
SUPPORTED_QUANTUM_BACKENDS = {"pyscf", "orca"}
SUPPORTED_QUANTUM_OUTPUTS = {
    "energies",
    "excited_state_energies",
    "excitation_energies",
    "oscillator_strengths",
    "transition_dipoles",
    "transition_quadrupoles",
    "transition_density_matrices",
    "tdm",
    "strongest_state",
    "mulliken",
    "mulliken_populations",
    "mulliken_charges",
    "opa",
    "orbital_participation",
}
SUPPORTED_CLASSICAL_OUTPUTS = {
    "axis_angle",
    "center_of_geometry",
    "center_of_mass",
    "radius_of_gyration",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Validated trajectory-analysis configuration.

    Group membership intentionally references attachment residues directly.
    Residues therefore need to be unique across ``[[attachments]]`` blocks.
    """

    data: dict


def validate_analysis_config(config):
    if not isinstance(config, dict):
        raise TypeError("traj.toml must define a TOML table")

    _validate_trajectory(config)
    attachment_residues = _validate_attachments(config)
    group_names = _validate_groups(config, attachment_residues)
    _apply_quantum_defaults(config)
    _validate_quantum(config, group_names)
    _validate_classical(config, group_names)
    _normalize_interactions(config)
    _validate_classical_interactions(config, attachment_residues, group_names)
    _validate_quantum_interactions(config, attachment_residues, group_names)
    _infer_quantum_requirements(config)
    _validate_analysis(config)
    _validate_quantum_scheduler(config)
    _validate_output(config)

    return AnalysisConfig(config)


def _validate_trajectory(config):
    traj = config.get("trajectory")
    if not isinstance(traj, dict):
        raise ValueError("traj.toml requires a [trajectory] table")

    required = ("run_directory", "topology_file", "trajectory_file", "frame_interval")
    missing = [key for key in required if key not in traj]
    if missing:
        raise ValueError(f"[trajectory] is missing required keys: {missing}")

    for key in ("run_directory", "topology_file", "trajectory_file"):
        if not isinstance(traj[key], str) or not traj[key]:
            raise TypeError(f"[trajectory].{key} must be a non-empty string")

    _validate_frame_interval_shape(traj["frame_interval"])

    if "optimize_caps" in traj and not isinstance(traj["optimize_caps"], bool):
        raise TypeError("[trajectory].optimize_caps must be true or false")
    if "basis" in traj and not isinstance(traj["basis"], str):
        raise TypeError("[trajectory].basis must be a string")


def _validate_frame_interval_shape(frame_interval):
    if not isinstance(frame_interval, list) or len(frame_interval) != 2:
        raise ValueError("[trajectory].frame_interval must be [initial_frame, final_frame]")

    start, stop = frame_interval
    if not isinstance(start, int) or not isinstance(stop, int):
        raise TypeError("[trajectory].frame_interval values must be integers")
    if start < 0:
        raise ValueError("[trajectory].frame_interval initial frame cannot be negative")
    if stop < start:
        raise ValueError("[trajectory].frame_interval final frame must be >= initial frame")


def _validate_attachments(config):
    attachments = config.get("attachments")
    if not isinstance(attachments, list) or not attachments:
        raise ValueError("traj.toml requires at least one [[attachments]] block")

    residues = []
    for index, item in enumerate(attachments, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"[[attachments]] block {index} must be a table")

        missing = [key for key in ("dye", "residue") if key not in item]
        if missing:
            raise ValueError(f"[[attachments]] block {index} is missing required keys: {missing}")

        if not isinstance(item["dye"], str) or not item["dye"]:
            raise TypeError(f"[[attachments]] block {index} dye must be a non-empty string")
        if not isinstance(item["residue"], int):
            raise TypeError(f"[[attachments]] block {index} residue must be an integer")

        cap = item.get("cap", "H")
        if not isinstance(cap, str):
            raise TypeError(f"[[attachments]] block {index} cap must be a string")
        cap = cap.upper()
        if cap not in SUPPORTED_CAPS:
            raise ValueError(
                f"Unsupported cap '{item.get('cap')}' for residue {item['residue']}; use H or CH3"
            )
        item["cap"] = cap
        residues.append(item["residue"])

    duplicates = sorted({residue for residue in residues if residues.count(residue) > 1})
    if duplicates:
        raise ValueError(
            "Duplicate [[attachments]].residue values are not allowed because "
            f"[[groups]].attachments references residues directly: {duplicates}"
        )

    return set(residues)


def _validate_groups(config, attachment_residues):
    groups = config.get("groups", [])
    if groups is None:
        groups = []
        config["groups"] = groups
    if not isinstance(groups, list):
        raise TypeError("[[groups]] blocks must form a list")

    names = []
    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise TypeError(f"[[groups]] block {index} must be a table")

        missing = [key for key in ("name", "attachments") if key not in group]
        if missing:
            raise ValueError(f"[[groups]] block {index} is missing required keys: {missing}")

        name = group["name"]
        if not isinstance(name, str) or not name:
            raise TypeError(f"[[groups]] block {index} name must be a non-empty string")
        if not _is_int_list(group["attachments"]):
            raise TypeError(f"[[groups]] '{name}' attachments must be a list of residues")
        if not group["attachments"]:
            raise ValueError(f"[[groups]] '{name}' must include at least one attachment residue")

        missing_residues = sorted(set(group["attachments"]) - attachment_residues)
        if missing_residues:
            raise ValueError(
                f"[[groups]] '{name}' references undefined attachment residues: {missing_residues}"
            )

        names.append(name)

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate [[groups]].name values are not allowed: {duplicates}")

    return set(names)


def _validate_quantum(config, group_names):
    quantum_jobs = config.get("quantum", [])
    if quantum_jobs is None:
        quantum_jobs = []
        config["quantum"] = quantum_jobs
    if not isinstance(quantum_jobs, list):
        raise TypeError("[[quantum]] blocks must form a list")

    for index, job in enumerate(quantum_jobs, start=1):
        if not isinstance(job, dict):
            raise TypeError(f"[[quantum]] block {index} must be a table")

        missing = [key for key in ("group", "method") if key not in job]
        if missing:
            raise ValueError(f"[[quantum]] block {index} is missing required keys: {missing}")

        _validate_group_reference(job["group"], group_names, f"[[quantum]] block {index}")

        method = job["method"]
        if not isinstance(method, str):
            raise TypeError(f"[[quantum]] block {index} method must be a string")
        method = method.lower()
        if method not in SUPPORTED_QUANTUM_METHODS:
            raise ValueError(
                f"Unsupported [[quantum]] method '{job['method']}'; "
                f"use one of {sorted(SUPPORTED_QUANTUM_METHODS)}"
            )
        job["method"] = method

        if "nstates" in job:
            _validate_positive_int(job["nstates"], f"[[quantum]] block {index} nstates")
        if "state_ids" in job:
            _validate_state_ids(job["state_ids"], f"[[quantum]] block {index}")
        if "outputs" in job:
            if not _is_str_list(job["outputs"]):
                raise TypeError(f"[[quantum]] block {index} outputs must be a list of strings")
            outputs = [output.lower() for output in job["outputs"]]
            unsupported = sorted(set(outputs) - SUPPORTED_QUANTUM_OUTPUTS)
            if unsupported:
                raise ValueError(
                    f"[[quantum]] block {index} has unsupported outputs: {unsupported}"
                )
            job["outputs"] = outputs
        else:
            job["outputs"] = []
        for bool_key in ("gpu", "density_fit", "tda", "singlet"):
            if bool_key in job and not isinstance(job[bool_key], bool):
                raise TypeError(f"[[quantum]] block {index} {bool_key} must be true or false")
        for int_key in ("scf_cycles", "verbosity"):
            if int_key in job and not isinstance(job[int_key], int):
                raise TypeError(f"[[quantum]] block {index} {int_key} must be an integer")
        if "backend" in job:
            if not isinstance(job["backend"], str) or not job["backend"]:
                raise TypeError(f"[[quantum]] block {index} backend must be a non-empty string")
            job["backend"] = job["backend"].lower()
            if job["backend"] not in SUPPORTED_QUANTUM_BACKENDS:
                raise ValueError(
                    f"Unsupported [[quantum]] backend '{job['backend']}'; "
                    f"use one of {sorted(SUPPORTED_QUANTUM_BACKENDS)}"
                )
        for str_key in ("basis", "xc"):
            if str_key in job and (not isinstance(job[str_key], str) or not job[str_key]):
                raise TypeError(f"[[quantum]] block {index} {str_key} must be a non-empty string")
        for int_key in ("charge", "spin"):
            if int_key in job and not isinstance(job[int_key], int):
                raise TypeError(f"[[quantum]] block {index} {int_key} must be an integer")


def _validate_classical(config, group_names):
    classical_jobs = config.get("classical", [])
    if classical_jobs is None:
        classical_jobs = []
        config["classical"] = classical_jobs
    if not isinstance(classical_jobs, list):
        raise TypeError("[[classical]] blocks must form a list")

    for index, job in enumerate(classical_jobs, start=1):
        if not isinstance(job, dict):
            raise TypeError(f"[[classical]] block {index} must be a table")

        if "group" in job:
            _validate_group_reference(job["group"], group_names, f"[[classical]] block {index}")
        else:
            raise ValueError(f"[[classical]] block {index} requires group")
        if "outputs" in job and not _is_str_list(job["outputs"]):
            raise TypeError(f"[[classical]] block {index} outputs must be a list of strings")
        if "outputs" in job:
            outputs = [output.lower() for output in job["outputs"]]
            unsupported = sorted(set(outputs) - SUPPORTED_CLASSICAL_OUTPUTS)
            if unsupported:
                raise ValueError(
                    f"[[classical]] block {index} has unsupported outputs: {unsupported}"
                )
            job["outputs"] = outputs


def _normalize_interactions(config):
    interactions = config.get("interactions", [])
    if interactions is None:
        interactions = []
        config["interactions"] = interactions
    if not isinstance(interactions, list):
        raise TypeError("[[interactions]] blocks must form a list")

    quantum_interactions = config.get("quantum_interactions", [])
    classical_interactions = config.get("classical_interactions", [])
    if quantum_interactions is None:
        quantum_interactions = []
    if classical_interactions is None:
        classical_interactions = []
    if not isinstance(quantum_interactions, list):
        raise TypeError("[[quantum_interactions]] blocks must form a list")
    if not isinstance(classical_interactions, list):
        raise TypeError("[[classical_interactions]] blocks must form a list")

    for index, interaction in enumerate(interactions, start=1):
        if not isinstance(interaction, dict):
            raise TypeError(f"[[interactions]] block {index} must be a table")
        interaction_type = interaction.get("type")
        if interaction_type == "coupling":
            quantum_interactions.append(interaction)
        elif interaction_type == "distance":
            classical_interactions.append(interaction)
        else:
            raise ValueError(
                f"[[interactions]] block {index} has unsupported type '{interaction_type}'"
            )

    config["quantum_interactions"] = quantum_interactions
    config["classical_interactions"] = classical_interactions


def _validate_classical_interactions(config, attachment_residues, group_names):
    interactions = config.get("classical_interactions", [])
    for index, interaction in enumerate(interactions, start=1):
        if not isinstance(interaction, dict):
            raise TypeError(f"[[classical_interactions]] block {index} must be a table")

        _validate_interaction_references(
            interaction,
            attachment_residues,
            group_names,
            f"[[classical_interactions]] block {index}",
        )

        if "type" not in interaction:
            raise ValueError(f"[[classical_interactions]] block {index} requires type")
        if not isinstance(interaction["type"], str) or not interaction["type"]:
            raise TypeError(f"[[classical_interactions]] block {index} type must be a non-empty string")
        interaction["type"] = interaction["type"].lower()
        if interaction["type"] != "distance":
            raise ValueError(f"[[classical_interactions]] block {index} currently supports type = 'distance'")

        if "groups" not in interaction:
            raise ValueError(f"[[classical_interactions]] block {index} distance requires groups")
        if len(interaction["groups"]) != 2:
            raise ValueError(f"[[classical_interactions]] block {index} distance requires exactly two groups")
        method = interaction.get("method", "center_of_geometry")
        if method not in ("center_of_geometry", "center_of_mass"):
            raise ValueError(
                f"[[classical_interactions]] block {index} distance method must be center_of_geometry or center_of_mass"
            )


def _validate_quantum_interactions(config, attachment_residues, group_names):
    interactions = config.get("quantum_interactions", [])
    for index, interaction in enumerate(interactions, start=1):
        if not isinstance(interaction, dict):
            raise TypeError(f"[[quantum_interactions]] block {index} must be a table")

        _validate_interaction_references(
            interaction,
            attachment_residues,
            group_names,
            f"[[quantum_interactions]] block {index}",
        )

        if "type" not in interaction:
            raise ValueError(f"[[quantum_interactions]] block {index} requires type")
        if not isinstance(interaction["type"], str) or not interaction["type"]:
            raise TypeError(f"[[quantum_interactions]] block {index} type must be a non-empty string")
        interaction["type"] = interaction["type"].lower()
        if interaction["type"] != "coupling":
            raise ValueError(f"[[quantum_interactions]] block {index} currently supports type = 'coupling'")

        if "groups" not in interaction:
            raise ValueError(f"[[quantum_interactions]] block {index} coupling requires groups")
        if len(interaction["groups"]) != 2:
            raise ValueError(f"[[quantum_interactions]] block {index} coupling requires exactly two groups")
        method = interaction.get("method", "tdm")
        if method != "tdm":
            raise ValueError(f"[[quantum_interactions]] block {index} coupling currently supports method = 'tdm'")
        if "state_pairs" in interaction:
            _validate_state_pairs(interaction["state_pairs"], f"[[quantum_interactions]] block {index}")
        coupling_type = interaction.get("coupling_type", "electronic")
        if coupling_type not in ("electronic", "cJ", "cK"):
            raise ValueError(
                f"[[quantum_interactions]] block {index} coupling_type must be electronic, cJ, or cK"
            )


def _validate_interaction_references(interaction, attachment_residues, group_names, context):
    has_groups = "groups" in interaction
    has_attachments = "attachments" in interaction
    if has_groups == has_attachments:
        raise ValueError(f"{context} must define exactly one of groups or attachments")

    if has_groups:
        groups = interaction["groups"]
        if not _is_str_list(groups) or len(groups) < 2:
            raise TypeError(f"{context} groups must be a list of at least two group names")
        missing = sorted(set(groups) - group_names)
        if missing:
            raise ValueError(f"{context} references undefined groups: {missing}")

    if has_attachments:
        residues = interaction["attachments"]
        if not _is_int_list(residues) or len(residues) < 2:
            raise TypeError(f"{context} attachments must be a list of at least two residues")
        missing = sorted(set(residues) - attachment_residues)
        if missing:
            raise ValueError(f"{context} references undefined attachment residues: {missing}")

    if "method" in interaction and not isinstance(interaction["method"], str):
        raise TypeError(f"{context} method must be a string")


def _validate_output(config):
    output = config.get("output", {})
    if output is None:
        output = {}
        config["output"] = output
    if not isinstance(output, dict):
        raise TypeError("[output] must be a table")

    if "quantum_file" in output:
        if not isinstance(output["quantum_file"], str) or not output["quantum_file"]:
            raise TypeError("[output].quantum_file must be a non-empty string")
    for key in ("interaction_file", "quantum_interactions_file", "classical_interactions_file"):
        if key in output and (not isinstance(output[key], str) or not output[key]):
            raise TypeError(f"[output].{key} must be a non-empty string")
    if "classical_file" in output:
        if not isinstance(output["classical_file"], str) or not output["classical_file"]:
            raise TypeError("[output].classical_file must be a non-empty string")


def _validate_analysis(config):
    analysis = config.get("analysis", {})
    if analysis is None:
        analysis = {}
        config["analysis"] = analysis
    if not isinstance(analysis, dict):
        raise TypeError("[analysis] must be a table")

    if "output_root" in analysis:
        if not isinstance(analysis["output_root"], str) or not analysis["output_root"]:
            raise TypeError("[analysis].output_root must be a non-empty string")
    if "name" in analysis:
        if not isinstance(analysis["name"], str) or not analysis["name"]:
            raise TypeError("[analysis].name must be a non-empty string")

    units = analysis.get("units", {})
    if units is None:
        units = {}
    if not isinstance(units, dict):
        raise TypeError("[analysis.units] must be a table")
    for key, value in DEFAULT_ANALYSIS_UNITS.items():
        units.setdefault(key, value)
    _validate_unit(units["energy"], SUPPORTED_ENERGY_UNITS, "[analysis.units].energy")
    _validate_unit(units["coupling"], SUPPORTED_ENERGY_UNITS, "[analysis.units].coupling")
    _validate_unit(units["distance"], SUPPORTED_DISTANCE_UNITS, "[analysis.units].distance")
    analysis["units"] = units

    save = analysis.get("save", {})
    if save is None:
        save = {}
        analysis["save"] = save
    if not isinstance(save, dict):
        raise TypeError("[analysis.save] must be a table")
    if "save_intermediates" in save and not isinstance(save["save_intermediates"], bool):
        raise TypeError("[analysis.save].save_intermediates must be true or false")


def _validate_quantum_scheduler(config):
    scheduler = config.get("quantum_scheduler", {})
    if scheduler is None:
        scheduler = {}
        config["quantum_scheduler"] = scheduler
    if not isinstance(scheduler, dict):
        raise TypeError("[quantum_scheduler] must be a table")

    if "parallel" in scheduler and not isinstance(scheduler["parallel"], bool):
        raise TypeError("[quantum_scheduler].parallel must be true or false")
    if "gpu_ids" in scheduler:
        if not _is_int_list(scheduler["gpu_ids"]) or not scheduler["gpu_ids"]:
            raise TypeError("[quantum_scheduler].gpu_ids must be a non-empty list of integers")
    if "max_workers" in scheduler:
        _validate_positive_int(scheduler["max_workers"], "[quantum_scheduler].max_workers")


def _apply_quantum_defaults(config):
    defaults = config.get("quantum_defaults", {})
    if defaults is None:
        defaults = {}
        config["quantum_defaults"] = defaults
    if not isinstance(defaults, dict):
        raise TypeError("[quantum_defaults] must be a table")

    if "optimize_caps" in defaults and not isinstance(defaults["optimize_caps"], bool):
        raise TypeError("[quantum_defaults].optimize_caps must be true or false")
    if "basis" in defaults and (not isinstance(defaults["basis"], str) or not defaults["basis"]):
        raise TypeError("[quantum_defaults].basis must be a non-empty string")

    for job in config.get("quantum", []) or []:
        for key, value in defaults.items():
            if key == "optimize_caps":
                continue
            job.setdefault(key, value)


def _infer_quantum_requirements(config):
    quantum_by_group = {
        job["group"]: job
        for job in config.get("quantum", [])
    }

    for job in quantum_by_group.values():
        requested = list(job.get("outputs", []))
        job["_write_outputs"] = requested
        job["_compute_outputs"] = list(requested)

    for interaction in config.get("quantum_interactions", []):
        if interaction["type"] == "coupling" and interaction.get("method", "tdm") == "tdm":
            for group in interaction["groups"]:
                if group not in quantum_by_group:
                    continue
                outputs = quantum_by_group[group]["_compute_outputs"]
                if "tdm" not in outputs:
                    outputs.append("tdm")
                if _uses_strongest_state(interaction) and "strongest_state" not in outputs:
                    outputs.append("strongest_state")

    for job in quantum_by_group.values():
        job["outputs"] = job["_compute_outputs"]


def _uses_strongest_state(interaction):
    for state_pair in interaction.get("state_pairs", []):
        if "strongest" in state_pair:
            return True
    return False


def _validate_state_pairs(state_pairs, context):
    if not isinstance(state_pairs, list):
        raise TypeError(f"{context} state_pairs must be a list")

    for pair in state_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{context} state_pairs entries must be two-item lists")
        for state in pair:
            if state == "strongest":
                continue
            if not isinstance(state, int) or state < 0:
                raise TypeError(f"{context} state entries must be non-negative integers or 'strongest'")


def _validate_group_reference(group, group_names, context):
    if not isinstance(group, str) or not group:
        raise TypeError(f"{context} group must be a non-empty string")
    if group not in group_names:
        raise ValueError(f"{context} references undefined group '{group}'")


def _validate_positive_int(value, context):
    if not isinstance(value, int) or value <= 0:
        raise TypeError(f"{context} must be a positive integer")


def _validate_unit(value, supported, context):
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a non-empty string")
    if value.lower() not in supported:
        raise ValueError(f"{context} must be one of {sorted(supported)}")


def _validate_state_ids(value, context):
    if not _is_int_list(value) or not value:
        raise TypeError(f"{context} state_ids must be a non-empty list of integers")
    if value != list(range(len(value))):
        raise ValueError(
            f"{context} state_ids must be contiguous and zero-based, e.g. [0, 1, 2]"
        )


def _is_int_list(value):
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def _is_str_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
