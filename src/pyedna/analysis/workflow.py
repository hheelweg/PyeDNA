"""High-level trajectory-analysis workflow orchestration."""

from pyedna.analysis.io import (
    append_classical_interaction_results,
    append_classical_results,
    append_quantum_interaction_results,
    append_quantum_results,
    prepare_output_files,
)
from pyedna.analysis.classical import run_classical_jobs, summarize_classical_result
from pyedna.analysis.interactions import (
    run_classical_interactions,
    run_quantum_interactions,
    summarize_interaction_result,
)
from pyedna.analysis.quantum.jobs import run_quantum_jobs, summarize_quantum_result
from pyedna.analysis.runtime import configure_thread_environment, detect_runtime_resources
from pyedna.trajectory.structure import build_group_fragments, build_groups
from pyedna.trajectory.trajectory import analyze_trajectory, validate_frame_interval


def run_trajectory_analysis(config_file="traj.toml"):
    traj, cfg = analyze_trajectory(config_file)
    traj_cfg = cfg["trajectory"]
    quantum_defaults = cfg.get("quantum_defaults", {})
    resources = detect_runtime_resources()
    configure_thread_environment(resources.num_cpus)
    start, stop = validate_frame_interval(traj_cfg["frame_interval"], traj.num_frames)

    attachments = cfg.get("attachments", [])
    optimize_caps = traj_cfg.get(
        "optimize_caps",
        quantum_defaults.get("optimize_caps", False),
    )
    basis = traj_cfg.get("basis", quantum_defaults.get("basis", "6-31g"))
    analysis_run = prepare_output_files(cfg, config_file=config_file)
    print(f"Analysis output: {analysis_run.directory}")
    print(
        "Runtime resources: "
        f"{resources.num_cpus} CPU cores, {resources.num_gpus} visible GPU(s)"
    )

    for frame in range(start, stop + 1):
        analyze_frame(
            cfg,
            traj,
            frame,
            analysis_run,
            attachments=attachments,
            optimize_caps=optimize_caps,
            basis=basis,
            resources=resources,
        )

    return analysis_run


def analyze_frame(
    config,
    traj,
    frame,
    analysis_run,
    attachments=None,
    optimize_caps=False,
    basis="6-31g",
    resources=None,
):
    resources = detect_runtime_resources() if resources is None else resources
    attachment_mols = {}

    for attachment in attachments or config.get("attachments", []):
        residue = attachment["residue"]
        attachment_mols[residue] = traj.get_capped_snapshot(
            frame=frame,
            initial_residue=residue,
            dye=attachment["dye"],
            cap_type=attachment.get("cap", "H"),
            optimize_caps=optimize_caps,
            basis=basis,
            resources=resources,
        )

    groups = build_groups(config, attachment_mols, basis=basis)
    group_fragments = build_group_fragments(config, attachment_mols)

    for name, mol in groups.items():
        print(
            f"Frame {frame}: group {name}, "
            f"{mol.natm} atoms, charge {mol.charge}"
        )

    classical_results = run_classical_jobs(config, groups, frame)
    append_classical_results(analysis_run, classical_results)

    for result in classical_results:
        print(summarize_classical_result(result))

    quantum_results = run_quantum_jobs(
        config,
        groups,
        frame,
        group_fragments=group_fragments,
        resources=resources,
    )
    append_quantum_results(analysis_run, quantum_results)

    for result in quantum_results:
        print(summarize_quantum_result(result))

    quantum_interaction_results = run_quantum_interactions(config, quantum_results)
    append_quantum_interaction_results(analysis_run, quantum_interaction_results)

    for result in quantum_interaction_results:
        print(summarize_interaction_result(result))

    classical_interaction_results = run_classical_interactions(config, groups=groups, frame=frame)
    append_classical_interaction_results(analysis_run, classical_interaction_results)

    for result in classical_interaction_results:
        print(summarize_interaction_result(result))

    return {
        "groups": groups,
        "classical": classical_results,
        "quantum": quantum_results,
        "quantum_interactions": quantum_interaction_results,
        "classical_interactions": classical_interaction_results,
    }
