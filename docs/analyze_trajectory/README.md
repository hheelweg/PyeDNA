# Analyze Trajectory

The analysis workflow reads one or more Amber topology/trajectory pairs, extracts capped dye snapshots, groups attachments, and runs classical and/or quantum calculations.

For trajectories produced by `pyedna md run`, point `[trajectory].run_directory` at the MD run directory and select ranked structures with `[trajectory].structures`. PyeDNA resolves each `structure_<NNN>/` topology and trajectory from the MD `manifest.toml`.

Quantum trajectory analysis runs with plain CPU PySCF when no CUDA GPU is visible to the job, and uses GPU4PySCF automatically when GPU resources and the validated GPU Python stack are available.

See [analyze_traj](analyze_traj.md) for `traj.toml` fields and output files.

See [Loading Analysis Results](loading_results.md) for short examples that load JSONL outputs into pandas dataframes.

The old analysis migration notes are preserved in [analysis_migration_audit](analysis_migration_audit.md).
