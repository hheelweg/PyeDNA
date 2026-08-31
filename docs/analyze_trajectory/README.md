# Analyze Trajectory

The analysis workflow reads an Amber topology and trajectory, extracts capped dye snapshots, groups attachments, and runs classical and/or quantum calculations.

Quantum trajectory analysis runs with plain CPU PySCF when no CUDA GPU is visible to the job, and uses GPU4PySCF automatically when GPU resources and the validated GPU Python stack are available.

See [analyze_traj](analyze_traj.md) for `traj.toml` fields and output files.

See [Loading Analysis Results](loading_results.md) for short examples that load JSONL outputs into pandas dataframes.

The old analysis migration notes are preserved in [analysis_migration_audit](analysis_migration_audit.md).
