# Analyze Trajectory

The analysis workflow reads an Amber topology and trajectory, extracts capped dye snapshots, groups attachments, and runs classical and/or quantum calculations.

The current quantum trajectory workflow requires GPU4PySCF and does not implement a CPU-only fallback. See [Installation](../getting_started/installation.md) for the validated GPU Python stack.

See [analyze_traj](analyze_traj.md) for `traj.toml` fields and output files.

See [Loading Analysis Results](loading_results.md) for short examples that load JSONL outputs into pandas dataframes.

The old analysis migration notes are preserved in [analysis_migration_audit](analysis_migration_audit.md).
