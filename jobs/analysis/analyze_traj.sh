#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
# Use gres=gpu:1 with [quantum_scheduler].parallel = false, or gres=gpu:2
# with [quantum_scheduler].parallel = true for two independent quantum jobs.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --job-name=analyze_traj
#SBATCH --output=analyze_traj.log

# -----------------------------------------------------------------------------
# PyeDNA trajectory analysis
# -----------------------------------------------------------------------------

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPU cores: ${SLURM_CPUS_PER_TASK:-unset}"

pyedna analysis trajectory "$@"
