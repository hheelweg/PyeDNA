#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=analyze_traj
#SBATCH --output=analyze_traj.log

# -----------------------------------------------------------------------------
# PyeDNA trajectory analysis
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "CPU cores: $SLURM_CPUS_PER_TASK"

pyedna analysis trajectory "$@"