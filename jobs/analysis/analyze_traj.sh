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

if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    echo "Run: source /path/to/PyeDNA/config.sh"
    exit 1
fi

source "$PYEDNA_HOME/config.sh"

# Keep CPU libraries from oversubscribing the allocated cores
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Unbuffered Python output for live SLURM logging
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "CPU cores: $SLURM_CPUS_PER_TASK"

python -m analyze_traj "$@"