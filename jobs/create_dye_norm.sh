#!/bin/bash
#SBATCH --partition=normal          # normal partition
#SBATCH --ntasks=8                  # 16 MPI tasks
#SBATCH --job-name=dye              # Job name
#SBATCH --output=dye.log            # Output file

# USAGE:
# sbatch this_script.sh

# Check if PYEDNA_HOME is set
if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set. Please set it in shell."
    exit 1
fi

# Load config.sh from the root of PyeDNA to set user-specific environment variables
CONFIG_FILE="$PYEDNA_HOME/config.sh"

if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file ($CONFIG_FILE) not found!"
    exit 1
fi

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run trajectory analysis calculation with GPU acceleration
mpirun -np $SLURM_NTASKS python -m create_dye