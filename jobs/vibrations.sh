#!/bin/bash
#SBATCH --partition=gpu             # GPU partition
#SBATCH --nodelist=gpu001           # Run on GPU node gpu001
#SBATCH --gres=gpu:2                # Request 2 GPU
#SBATCH --cpus-per-task=48          # Request 48 CPU cores
#SBATCH --job-name=orca             # Job name
#SBATCH --output=orca.log           # Output file

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
mpirun --version

# Run ORCA vibrational analysis calculation
python -m vibrational