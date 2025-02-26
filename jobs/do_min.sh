#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=normal                      # CPU partition		
#SBATCH --ntasks=8                              # # of tasks
#SBATCH --job-name=dummy                        # Use provided job name or "default_job" if none given
#SBATCH --output=slurm-%j.log                   # Name output log file

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

# Default job name
JOB_NAME="dna_md"

# Check if first argument is not an option (i.e., doesn't start with "--")
if [[ $# -gt 0 && "$1" != --* ]]; then
    JOB_NAME=$1
    shift  
fi

# Update SLURM job name
scontrol update JobID=$SLURM_JOB_ID Name=$JOB_NAME


# Run python module for MD simulation
python -m do_md "$@"

# Rename output file dynamically
NEW_OUTPUT="${JOB_NAME}.log"
mv slurm-${SLURM_JOB_ID}.log $NEW_OUTPUT
