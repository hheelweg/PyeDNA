#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu                         # GPU partition	
#SBATCH --nodelist=gpu001                       # Run on GPU node gpu001	
#SBATCH --ntasks=1                              # # of tasks
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --cpus-per-task=8                       # use 4-8 CPUs per GPU
#SBATCH --job-name=dummy                        # Use provided job name or "default_job" if none given
#SBATCH --output=slurm-%j.log                   # Name output log file

# USAGE:
# sbatch "$PYEDNA_HOME/jobs/md/do_md_gpu.sh" [MD_CONFIG]
#
# MD_CONFIG is optional and defaults to "md.toml" in the current working
# directory.


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


if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [MD_CONFIG]"
    exit 1
fi

MD_CONFIG="${1:-md.toml}"

if [[ ! -f "$MD_CONFIG" ]]; then
    echo "Error: MD configuration not found: $MD_CONFIG"
    exit 1
fi

JOB_NAME="$(basename "$MD_CONFIG" .toml)_md"

# Update SLURM job name
scontrol update JobID=$SLURM_JOB_ID Name=$JOB_NAME


# Run python module for MD simulation
python -m do_md "$MD_CONFIG"

# Rename output file dynamically
NEW_OUTPUT="${JOB_NAME}.log"
mv slurm-${SLURM_JOB_ID}.log $NEW_OUTPUT
