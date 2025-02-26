#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu                         # GPU partition	
#SBATCH --nodelist=gpu001                       # Run on GPU node gpu001	
#SBATCH --ntasks=1                              # # of tasks
#SBATCH --gres=gpu:2                            # Request 2 GPU
#SBATCH --cpus-per-task=8                       # use 4-8 CPUs per GPU
#SBATCH --job-name=dummy                        # Use provided job name or "default_job" if none given
#SBATCH --output=slurm-%j.log                   # Name output log file

# USAGE:
# sbatch this_script.sh [my_job_name] --sim [sim_program] --clean [clean_level]

# # Source conda environment AmberTools24
# source activate AmberTools24

# # Add path to PyeDNA and define PyeDNA home
# export PYTHONPATH=$PYTHONPATH:/home/hheelweg/cy3cy5/PyeDNA/scripts
# export PYEDNA_HOME="/home/hheelweg/cy3cy5/PyeDNA"

# # Add path to AMBER MD executables
# export AMBERHOME=/home/hheelweg/.conda/envs/AmberTools24/amber24
# export PATH=$AMBERHOME/bin:$PATH
# export LD_LIBRARY_PATH=$AMBERHOME/lib:$LD_LIBRARY_PATH


# Load user-defined paths
CONFIG_FILE="$(dirname "$0")/../config.sh"
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