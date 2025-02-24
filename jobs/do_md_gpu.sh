#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu                         # GPU partition	
#SBATCH --nodelist=gpu001                       # Run on GPU node gpu001	
#SBATCH --ntasks=1                              # # of tasks
#SBATCH --gres=gpu:2                            # Request 2 GPU
#SBATCH --cpus-per-task=8                       # use 4-8 CPUs per GPU
#SBATCH --job-name=dummy                        # Use provided job name or "default_job" if none given
#SBATCH --output=slurm-%j.out                   # Name output log file

# USAGE:
# sbatch this_script.sh my_job_name

# Source conda environment AmberTools24
source activate AmberTools24

# Add path to PyeDNA and define PyeDNA home
export PYTHONPATH=$PYTHONPATH:/home/hheelweg/cy3cy5/PyeDNA/scripts
export PYEDNA_HOME="/home/hheelweg/cy3cy5/PyeDNA"

# Add path to AMBER MD executables
export AMBERHOME=/home/hheelweg/.conda/envs/AmberTools24/amber24
export PATH=$AMBERHOME/bin:$PATH
export LD_LIBRARY_PATH=$AMBERHOME/lib:$LD_LIBRARY_PATH


# Rename job dynamically
JOB_NAME=${1:-dna_md}
scontrol update JobID=$SLURM_JOB_ID Name=$JOB_NAME


# run python module for MD simulation
python -m do_md 

# Rename output file dynamically
NEW_OUTPUT="${JOB_NAME}.out"
mv slurm-${SLURM_JOB_ID}.out $NEW_OUTPUT