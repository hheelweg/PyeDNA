#!/bin/bash

# dynamic job name
JOB_NAME = $(scontrol show job $SLURM_JOB_ID | awk -F= '/JobName/ {print $2}')
JOB_NAME = ${SLURM_JOB_NAME:-default_job}

# SBATCH --nodes=1
# SBATCH --partition=gpu                # GPU partition	
# SBATCH --nodelist=gpu001              # Run on GPU node gpu001	
# SBATCH --ntasks=1                     # # of tasks
# SBATCH --gres=gpu:2                   # Request 2 GPU
# SBATCH --cpus-per-task=8              # use 4-8 CPUs per GPU
# SBATCH --job-name=${JOB_NAME}         # Use provided job name or "default_job" if none given
# SBATCH --output=${JOB_NAME}.log       # Name output log file

# USAGE:
# sbatch this_script.sh --job-name=my_name

# Source conda environment AmberTools24
source activate AmberTools24

# Add path to PyeDNA and define PyeDNA home
export PYTHONPATH=$PYTHONPATH:/home/hheelweg/cy3cy5/PyeDNA/scripts
export PYEDNA_HOME="/home/hheelweg/cy3cy5/PyeDNA"

# Add path to AMBER MD executables
export AMBERHOME=/home/hheelweg/.conda/envs/AmberTools24/amber24
export PATH=$AMBERHOME/bin:$PATH
export LD_LIBRARY_PATH=$AMBERHOME/lib:$LD_LIBRARY_PATH


# TODO : write this so that I can individually switch which minimizations/equilibration steps I want to do!
# I guess the best way to do this is by executing through a python file 

# don't to it like THIS:

# (1) perform static minimizations

# (2) perform equilibrations

# (3) production run

# run python module for MD simulation
python -m do_md > ${JOB_NAME}_output.log 2>&1