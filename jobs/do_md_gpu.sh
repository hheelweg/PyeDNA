#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu             # GPU partition	
#SBATCH --nodelist=gpu001           # Run on GPU node gpu001	
#SBATCH --ntasks=1                  # # of tasks
#SBATCH --gres=gpu:2                # Request 2 GPU
##SBATCH --cpus-per-task=8          # use 4-8 CPUs per GPU
#SBATCH --job-name=dna_test         # Job name
#SBATCH --output=dna_test.log       # Output file

# Source conda environment AmberTools24
source activate AmberTools24

# Add path to PyeDNA and define PyeDNA home
export PYTHONPATH=$PYTHONPATH:/home/hheelweg/cy3cy5/PyeDNA/scripts
export PYEDNA_HOME="/home/hheelweg/cy3cy5/PyeDNA"

# Add path to AMBER MD executables
export AMBERHOME=/home/hheelweg/.conda/envs/AmberTools24/amber24
export PATH=$AMBERHOME/bin:$PATH
export LD_LIBRARY_PATH=$AMBERHOME/lib:$LD_LIBRARY_PATH

# (1) perform static minimizations

# (2) perform equilibrations

# (3) production run