#!/bin/bash
#SBATCH --job-name=pyscf_gpu     # Job name
#SBATCH --partition=gpu          # GPU partition
#SBATCH --nodelist=gpu001        # Run on GPU node gpu001
#SBATCH --gres=gpu:2             # Request 2 GPU
#SBATCH --cpus-per-task=48       # Request 48 CPU cores
#SBATCH --output=out_gpu.log     # Output file

# source conda environment
source activate AmberTools24

# Add path to PyeDNA and define PyeDNA home
export PYTHONPATH=$PYTHONPATH:/home/hheelweg/cy3cy5/PyeDNA/scripts
export PYEDNA_HOME="/home/hheelweg/cy3cy5/PyeDNA"

# Print allocated GPUs
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Run tracjetory analysis calculation with GPU acceleration
python -m analyze_traj