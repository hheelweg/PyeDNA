#!/bin/bash
#SBATCH --job-name=pyscf_gpu     # Job name
#SBATCH --partition=gpu          # GPU partition
#SBATCH --nodelist=gpu001        # Run on GPU node gpu001
#SBATCH --gres=gpu:2             # Request 2 GPU
#SBATCH --cpus-per-task=48       # Request 48 CPU cores
#SBATCH --output=out_coup.log    # Output file

# Activate conda environment
source activate AmberTools24

# Read command-line arguments
MOLECULE_1_ID=$1
MOLECULE_2_ID=$2

# Run PySCF DFT calculation with PyTorch acceleration
python coupling_gpu.py "$MOLECULE_1_ID" "$MOLECULE_2_ID"