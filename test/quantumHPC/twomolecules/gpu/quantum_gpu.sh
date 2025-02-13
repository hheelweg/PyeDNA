#!/bin/bash
#SBATCH --job-name=pyscf_gpu     # Job name
#SBATCH --partition=gpu          # GPU partition
#SBATCH --nodelist=gpu001        # Run on GPU node gpu001
#SBATCH --gres=gpu:2             # Request 2 GPU
#SBATCH --cpus-per-task=48       # Request 48 CPU cores
#SBATCH --output=out_gpu.log     # Output file

# Activate conda environment
source activate AmberTools24

# Read command-line arguments
MOLECULE_1_ID=$1
MOLECULE_2_ID=$2
TIME_IDX=$3

# Print allocated GPUs
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Run PySCF DFT calculation with PyTorch acceleration
python dft_control.py "$MOLECULE_1_ID" "$MOLECULE_2_ID" "$TIME_IDX"