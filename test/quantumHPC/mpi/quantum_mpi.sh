#!/bin/bash
#SBATCH --job-name=pyscf_mpi    # Job name
#SBATCH --nodes=1               # Run on a single node
#SBATCH --ntasks=48             # Request 48 MPI tasks
#SBATCH --cpus-per-task=1       # Each MPI task gets 1 CPU
#SBATCH --time=02:00:00         # Time limit (2 hours)
#SBATCH --output=out_mpi.log    # Output file


# Activate Conda environment with PySCF and PyTorch
source activate AmberTools24

# Set OpenMP and MKL threading options for PySCF
export OMP_NUM_THREADS=1        # Prevent OpenMP from interfering with MPI
export MKL_NUM_THREADS=1        # Prevent extra MKL threads

# Print system info for debugging
echo "Running on CPU node: $(hostname)"
echo "MPI tasks: $SLURM_NTASKS"

# Read command-line arguments
MOLECULE_ID=$1
TIME_IDX=$2
DO_TDDFT=""

# Check if the third argument is --do-tddft
if [[ "$3" == "--do-tddft" ]]; then
    DO_TDDFT="--do-tddft"
fi


# Run PySCF DFT calculation with MPI
mpirun -np 48 python DFT.py "$MOLECULE_ID" "$TIME_IDX" $DO_TDDFT