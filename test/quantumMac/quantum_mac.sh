#!/bin/bash

# Ensure script stops on error
set -e

# Read command-line arguments
MOLECULE_ID=$1
TIME_IDX=$2
DO_TDDFT=""

# Check if the third argument is --do-tddft
if [[ "$3" == "--do-tddft" ]]; then
    DO_TDDFT="--do-tddft"
fi

# Check if mandatory arguments are provided
if [ -z "$MOLECULE_ID" ] || [ -z "$TIME_IDX" ]; then
    echo "Usage: ./submit.sh <molecule_id> <time_idx> [--do-tddft]"
    exit 1
fi


# Set OpenMP and MKL environment variables for PySCF on macOS
export OMP_NUM_THREADS=8  							# Use 8 threads
export MKL_NUM_THREADS=1  							# Prevents MKL from spawning extra threads
export DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH  	# Use GNU OpenMP
#ulimit -s unlimited  								# Increase stack size to prevent segmentation faults

# Run the Python script with the provided arguments
python DFT.py "$MOLECULE_ID" "$TIME_IDX" $DO_TDDFT > output.log
