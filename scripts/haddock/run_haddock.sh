#!/bin/bash
#SBATCH --job-name=haddock
#SBATCH --cpus-per-task=32
#SBATCH --output=haddock_slurm.out
#SBATCH --error=haddock_slurm.err

set -e

cd "$SLURM_SUBMIT_DIR"

if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

source "$PYEDNA_HOME/config.sh"

conda run -n haddock haddock3 docking_config.cfg