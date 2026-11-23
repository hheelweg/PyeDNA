#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=n004
#SBATCH --job-name=cluster_structures
#SBATCH --output=cluster_structures.log
#SBATCH --error=cluster_structures.err

# -----------------------------------------------------------------------------
# PyeDNA structure preparation
# -----------------------------------------------------------------------------

set -e

cd "$SLURM_SUBMIT_DIR"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "gcc=$(command -v gcc)"

echo "libgfortran:"
ls -l "$CONDA_PREFIX/lib/libgfortran.so"*

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch $0 [STRUCTURE_CONFIG]"
    exit 1
fi

STRUCTURE_CONFIG="${1:-structure.toml}"

if [[ ! -f "$STRUCTURE_CONFIG" ]]; then
    echo "Error: structure configuration not found: $STRUCTURE_CONFIG"
    exit 1
fi

echo "Clustering HADDOCK structures..."
pyedna structure cluster "$STRUCTURE_CONFIG"

rm -f docking_config.cfg
