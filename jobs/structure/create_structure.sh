#!/bin/bash
#SBATCH --job-name=create_structure
#SBATCH --cpus-per-task=32
#SBATCH --output=create_structure.log
#SBATCH --error=create_structure.err

set -e

cd "$SLURM_SUBMIT_DIR"

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch $0 [STRUCTURE_CONFIG]"
    exit 1
fi

STRUCTURE_CONFIG="${1:-structure.toml}"

if [[ ! -f "$STRUCTURE_CONFIG" ]]; then
    echo "Error: structure configuration not found: $STRUCTURE_CONFIG"
    exit 1
fi

echo "Preparing structure..."
pyedna structure prepare "$STRUCTURE_CONFIG"

echo "Running HADDOCK..."
pyedna structure dock "$STRUCTURE_CONFIG"

echo "Finalizing docked structure..."
pyedna structure finalize "$STRUCTURE_CONFIG"

echo "Preparing Amber system..."
pyedna structure amber "$STRUCTURE_CONFIG"

rm -f docking_config.cfg
