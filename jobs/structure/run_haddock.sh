#!/bin/bash
#SBATCH --job-name=haddock
#SBATCH --cpus-per-task=32
#SBATCH --output=haddock/haddock_slurm.out
#SBATCH --error=haddock/haddock_slurm.err

set -e

cd "$SLURM_SUBMIT_DIR"

PYEDNA_STRUCTURE_CONFIG="${PYEDNA_STRUCTURE_CONFIG:-structure.toml}"

if [[ ! -f "$PYEDNA_STRUCTURE_CONFIG" ]]; then
    echo "Error: structure configuration not found: $PYEDNA_STRUCTURE_CONFIG"
    exit 1
fi

if [[ ! -f docking_config.cfg ]]; then
    echo "Error: docking_config.cfg not found."
    exit 1
fi

RUN_DIR="haddock/run"
rm -rf "$RUN_DIR"

echo "Starting HADDOCK..."
pyedna structure dock "$PYEDNA_STRUCTURE_CONFIG"

echo "Selecting and processing HADDOCK structures..."
pyedna structure finalize "$PYEDNA_STRUCTURE_CONFIG"

echo "Preparing Amber system..."
pyedna structure amber "$PYEDNA_STRUCTURE_CONFIG"

rm -f docking_config.cfg