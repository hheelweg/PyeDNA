#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=forces
#SBATCH --output=slurm-%j.log

# USAGE:
#   sbatch this_script.sh <name>
#
# EXAMPLE:
#   sbatch this_script.sh dna_0nt
#
# REQUIRES:
#   - <name>.nc
#   - <name>.prmtop
#   - PYEDNA_HOME set
#   - cpptraj and sander in PATH
#   - python with ParmEd and netCDF4 installed

set -eo pipefail

# Require exactly one command-line argument.
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <name>"
    echo "Example: sbatch $0 dna_0nt"
    exit 1
fi

NAME="$1"

PRMTOP="${NAME}.prmtop"
TRAJECTORY="${NAME}.nc"

NONBOND_PRMTOP="${NAME}_nonbond.prmtop"
BOND_PRMTOP="${NAME}_bond.prmtop"

# Check required input files.
if [[ ! -f "$PRMTOP" ]]; then
    echo "Error: topology file '$PRMTOP' not found."
    exit 1
fi

if [[ ! -f "$TRAJECTORY" ]]; then
    echo "Error: trajectory file '$TRAJECTORY' not found."
    exit 1
fi

# Check the PyeDNA environment variable.
if [[ -z "${PYEDNA_HOME:-}" ]]; then
    echo "Error: PYEDNA_HOME is not set. Please set it in shell."
    exit 1
fi

CONFIG_FILE="$PYEDNA_HOME/config.sh"

if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Error: configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Generate the two modified topologies:
#   1. dye nonbonded interactions disabled
#   2. dye-DNA bonded interactions disabled
python -m modify_prmtop \
    "$PRMTOP" \
    "$NONBOND_PRMTOP" \
    "$BOND_PRMTOP"

echo
echo "Topology generation completed:"
echo "  Original:       $PRMTOP"
echo "  Nonbonded off:  $NONBOND_PRMTOP"
echo "  Bonded off:     $BOND_PRMTOP"