#!/bin/bash

# USAGE:
# bash create_structure.sh

if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

CONFIG_FILE="$PYEDNA_HOME/config.sh"

if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file ($CONFIG_FILE) not found."
    exit 1
fi

# Prepare all structure/HADDOCK input files
python -m create_structure > output.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "Error: structure preparation failed. See output.log."
    exit 1
fi

# Submit HADDOCK job
sbatch "$PYEDNA_HOME/scripts/haddock/run_haddock.sh"