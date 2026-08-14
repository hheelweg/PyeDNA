#!/bin/bash

# USAGE
# -----
# bash "$PYEDNA_HOME/jobs/create_structure.sh" [STRUCTURE_CONFIG]
#
# STRUCTURE_CONFIG is optional and defaults to "structure.toml" in the current
# working directory. The selected file is used for both HADDOCK preparation and
# finalization, including when finalization runs later in the submitted job.
#
# Examples:
#   bash "$PYEDNA_HOME/jobs/create_structure.sh"
#   bash "$PYEDNA_HOME/jobs/create_structure.sh" my_structure.toml

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [STRUCTURE_CONFIG]"
    exit 1
fi

PYEDNA_STRUCTURE_CONFIG="${1:-structure.toml}"

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

if [[ ! -f "$PYEDNA_STRUCTURE_CONFIG" ]]; then
    echo "Error: structure configuration not found: $PYEDNA_STRUCTURE_CONFIG"
    exit 1
fi

# Prepare all structure/HADDOCK input files
python "$PYEDNA_HOME/scripts/create_structure.py" prepare \
    --config "$PYEDNA_STRUCTURE_CONFIG" > structure_prepare.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "Error: structure preparation failed. See structure_prepare.log."
    exit 1
fi

# Submit HADDOCK job
sbatch --export=ALL,PYEDNA_STRUCTURE_CONFIG="$PYEDNA_STRUCTURE_CONFIG" \
    "$PYEDNA_HOME/jobs/run_haddock.sh"
