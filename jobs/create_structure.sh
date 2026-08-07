#!/bin/bash

# USAGE:
# bash this_script.sh

# Check if PYEDNA_HOME is set
if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set. Please set it in shell."
    exit 1
fi

# Load config.sh from the root of PyeDNA to set user-specific environment variables
CONFIG_FILE="$PYEDNA_HOME/config.sh"

if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file ($CONFIG_FILE) not found!"
    exit 1
fi

# run python module for structure creation
python -m create_structure > output.log 2>&1