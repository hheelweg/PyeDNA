#!/bin/bash

# USAGE:
# bash this_script.sh

# # Source conda environment AmberTools24
# source activate AmberTools24

# # Add path to AMBER executuable in AmberTools24 environment
# export AMBERHOME="${CONDA_PREFIX}/amber24"
# export PATH="$AMBERHOME/bin:$PATH"

# # Add path to PyeDNA and define PyeDNA home
# export PYTHONPATH=$PYTHONPATH:/home/hheelweg/cy3cy5/PyeDNA/scripts
# export PYEDNA_HOME="/home/hheelweg/cy3cy5/PyeDNA"

# # Add path to dye library for structural information of dyes
# export DYE_DIR="/home/hheelweg/cy3cy5/bib"

# Load user-defined paths
CONFIG_FILE="$(dirname "$0")/../config.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file ($CONFIG_FILE) not found!"
    exit 1
fi


# run python module for structure creation
python -m create_structure > output.log 2>&1
