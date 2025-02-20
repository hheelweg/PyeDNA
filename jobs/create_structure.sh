#!/bin/bash

# USAGE:
# bash this_script.sh

# source conda environment
source activate AmberTools24

# Add path to PyeDNA and define PyeDNA home
export PYTHONPATH=$PYTHONPATH:/home/hheelweg/cy3cy5/PyeDNA/scripts
export PYEDNA_HOME="/home/hheelweg/cy3cy5/PyeDNA"

# Add path to dye library for structural information of dyes
export DYE_DIR="/home/hheelweg/cy3cy5/bib"


# run file for structure creation
python -m create_structure > output.log 2>&1
