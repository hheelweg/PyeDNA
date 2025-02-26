#!/bin/bash

# NOTE : convert .prmtop & .rst7 into .pdb 

# USAGE:
# bash this_script.sh structure_name
# requires files: sturcture_name.prmtop and structure_name.rst7

# check if argument provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <structure_name>"
    exit 1
fi

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

# Get the structure name from the command line
STRUCTURE=$1  

# Create a temporary cpptraj input file
CPPTRAJ_SCRIPT=$(mktemp)

# Write cpptraj commands to the temporary file
cat > "$CPPTRAJ_SCRIPT" << EOF
parm ${STRUCTURE}.prmtop
trajin ${STRUCTURE}.rst7
trajout ${STRUCTURE}.pdb pdb
EOF

# Run cpptraj with the generated script
cpptraj -i "$CPPTRAJ_SCRIPT"

# Remove the temporary file
rm "$CPPTRAJ_SCRIPT"