#!/bin/bash

# USAGE:
# bash this_script.sh structure_name
# reuires files: sturcture_name.prmtop and structure_name.rst7

# check if argument provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <structure_name>"
    exit 1
fi

# Source conda environment AmberTools24
source activate AmberTools24

# Add path to AMBER executuable in AmberTools24 environment
export AMBERHOME="${CONDA_PREFIX}/amber24"
export PATH="$AMBERHOME/bin:$PATH"

# Get the structure name from the command line
STRUCTURE=$1  

# Create a temporary .leap file
LEAP_SCRIPT=$(mktemp)

# Write tleap commands to the temporary file
cat > "$LEAP_SCRIPT" << EOF
source leaprc.protein.ff14SB
mol = loadamberparams ${STRUCTURE}.prmtop
loadambercoords mol ${STRUCTURE}.rst7
savepdb mol ${STRUCTURE}.pdb
quit
EOF

# Run tleap with the generated script
tleap -f "$LEAP_SCRIPT"

# Remove the temporary file
rm "$LEAP_SCRIPT"