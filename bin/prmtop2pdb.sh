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