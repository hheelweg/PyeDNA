#!/bin/bash

# Ensure correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 topology.prmtop trajectory.nc"
    exit 1
fi

export PRMTOP="$1"
export TRAJIN="$2"
export TRAJOUT="${TRAJIN%.nc}.dcd"

# Create a cpptraj input script dynamically
cat > convert_traj.in << EOF
parm $PRMTOP
trajin $TRAJIN
trajout $TRAJOUT dcd
EOF

# Run cpptraj with a parameterized input file
cpptraj -i convert_traj.in

# Remove input script
rm -f convert_traj.in

# ----------- Note for usage -----------------------
# need to have AmberTools environment actiavtes
# bash {this_file}.sh {prmtop_file}.prmtop {nc_file}.nc