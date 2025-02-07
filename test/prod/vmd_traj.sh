#!/bin/bash

# Ensure correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 topology.prmtop trajectory.nc"
    exit 1
fi

export PRMTOP="$1"
export TRAJIN="$2"
export TRAJOUT="${TRAJIN%.nc}.dcd"


## (1) convert .nc to .dcd for VMD 
# (1.1) create a cpptraj input script dynamically
cat > convert_traj.in << EOF
parm $PRMTOP
trajin $TRAJIN
trajout $TRAJOUT dcd
EOF
# (1.2) run cpptraj with a parameterized input file to convert .nc to .dcd file
cpptraj -i convert_traj.in

## (2) load trajectory into VMD
# (2.1) find the correct VMD binary path (adjust if necessary)
VMD_BIN="/Applications/VMD.app/Contents/MacOS/startup.command"
# (2.2) run VMD using the correct binary
"$VMD_BIN" -e <(echo "mol new $PRMTOP; mol addfile $TRAJOUT waitfor all")

## (3) remove files to clean up
rm -f convert_traj.in
rm -f "$TRAJOUT"


# ----------- Note for usage -----------------------
# need to have AmberTools environment actiavted
# bash {this_file}.sh {prmtop_file}.prmtop {nc_file}.nc