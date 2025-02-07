#!/bin/bash

# Check for correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 topology.prmtop trajectory.nc"
    exit 1
fi

# Assign input arguments
TOPOLOGY=$1
INPUT_TRAJ=$2
TEMP_TRAJ="${INPUT_TRAJ}.tmp"  # Temporary file for conversion

# Run cpptraj to convert NetCDF4 to NetCDF3 format
cpptraj -p "$TOPOLOGY" <<EOF
trajin $INPUT_TRAJ
trajout $TEMP_TRAJ netcdf
go
EOF

# Overwrite the original file with the converted one
mv "$TEMP_TRAJ" "$INPUT_TRAJ"

echo "Conversion complete: Overwritten $INPUT_TRAJ with NetCDF3 format."
