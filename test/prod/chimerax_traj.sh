#!/bin/bash


# Ensure two arguments (prmtop and nc file) are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <topology.prmtop> <trajectory.nc> [--export-obj]"
    exit 1
fi

export PRMTOP="$1"
export TRAJIN="$2"
export PDB="${PRMTOP%.prmtop}.pdb"
export DCD="${TRAJIN%.nc}.dcd"
export EXPORT_OBJ=false

# Check if the "--export-obj" flag is provided
if [[ "$3" == "--export-obj" ]]; then
    EXPORT_OBJ=true
fi

# (1) convert the AMBER trajectory to .dcd and extract a representative .pdb and remove solvent
# (1.1) create a cpptraj input script dynamically 
cat > convert_traj.in << EOF
parm $PRMTOP
trajin $TRAJIN
strip :WAT,HOH,CL,NA,K,MG,CA
trajout $DCD dcd
trajout $PDB pdb onlyframes 1
EOF
# (1.2) run cpptraj with a parameterized input file to convert .nc to .dcd file
cpptraj -i convert_traj.in

# (2) load trajectory into ChimeraX
# (2.1) create axuiliarx ChimeraX script
cat > chimera_script.cxc << EOF
open $PDB
open $DCD
EOF
# (2.2) make .obj trajectory files in subdirectory
if [[ "$EXPORT_OBJ" == true ]]; then
    OBJ_DIR="frames"
    mkdir -p "$OBJ_DIR"
    echo 'perframe "export format obj filename '"$OBJ_DIR"'/frame_{frame:04d}.obj"' >>  chimera_script.cxc
fi 
# (2.3) run chimera script
chimerax --nogui chimera_script.cxc


# chimerax --cmd "open $PDB; open $DCD"

# (3) remove files
rm -f convert_traj.in
rm -f chimera_script.cxc
#rm -f $DCD
#rm -f $PDB


# ----------- Note for usage -----------------------
# need to have AmberTools environment actiavted
# need ChimeraX as part of PATH variable (e.g. /usr/local/bin/chimerax)
# bash {this_file}.sh {prmtop_file}.prmtop {nc_file}.nc --export-obj