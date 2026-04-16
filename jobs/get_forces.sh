#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=normal                      # CPU partition	
#SBATCH --ntasks=1                              # # of tasks
#SBATCH --cpus-per-task=8                       # use 4-8 CPUs per GPU
#SBATCH --job-name=forces                       # Use provided job name or "default_job" if none given
#SBATCH --output=slurm-%j.log                   # Name output log file

# USAGE:
# sbatch this_script.sh [name] [every_int]
# NOTE : reequires name.nc and name.prmtop file

# Check if PYEDNA_HOME is set
if [[ -z "${PYEDNA_HOME:-}" ]]; then
    echo "Error: PYEDNA_HOME is not set. Please set it in shell."
    exit 1
fi

CONFIG_FILE="$PYEDNA_HOME/config.sh"

if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file ($CONFIG_FILE) not found!"
    exit 1
fi

# check input arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 <name> <every_int>"
    exit 1
fi

NAME="$1"
EVERY_INT="$2"


TOP="${NAME}.prmtop"
TRAJ="${NAME}.nc"
FORCE_TEMPLATE="$PYEDNA_HOME/data/md_templates/forces.in"

THIN_TRAJ="${NAME}_thin_${EVERY_INT}.nc"
CPPTRAJ_IN="${NAME}_thin_${EVERY_INT}.cpptraj.in"

OUT_LOG="${NAME}_forces.out"
FORCE_TRAJ="${NAME}_forces.mdfrc"
RESTART_OUT="${NAME}_forces.rst7"
INFO_OUT="${NAME}_forces.mdinfo"


# check required files
if [[ ! -f "$TOP" ]]; then
    echo "Error: Topology file $TOP not found."
    exit 1
fi

if [[ ! -f "$TRAJ" ]]; then
    echo "Error: Trajectory file $TRAJ not found."
    exit 1
fi

if [[ ! -f "$FORCE_TEMPLATE" ]]; then
    echo "Error: Force template $FORCE_TEMPLATE not found."
    exit 1
fi

# (1) Thin trajectory
cat > "$CPPTRAJ_IN" << EOF
parm $TOP
trajin $TRAJ 1 last $EVERY_INT
trajout $THIN_TRAJ netcdf
run
quit
EOF

echo "Running cpptraj to create thinned trajectory: $THIN_TRAJ"
cpptraj -i "$CPPTRAJ_IN"


# (2) Run force evaluation
echo "Running sander force evaluation on thinned trajectory..."
sander -O \
    -i "$FORCE_TEMPLATE" \
    -p "$TOP" \
    -c "$THIN_TRAJ" \
    -y "$THIN_TRAJ" \
    -o "$OUT_LOG" \
    -r "$RESTART_OUT" \
    -inf "$INFO_OUT" \
    -frc "$FORCE_TRAJ"


# (3) Clean temporary files
echo "Cleaning temporary files..."
rm -f "$CPPTRAJ_IN"
rm -f "$THIN_TRAJ"


