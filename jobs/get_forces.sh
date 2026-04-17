#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=forces
#SBATCH --output=slurm-%j.log

set -euo pipefail

# USAGE:
# sbatch this_script.sh [name] [every_int]
# NOTE: requires name.nc and name.prmtop file

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

if ! [[ "$EVERY_INT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: every_int must be a positive integer."
    exit 1
fi

TOP="${NAME}.prmtop"
TRAJ="${NAME}.nc"
FORCE_TEMPLATE="$PYEDNA_HOME/data/md_templates/forces.in"

THIN_TRAJ="${NAME}_thin_${EVERY_INT}.nc"
CPPTRAJ_IN="${NAME}_thin_${EVERY_INT}.cpptraj.in"

FORCE_DIR="${NAME}_forces_every_${EVERY_INT}"
TMP_DIR="${NAME}_forces_tmp_every_${EVERY_INT}"

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

mkdir -p "$FORCE_DIR"
mkdir -p "$TMP_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

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

# Count frames in thinned trajectory
NFRAMES=$(ncdump -h "$THIN_TRAJ" | awk '
/frame = UNLIMITED/ {
    if (match($0, /\(([0-9]+) currently\)/)) {
        s = substr($0, RSTART+1, RLENGTH-11)
        print s
    }
}')

if [[ -z "$NFRAMES" ]]; then
    echo "Error: Could not determine number of frames in $THIN_TRAJ"
    exit 1
fi

echo "Thinned trajectory contains $NFRAMES frames."

# (2) Loop over frames: extract one restart, run sander, store one force file
for (( i=1; i<=NFRAMES; i++ )); do
    FRAME_TAG=$(printf "%06d" "$i")
    FRAME_RST="$TMP_DIR/frame_${FRAME_TAG}.rst7"
    FRAME_CPPTRAJ_IN="$TMP_DIR/frame_${FRAME_TAG}.cpptraj.in"

    FRAME_OUT="$FORCE_DIR/frame_${FRAME_TAG}.out"
    FRAME_INFO="$FORCE_DIR/frame_${FRAME_TAG}.mdinfo"
    FRAME_RESTART="$FORCE_DIR/frame_${FRAME_TAG}.rst7"
    FRAME_FORCE="$FORCE_DIR/frame_${FRAME_TAG}.nc"

    echo "Processing frame $i / $NFRAMES"

    cat > "$FRAME_CPPTRAJ_IN" << EOF
parm $TOP
trajin $THIN_TRAJ $i $i
trajout $FRAME_RST restart
run
quit
EOF

    cpptraj -i "$FRAME_CPPTRAJ_IN"

    sander -O \
      -i "$FORCE_TEMPLATE" \
      -p "$TOP" \
      -c "$FRAME_RST" \
      -o "$FRAME_OUT" \
      -r "$FRAME_RESTART" \
      -inf "$FRAME_INFO" \
      -frc "$FRAME_FORCE"

    rm -f "$FRAME_CPPTRAJ_IN"
    rm -f "$FRAME_RST"
done

# (3) Clean temporary files
echo "Cleaning temporary files..."
rm -f "$CPPTRAJ_IN"
rm -f "$THIN_TRAJ"
rmdir "$TMP_DIR" 2>/dev/null || true

echo "Done."
echo "Force files written to: $FORCE_DIR"