#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=forces
#SBATCH --output=slurm-%j.log

set -eo pipefail

# USAGE:
#   sbatch this_script.sh <name> <every_int>
#
# REQUIRES:
#   - <name>.nc
#   - <name>.prmtop
#   - PYEDNA_HOME set
#   - cpptraj and sander in PATH
#   - python with netCDF4 installed for merge step

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

THIN_TRAJ="${NAME}_thin_${EVERY_INT}.nc"
THIN_TOP="${NAME}_thin_${EVERY_INT}.prmtop"
CPPTRAJ_IN="${NAME}_thin_${EVERY_INT}.cpptraj.in"

FORCE_DIR="${NAME}_forces_tmp_frames"
TMP_DIR="${NAME}_forces_tmp_work"
FORCE_TEMPLATE="${TMP_DIR}/forces.in"
MERGED_FORCE_NC="${NAME}_forces.nc"

if [[ ! -f "$TOP" ]]; then
    echo "Error: Topology file $TOP not found."
    exit 1
fi

if [[ ! -f "$TRAJ" ]]; then
    echo "Error: Trajectory file $TRAJ not found."
    exit 1
fi

mkdir -p "$FORCE_DIR"
mkdir -p "$TMP_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Check merge dependency early
python - << 'EOF'
try:
    import netCDF4
except ImportError:
    raise SystemExit("Error: Python package 'netCDF4' is not installed in this environment.")
EOF

# Write force input locally
cat > "$FORCE_TEMPLATE" << 'EOF'
force evaluation via 1 MD step
&cntrl
  imin      = 0,
  ntx       = 1,
  irest     = 0,
  nstlim    = 1,
  dt        = 0.0,
  ntb       = 1,
  ntp       = 0,
  ntc       = 1,
  ntf       = 1,
  ntpr      = 1,
  ntwx      = 0,
  ntwf      = 1,
  ioutfm    = 1,
  cut       = 8.0,
/
EOF

echo "Using force input file:"
nl -ba "$FORCE_TEMPLATE"
cat -A "$FORCE_TEMPLATE"

# Decide whether to thin or not
if [[ "$EVERY_INT" -eq 1 ]]; then
    echo "EVERY_INT = 1, skipping thinning and using original trajectory."
    WORK_TRAJ="$TRAJ"
    WORK_TOP="$TOP"
else
    echo "Running cpptraj to create thinned trajectory: $THIN_TRAJ"

    cat > "$CPPTRAJ_IN" << EOF
parm $TOP
trajin $TRAJ 1 last $EVERY_INT
trajout $THIN_TRAJ netcdf
run
quit
EOF

    cpptraj -i "$CPPTRAJ_IN"

    cp -f "$TOP" "$THIN_TOP"

    WORK_TRAJ="$THIN_TRAJ"
    WORK_TOP="$THIN_TOP"
fi

# Count frames in working trajectory
NFRAMES=$(ncdump -h "$WORK_TRAJ" | awk '
/frame = UNLIMITED/ {
    if (match($0, /\(([0-9]+) currently\)/)) {
        s = substr($0, RSTART+1, RLENGTH-11)
        print s
    }
}')

if [[ -z "$NFRAMES" ]]; then
    echo "Error: Could not determine number of frames in $WORK_TRAJ"
    exit 1
fi

echo "Working trajectory contains $NFRAMES frames."

# Loop over frames
for (( i=1; i<=NFRAMES; i++ )); do
    FRAME_TAG=$(printf "%06d" "$i")
    FRAME_RST="${TMP_DIR}/frame_${FRAME_TAG}.rst7"
    FRAME_CPPTRAJ_IN="${TMP_DIR}/frame_${FRAME_TAG}.cpptraj.in"

    FRAME_OUT="${FORCE_DIR}/frame_${FRAME_TAG}.out"
    FRAME_INFO="${FORCE_DIR}/frame_${FRAME_TAG}.mdinfo"
    FRAME_RESTART="${FORCE_DIR}/frame_${FRAME_TAG}.rst7"
    FRAME_FORCE="${FORCE_DIR}/frame_${FRAME_TAG}.nc"

    echo "Processing frame $i / $NFRAMES"

    cat > "$FRAME_CPPTRAJ_IN" << EOF
parm $WORK_TOP
trajin $WORK_TRAJ $i $i
trajout $FRAME_RST restart
run
quit
EOF

    cpptraj -i "$FRAME_CPPTRAJ_IN"

    sander -O \
      -i "$FORCE_TEMPLATE" \
      -p "$WORK_TOP" \
      -c "$FRAME_RST" \
      -o "$FRAME_OUT" \
      -r "$FRAME_RESTART" \
      -inf "$FRAME_INFO" \
      -frc "$FRAME_FORCE"

    rm -f "$FRAME_CPPTRAJ_IN"
    rm -f "$FRAME_RST"
done

# Merge all per-frame force NetCDF files into one force trajectory
echo "Merging per-frame force files into: $MERGED_FORCE_NC"

python - << EOF
import glob
from netCDF4 import Dataset

files = sorted(glob.glob("${FORCE_DIR}/frame_*.nc"))
if not files:
    raise SystemExit("Error: no per-frame force files found to merge.")

with Dataset(files[0], "r") as src0:
    natom = len(src0.dimensions["atom"])
    nspatial = len(src0.dimensions["spatial"])
    spatial_vals = src0.variables["spatial"][:]

with Dataset("${MERGED_FORCE_NC}", "w", format="NETCDF4_CLASSIC") as dst:
    dst.createDimension("frame", None)
    dst.createDimension("atom", natom)
    dst.createDimension("spatial", nspatial)

    time_var = dst.createVariable("time", "f4", ("frame",))
    time_var.units = "picosecond"

    spatial_var = dst.createVariable("spatial", "S1", ("spatial",))
    spatial_var[:] = spatial_vals

    forces_var = dst.createVariable("forces", "f4", ("frame", "atom", "spatial"))
    forces_var.units = "kilocalorie/mole/angstrom"

    frame_index_var = dst.createVariable("frame_index", "i4", ("frame",))

    dst.title = "merged_force_trajectory"
    dst.application = "AMBER"
    dst.program = "sander + python merge"
    dst.Conventions = "AMBER"
    dst.ConventionVersion = "1.0"

    for i, fn in enumerate(files):
        with Dataset(fn, "r") as src:
            if len(src.dimensions["frame"]) != 1:
                raise SystemExit(f"Error: {fn} does not contain exactly 1 frame.")
            time_var[i] = src.variables["time"][0]
            forces_var[i, :, :] = src.variables["forces"][0, :, :]
            frame_index_var[i] = i + 1
EOF

# Clean temporary and auxiliary files/directories
echo "Cleaning temporary and auxiliary files..."

rm -f "$FORCE_TEMPLATE"

if [[ "$EVERY_INT" -ne 1 ]]; then
    rm -f "$CPPTRAJ_IN"
fi

# Remove per-frame temporary outputs after successful merge
rm -f "${FORCE_DIR}"/frame_*.nc
rm -f "${FORCE_DIR}"/frame_*.out
rm -f "${FORCE_DIR}"/frame_*.mdinfo
rm -f "${FORCE_DIR}"/frame_*.rst7
rmdir "$FORCE_DIR" 2>/dev/null || true

# Remove temp working directory
rm -f "${TMP_DIR}"/*
rmdir "$TMP_DIR" 2>/dev/null || true

echo "Done."

if [[ "$EVERY_INT" -eq 1 ]]; then
    echo "Kept coordinate trajectory: $TRAJ"
    echo "Kept topology:             $TOP"
else
    echo "Kept coordinate trajectory: $THIN_TRAJ"
    echo "Kept matching topology:     $THIN_TOP"
fi

echo "Kept merged force file:     $MERGED_FORCE_NC"