#!/bin/bash

#SBATCH --nodes=3
#SBATCH --partition=normal
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=forces
#SBATCH --output=slurm-%j.log

# ============================================================================
# Force evaluation for:
#
#   1. original topology
#   2. nonbonded-off topology
#   3. bonded-off topology
#
# USAGE:
#
#   sbatch this_script.sh <name>
#
# EXAMPLE:
#
#   sbatch this_script.sh dna_0nt
#
# INPUT:
#
#   dna_0nt.prmtop
#   dna_0nt.nc
#
# OUTPUT:
#
#   dna_0nt_nonbond.prmtop
#   dna_0nt_bond.prmtop
#
#   dna_0nt_forces.nc
#   dna_0nt_nonbond_forces.nc
#   dna_0nt_bond_forces.nc
#
# The force calculations follow the established get_forces.sh procedure:
#
#   1. extract each trajectory frame as an Amber restart;
#   2. run one sander MD step with dt = 0;
#   3. write one force NetCDF per frame;
#   4. merge the per-frame NetCDF files;
#   5. remove all temporary files and directories.
#
# The three topology calculations run concurrently.
# ============================================================================

set -eo pipefail


# ----------------------------------------------------------------------------
# Read command-line argument
# ----------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <name>"
    echo "Example: sbatch $0 dna_0nt"
    exit 1
fi

NAME="$1"

# For the current test, only use the first 10 trajectory frames.
NFRAMES_REQUESTED=10


# ----------------------------------------------------------------------------
# Input and output names
# ----------------------------------------------------------------------------

PRMTOP="${NAME}.prmtop"
TRAJECTORY="${NAME}.nc"

NONBOND_PRMTOP="${NAME}_nonbond.prmtop"
BOND_PRMTOP="${NAME}_bond.prmtop"

FULL_FORCES="${NAME}_forces.nc"
NONBOND_FORCES="${NAME}_nonbond_forces.nc"
BOND_FORCES="${NAME}_bond_forces.nc"


# ----------------------------------------------------------------------------
# Load environment
# ----------------------------------------------------------------------------

if [[ -z "${PYEDNA_HOME:-}" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

CONFIG_FILE="$PYEDNA_HOME/config.sh"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Conda activation/deactivation scripts can reference unset variables.
# Therefore, do not enable nounset until after sourcing the configuration.
set +u
source "$CONFIG_FILE"
set -u


# ----------------------------------------------------------------------------
# Check input files
# ----------------------------------------------------------------------------

if [[ ! -f "$PRMTOP" ]]; then
    echo "Error: topology file '$PRMTOP' not found."
    exit 1
fi

if [[ ! -f "$TRAJECTORY" ]]; then
    echo "Error: trajectory file '$TRAJECTORY' not found."
    exit 1
fi


# ----------------------------------------------------------------------------
# Check required executables
# ----------------------------------------------------------------------------

for PROGRAM in python cpptraj sander ncdump realpath; do
    if ! command -v "$PROGRAM" >/dev/null 2>&1; then
        echo "Error: required program '$PROGRAM' is not available."
        exit 1
    fi
done


# ----------------------------------------------------------------------------
# Check Python dependencies
# ----------------------------------------------------------------------------

python - <<'PY'
try:
    import netCDF4
    import parmed
except ImportError as exc:
    raise SystemExit(
        f"Error: required Python package is unavailable: {exc}"
    )
PY


# ----------------------------------------------------------------------------
# Generate modified topologies
# ----------------------------------------------------------------------------

echo "Generating modified topologies..."

python -m modify_prmtop \
    "$PRMTOP" \
    "$NONBOND_PRMTOP" \
    "$BOND_PRMTOP"

if [[ ! -s "$NONBOND_PRMTOP" ]]; then
    echo "Error: modified topology '$NONBOND_PRMTOP' was not created."
    exit 1
fi

if [[ ! -s "$BOND_PRMTOP" ]]; then
    echo "Error: modified topology '$BOND_PRMTOP' was not created."
    exit 1
fi


# ----------------------------------------------------------------------------
# Absolute paths
#
# Worker processes change into temporary directories, so all important paths
# are converted to absolute paths.
# ----------------------------------------------------------------------------

SUBMIT_DIR="$(pwd)"

PRMTOP_ABS="$(realpath "$PRMTOP")"
TRAJECTORY_ABS="$(realpath "$TRAJECTORY")"

NONBOND_PRMTOP_ABS="$(realpath "$NONBOND_PRMTOP")"
BOND_PRMTOP_ABS="$(realpath "$BOND_PRMTOP")"

FULL_FORCES_ABS="$SUBMIT_DIR/$FULL_FORCES"
NONBOND_FORCES_ABS="$SUBMIT_DIR/$NONBOND_FORCES"
BOND_FORCES_ABS="$SUBMIT_DIR/$BOND_FORCES"


# ----------------------------------------------------------------------------
# Shared temporary root
#
# This is created in the submission directory so that it is visible from all
# three allocated nodes. Do not use node-local /tmp for a multi-node job.
# ----------------------------------------------------------------------------

TEMP_ROOT="$SUBMIT_DIR/.${NAME}_forces_tmp_${SLURM_JOB_ID}"

SHORT_TRAJECTORY="$TEMP_ROOT/${NAME}_first_frames.nc"
SHORT_TRAJECTORY_INPUT="$TEMP_ROOT/extract_first_frames.cpptraj.in"

FORCE_INPUT="$TEMP_ROOT/forces.in"
WORKER_SCRIPT="$TEMP_ROOT/run_force_worker.sh"

FULL_ROOT="$TEMP_ROOT/full"
NONBOND_ROOT="$TEMP_ROOT/nonbond"
BOND_ROOT="$TEMP_ROOT/bond"

mkdir -p "$FULL_ROOT"
mkdir -p "$NONBOND_ROOT"
mkdir -p "$BOND_ROOT"


# ----------------------------------------------------------------------------
# Cleanup
#
# The complete temporary directory is removed whenever the job exits,
# including after a failure or cancellation handled by the shell.
# ----------------------------------------------------------------------------

cleanup() {
    if [[ -n "${TEMP_ROOT:-}" && -d "$TEMP_ROOT" ]]; then
        echo "Removing temporary directory:"
        echo "  $TEMP_ROOT"
        rm -rf "$TEMP_ROOT"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


# ----------------------------------------------------------------------------
# Count available trajectory frames using cpptraj
#
# This works even when the file is readable by Amber/cpptraj but is not a
# NetCDF format understood by the installed ncdump executable.
# ----------------------------------------------------------------------------

TOTAL_FRAMES=$(
    cpptraj -p "$PRMTOP_ABS" -y "$TRAJECTORY_ABS" -tl 2>/dev/null |
    awk '/Frames:/ {print $2; exit}'
)

if [[ -z "$TOTAL_FRAMES" || ! "$TOTAL_FRAMES" =~ ^[0-9]+$ ]]; then
    echo "Error: cpptraj could not determine the number of frames in:"
    echo "  $TRAJECTORY"
    exit 1
fi

if (( TOTAL_FRAMES < 1 )); then
    echo "Error: trajectory '$TRAJECTORY' contains no frames."
    exit 1
fi

if (( TOTAL_FRAMES < NFRAMES_REQUESTED )); then
    NFRAMES="$TOTAL_FRAMES"
else
    NFRAMES="$NFRAMES_REQUESTED"
fi

echo "Trajectory contains $TOTAL_FRAMES frames."
echo "Using the first $NFRAMES frames."


# ----------------------------------------------------------------------------
# Create a temporary trajectory containing only the desired frames
# ----------------------------------------------------------------------------

echo "Extracting the first $NFRAMES frames..."

cat > "$SHORT_TRAJECTORY_INPUT" <<EOF
parm $PRMTOP_ABS
trajin $TRAJECTORY_ABS 1 $NFRAMES 1
trajout $SHORT_TRAJECTORY netcdf
run
quit
EOF

cpptraj -i "$SHORT_TRAJECTORY_INPUT" \
    > "$TEMP_ROOT/extract_first_frames.log" 2>&1

if [[ ! -s "$SHORT_TRAJECTORY" ]]; then
    echo "Error: temporary trajectory was not created."
    cat "$TEMP_ROOT/extract_first_frames.log"
    exit 1
fi


# ----------------------------------------------------------------------------
# Force-evaluation input
#
# This is the force-evaluation procedure from the established get_forces.sh.
#
# One MD step is requested, but dt=0 means the coordinates do not advance.
# The forces corresponding to the supplied restart coordinates are written
# through -frc.
# ----------------------------------------------------------------------------

cat > "$FORCE_INPUT" <<'EOF'
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


# ----------------------------------------------------------------------------
# Worker script
#
# One independent copy of this worker is launched for each topology.
#
# Temporary directory structure for each topology:
#
#   full/
#       work/
#           frame_000001.rst7
#           frame_000001.cpptraj.in
#           ...
#
#       force_frames/
#           frame_000001.nc
#           frame_000001.out
#           frame_000001.mdinfo
#           frame_000001.rst7
#           ...
#
# The worker merges the per-frame force files into the requested final
# trajectory. The top-level cleanup trap later removes all temporary files.
# ----------------------------------------------------------------------------

cat > "$WORKER_SCRIPT" <<'WORKER_EOF'
#!/bin/bash

set -eo pipefail

LABEL="$1"
TOPOLOGY="$2"
TRAJECTORY="$3"
FORCE_INPUT="$4"
NFRAMES="$5"
TOPOLOGY_ROOT="$6"
FINAL_FORCE_FILE="$7"

WORK_DIR="$TOPOLOGY_ROOT/work"
FORCE_DIR="$TOPOLOGY_ROOT/force_frames"

mkdir -p "$WORK_DIR"
mkdir -p "$FORCE_DIR"

echo "[$LABEL] Beginning force evaluation."
echo "[$LABEL] Topology: $TOPOLOGY"
echo "[$LABEL] Frames:   $NFRAMES"


# --------------------------------------------------------------------------
# Evaluate one trajectory frame at a time
# --------------------------------------------------------------------------

for (( i=1; i<=NFRAMES; i++ )); do

    FRAME_TAG=$(printf "%06d" "$i")

    FRAME_RST="$WORK_DIR/frame_${FRAME_TAG}.rst7"
    FRAME_CPPTRAJ_INPUT="$WORK_DIR/frame_${FRAME_TAG}.cpptraj.in"
    FRAME_CPPTRAJ_LOG="$WORK_DIR/frame_${FRAME_TAG}.cpptraj.log"

    FRAME_OUT="$FORCE_DIR/frame_${FRAME_TAG}.out"
    FRAME_INFO="$FORCE_DIR/frame_${FRAME_TAG}.mdinfo"
    FRAME_RESTART="$FORCE_DIR/frame_${FRAME_TAG}.rst7"
    FRAME_FORCE="$FORCE_DIR/frame_${FRAME_TAG}.nc"

    echo "[$LABEL] Processing frame $i / $NFRAMES"

    # Extract one coordinate frame as an Amber restart.
    cat > "$FRAME_CPPTRAJ_INPUT" <<EOF
parm $TOPOLOGY
trajin $TRAJECTORY $i $i
trajout $FRAME_RST restart
run
quit
EOF

    cpptraj -i "$FRAME_CPPTRAJ_INPUT" \
        > "$FRAME_CPPTRAJ_LOG" 2>&1

    if [[ ! -s "$FRAME_RST" ]]; then
        echo "[$LABEL] Error: failed to extract restart for frame $i."
        cat "$FRAME_CPPTRAJ_LOG"
        exit 1
    fi

    # Run exactly the same single-frame sander force evaluation used in
    # the established get_forces.sh.
    sander -O \
        -i "$FORCE_INPUT" \
        -p "$TOPOLOGY" \
        -c "$FRAME_RST" \
        -o "$FRAME_OUT" \
        -r "$FRAME_RESTART" \
        -inf "$FRAME_INFO" \
        -frc "$FRAME_FORCE"

    if [[ ! -s "$FRAME_FORCE" ]]; then
        echo "[$LABEL] Error: force file was not created for frame $i."
        exit 1
    fi

    # The extracted input restart and cpptraj files are no longer needed.
    rm -f "$FRAME_CPPTRAJ_INPUT"
    rm -f "$FRAME_CPPTRAJ_LOG"
    rm -f "$FRAME_RST"

done


# --------------------------------------------------------------------------
# Merge all single-frame force files
#
# This follows the merge logic from get_forces.sh.
# --------------------------------------------------------------------------

echo "[$LABEL] Merging per-frame force files..."

rm -f "$FINAL_FORCE_FILE"

python - "$FORCE_DIR" "$FINAL_FORCE_FILE" <<'PY'
import glob
import os
import sys

from netCDF4 import Dataset


force_directory = sys.argv[1]
output_file = sys.argv[2]

files = sorted(
    glob.glob(os.path.join(force_directory, "frame_*.nc"))
)

if not files:
    raise SystemExit(
        f"Error: no per-frame force files found in {force_directory}"
    )

with Dataset(files[0], "r") as source:
    natom = len(source.dimensions["atom"])
    nspatial = len(source.dimensions["spatial"])
    spatial_values = source.variables["spatial"][:]

with Dataset(output_file, "w", format="NETCDF4_CLASSIC") as destination:
    destination.createDimension("frame", None)
    destination.createDimension("atom", natom)
    destination.createDimension("spatial", nspatial)

    time_variable = destination.createVariable(
        "time",
        "f4",
        ("frame",),
    )
    time_variable.units = "picosecond"

    spatial_variable = destination.createVariable(
        "spatial",
        "S1",
        ("spatial",),
    )
    spatial_variable[:] = spatial_values

    force_variable = destination.createVariable(
        "forces",
        "f4",
        ("frame", "atom", "spatial"),
    )
    force_variable.units = "kilocalorie/mole/angstrom"

    frame_index_variable = destination.createVariable(
        "frame_index",
        "i4",
        ("frame",),
    )

    destination.title = "merged_force_trajectory"
    destination.application = "AMBER"
    destination.program = "sander + python merge"
    destination.Conventions = "AMBER"
    destination.ConventionVersion = "1.0"

    for frame_index, filename in enumerate(files):
        with Dataset(filename, "r") as source:
            if len(source.dimensions["frame"]) != 1:
                raise SystemExit(
                    f"Error: {filename} does not contain exactly one frame."
                )

            time_variable[frame_index] = source.variables["time"][0]

            force_variable[frame_index, :, :] = (
                source.variables["forces"][0, :, :]
            )

            frame_index_variable[frame_index] = frame_index + 1

print(f"Merged {len(files)} force frames into {output_file}")
PY

if [[ ! -s "$FINAL_FORCE_FILE" ]]; then
    echo "[$LABEL] Error: merged force file was not created."
    exit 1
fi

echo "[$LABEL] Completed successfully:"
echo "[$LABEL] $FINAL_FORCE_FILE"
WORKER_EOF

chmod +x "$WORKER_SCRIPT"


# ----------------------------------------------------------------------------
# Remove old final force trajectories
# ----------------------------------------------------------------------------

rm -f \
    "$FULL_FORCES_ABS" \
    "$NONBOND_FORCES_ABS" \
    "$BOND_FORCES_ABS"


# ----------------------------------------------------------------------------
# Start the three independent topology evaluations concurrently
#
# Each srun receives one node and one serial sander worker.
#
# The frame loop within a topology remains sequential, exactly as in the
# established get_forces.sh. The parallelism is across the three topologies.
# ----------------------------------------------------------------------------

echo "Starting three parallel force evaluations..."


srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    "$WORKER_SCRIPT" \
    "full" \
    "$PRMTOP_ABS" \
    "$SHORT_TRAJECTORY" \
    "$FORCE_INPUT" \
    "$NFRAMES" \
    "$FULL_ROOT" \
    "$FULL_FORCES_ABS" \
    > "$FULL_ROOT/worker.log" 2>&1 &

PID_FULL=$!


srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    "$WORKER_SCRIPT" \
    "nonbonded-off" \
    "$NONBOND_PRMTOP_ABS" \
    "$SHORT_TRAJECTORY" \
    "$FORCE_INPUT" \
    "$NFRAMES" \
    "$NONBOND_ROOT" \
    "$NONBOND_FORCES_ABS" \
    > "$NONBOND_ROOT/worker.log" 2>&1 &

PID_NONBOND=$!


srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    "$WORKER_SCRIPT" \
    "bonded-off" \
    "$BOND_PRMTOP_ABS" \
    "$SHORT_TRAJECTORY" \
    "$FORCE_INPUT" \
    "$NFRAMES" \
    "$BOND_ROOT" \
    "$BOND_FORCES_ABS" \
    > "$BOND_ROOT/worker.log" 2>&1 &

PID_BOND=$!


# ----------------------------------------------------------------------------
# Wait for all three workers
# ----------------------------------------------------------------------------

STATUS=0

if ! wait "$PID_FULL"; then
    echo
    echo "Error: full-system force evaluation failed."
    echo "Worker log:"
    cat "$FULL_ROOT/worker.log"
    STATUS=1
fi

if ! wait "$PID_NONBOND"; then
    echo
    echo "Error: nonbonded-off force evaluation failed."
    echo "Worker log:"
    cat "$NONBOND_ROOT/worker.log"
    STATUS=1
fi

if ! wait "$PID_BOND"; then
    echo
    echo "Error: bonded-off force evaluation failed."
    echo "Worker log:"
    cat "$BOND_ROOT/worker.log"
    STATUS=1
fi

if [[ "$STATUS" -ne 0 ]]; then
    echo
    echo "One or more force evaluations failed."
    echo "Temporary files will now be removed."
    exit 1
fi


# ----------------------------------------------------------------------------
# Verify final force trajectories
# ----------------------------------------------------------------------------

for FORCE_FILE in \
    "$FULL_FORCES_ABS" \
    "$NONBOND_FORCES_ABS" \
    "$BOND_FORCES_ABS"
do
    if [[ ! -s "$FORCE_FILE" ]]; then
        echo "Error: expected force file was not created:"
        echo "  $FORCE_FILE"
        exit 1
    fi
done


# ----------------------------------------------------------------------------
# Completion
#
# The cleanup trap removes:
#
#   - the shortened temporary coordinate trajectory;
#   - the force input;
#   - the worker script;
#   - all extracted per-frame restart files;
#   - all cpptraj input and log files;
#   - all individual force NetCDF files;
#   - all sander .out files;
#   - all sander .mdinfo files;
#   - all sander-generated restart files;
#   - all topology-specific temporary subdirectories;
#   - the complete temporary root directory.
# ----------------------------------------------------------------------------

echo
echo "Force evaluation completed for the first $NFRAMES frames:"
echo "  Full system:       $FULL_FORCES"
echo "  Nonbonded off:     $NONBOND_FORCES"
echo "  Bonded off:        $BOND_FORCES"