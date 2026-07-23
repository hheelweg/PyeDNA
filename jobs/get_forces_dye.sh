#!/bin/bash

#SBATCH --nodes=3
#SBATCH --partition=normal
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=forces
#SBATCH --output=slurm-%j.log

# ============================================================================
# Evaluate forces for an Amber trajectory and construct a dye/DNA force file.
#
# Usage:
#   sbatch get_forces_dye.sh <name> [nframes]
#
# Examples:
#   sbatch get_forces_dye.sh dna_0nt
#   sbatch get_forces_dye.sh dna_0nt 100
#
# Inputs:
#   <name>.prmtop
#   <name>.nc
#
# Retained outputs:
#   <name>_full_forces.nc
#       Full forces from the unmodified topology.
#
#   <name>_forces_dye_dna.nc
#       Full forces on CY3/CY5 atoms, dye-induced bonded and nonbonded forces
#       on DNA atoms, and zero forces on all remaining atoms.
#
# If nframes is omitted, all trajectory frames are processed. Otherwise, the
# first nframes are used. Temporary modified topologies and decomposition force
# trajectories are removed after successful completion.
# ============================================================================

set -eo pipefail


# ----------------------------------------------------------------------------
# Read command-line argument
# ----------------------------------------------------------------------------

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch $0 <name> [nframes]"
    exit 1
fi

NAME="$1"
NFRAMES_REQUESTED="${2:-}"

if [[ -n "$NFRAMES_REQUESTED" ]] && \
   [[ ! "$NFRAMES_REQUESTED" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: nframes must be a positive integer."
    exit 1
fi


# ----------------------------------------------------------------------------
# Input and output names
# ----------------------------------------------------------------------------

PRMTOP="${NAME}.prmtop"
TRAJECTORY="${NAME}.nc"

NONBOND_PRMTOP="${NAME}_nonbond.prmtop"
BOND_PRMTOP="${NAME}_bond.prmtop"

# Retained full-force trajectory
FULL_FORCES="${NAME}_full_forces.nc"
# Temporary decomposition trajectories
NONBOND_FORCES="${NAME}_nonbond_aux.nc"
BOND_FORCES="${NAME}_bond_aux.nc"

# Final dye/DNA force trajectory
FINAL_FORCES="${NAME}_forces_dye_dna.nc"


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

for PROGRAM in python cpptraj sander realpath; do
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

if [[ -z "$NFRAMES_REQUESTED" ]]; then
    NFRAMES="$TOTAL_FRAMES"
elif (( NFRAMES_REQUESTED > TOTAL_FRAMES )); then
    echo "Error: requested $NFRAMES_REQUESTED frames, but trajectory contains $TOTAL_FRAMES."
    exit 1
else
    NFRAMES="$NFRAMES_REQUESTED"
fi

echo "Processing $NFRAMES of $TOTAL_FRAMES trajectory frames."


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
# Construct the single final force trajectory
#
# For atom i:
#
#   i in CY3/CY5:
#
#       F_final(i) = F_full(i)
#
#   i in DNA:
#
#       F_final(i)
#         = F_dye-DNA,nonbonded(i) + F_dye-DNA,bonded(i)
#
#         = [F_full(i) - F_nonbond_off(i)]
#           + [F_full(i) - F_bond_off(i)]
#
#         = 2 F_full(i)
#           - F_nonbond_off(i)
#           - F_bond_off(i)
#
#   i in solvent, ions, or other residues:
#
#       F_final(i) = 0
#
# The output retains the original atom count and atom ordering, so it remains
# directly compatible with the original unmodified topology.
# ----------------------------------------------------------------------------

FINAL_FORCES_ABS="$SUBMIT_DIR/$FINAL_FORCES"

rm -f "$FINAL_FORCES_ABS"

python - \
    "$PRMTOP_ABS" \
    "$FULL_FORCES_ABS" \
    "$NONBOND_FORCES_ABS" \
    "$BOND_FORCES_ABS" \
    "$FINAL_FORCES_ABS" <<'PY'
import sys
from pathlib import Path

import netCDF4
import numpy as np
import parmed as pmd


(
    topology_filename,
    full_filename,
    nonbond_filename,
    bond_filename,
    output_filename,
) = map(Path, sys.argv[1:])


DYE_RESNAMES = {
    "CY3",
    "CY5",
}

DNA_RESNAMES = {
    "DA", "DA3", "DA5",
    "DC", "DC3", "DC5",
    "DG", "DG3", "DG5",
    "DT", "DT3", "DT5",
    "DU", "DU3", "DU5",
}


def read_force_file(filename):
    with netCDF4.Dataset(filename, "r") as dataset:
        if "forces" not in dataset.variables:
            raise RuntimeError(
                f"{filename} does not contain a 'forces' variable."
            )

        forces = np.asarray(
            dataset.variables["forces"][:],
            dtype=np.float64,
        )

        time = None
        if "time" in dataset.variables:
            time = np.asarray(
                dataset.variables["time"][:],
                dtype=np.float64,
            )

        spatial = None
        if "spatial" in dataset.variables:
            spatial = np.asarray(
                dataset.variables["spatial"][:]
            )

    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected force shape in {filename}: {forces.shape}"
        )

    return forces, time, spatial


topology = pmd.load_file(str(topology_filename))

full_forces, full_time, spatial = read_force_file(full_filename)
nonbond_forces, nonbond_time, _ = read_force_file(nonbond_filename)
bond_forces, bond_time, _ = read_force_file(bond_filename)


# ----------------------------------------------------------------------
# Validate matching trajectories
# ----------------------------------------------------------------------

if full_forces.shape != nonbond_forces.shape:
    raise RuntimeError(
        "Full and nonbonded-off force arrays differ in shape: "
        f"{full_forces.shape} versus {nonbond_forces.shape}"
    )

if full_forces.shape != bond_forces.shape:
    raise RuntimeError(
        "Full and bonded-off force arrays differ in shape: "
        f"{full_forces.shape} versus {bond_forces.shape}"
    )

if len(topology.atoms) != full_forces.shape[1]:
    raise RuntimeError(
        "Topology and force trajectory atom counts differ: "
        f"{len(topology.atoms)} versus {full_forces.shape[1]}"
    )

if full_time is not None and nonbond_time is not None:
    if not np.allclose(full_time, nonbond_time):
        raise RuntimeError(
            "Full and nonbonded-off trajectories have different times."
        )

if full_time is not None and bond_time is not None:
    if not np.allclose(full_time, bond_time):
        raise RuntimeError(
            "Full and bonded-off trajectories have different times."
        )


# ----------------------------------------------------------------------
# Identify dye and DNA atoms using the original topology
# ----------------------------------------------------------------------

dye_mask = np.asarray(
    [
        atom.residue.name.upper() in DYE_RESNAMES
        for atom in topology.atoms
    ],
    dtype=bool,
)

dna_mask = np.asarray(
    [
        atom.residue.name.upper() in DNA_RESNAMES
        for atom in topology.atoms
    ],
    dtype=bool,
)

if not np.any(dye_mask):
    raise RuntimeError(
        "No CY3 or CY5 atoms were found in the topology."
    )

if not np.any(dna_mask):
    residue_names = sorted(
        {residue.name for residue in topology.residues}
    )

    raise RuntimeError(
        "No recognized DNA residues were found. "
        f"Residues present: {residue_names}"
    )


# ----------------------------------------------------------------------
# Construct final force array
#
# Begin with zero forces for every atom.
#
# Dye atoms:
#
#     F_final = F_full
#
# DNA atoms:
#
#     F_final
#       = (F_full - F_nonbond_off)
#         + (F_full - F_bond_off)
#
#       = 2*F_full - F_nonbond_off - F_bond_off
#
# All remaining atoms stay zero.
# ----------------------------------------------------------------------

final_forces = np.zeros_like(
    full_forces,
    dtype=np.float64,
)

final_forces[:, dye_mask, :] = full_forces[:, dye_mask, :]

final_forces[:, dna_mask, :] = (
    2.0 * full_forces[:, dna_mask, :]
    - nonbond_forces[:, dna_mask, :]
    - bond_forces[:, dna_mask, :]
)


# ----------------------------------------------------------------------
# Write an all-atom NetCDF compatible with the original topology
# ----------------------------------------------------------------------

nframes, natoms, nspatial = final_forces.shape

with netCDF4.Dataset(
    output_filename,
    "w",
    format="NETCDF4_CLASSIC",
) as output:

    output.createDimension("frame", None)
    output.createDimension("atom", natoms)
    output.createDimension("spatial", nspatial)

    spatial_variable = output.createVariable(
        "spatial",
        "S1",
        ("spatial",),
    )

    if spatial is not None:
        spatial_variable[:] = spatial
    else:
        spatial_variable[:] = np.asarray(
            [b"x", b"y", b"z"],
            dtype="S1",
        )

    force_variable = output.createVariable(
        "forces",
        "f8",
        ("frame", "atom", "spatial"),
    )
    force_variable.units = "kilocalorie/mole/angstrom"
    force_variable[:] = final_forces

    if full_time is not None:
        time_variable = output.createVariable(
            "time",
            "f8",
            ("frame",),
        )
        time_variable.units = "picosecond"
        time_variable[:] = full_time

    output.title = (
        "Full dye forces and dye-induced forces on DNA"
    )

    output.application = "AMBER force decomposition"
    output.program = "force-generation SLURM script"
    output.Conventions = "AMBER"
    output.ConventionVersion = "1.0"

    output.comment = (
        "This trajectory has the same atom count and atom ordering as the "
        "original topology. For CY3/CY5 atoms, F_final = F_full from the "
        "unmodified topology. For DNA atoms, only forces imposed by the dyes "
        "are retained: F_final = "
        "(F_full - F_nonbond_off) + "
        "(F_full - F_bond_off) = "
        "2*F_full - F_nonbond_off - F_bond_off. "
        "For solvent, ions, and all other atoms, F_final = 0."
    )

    output.dye_force_definition = (
        "F_final(i) = F_full(i) for i in CY3 or CY5."
    )

    output.dna_force_definition = (
        "F_final(j) = 2*F_full(j) - F_nonbond_off(j) "
        "- F_bond_off(j) for j in DNA."
    )

    output.other_force_definition = (
        "F_final(k) = 0 for atoms outside CY3, CY5, and DNA."
    )


PY


if [[ ! -s "$FINAL_FORCES_ABS" ]]; then
    echo "Error: final force trajectory was not created:"
    echo "  $FINAL_FORCES_ABS"
    exit 1
fi


# ----------------------------------------------------------------------------
# Remove auxiliary modified topologies and decomposition force files.
# Retain both the full-force and dye/DNA force trajectories.
# ----------------------------------------------------------------------------

rm -f \
    "$NONBOND_PRMTOP_ABS" \
    "$BOND_PRMTOP_ABS" \
    "$NONBOND_FORCES_ABS" \
    "$BOND_FORCES_ABS"

echo "Completed successfully."
echo "  Full forces:    $FULL_FORCES"
echo "  Dye/DNA forces: $FINAL_FORCES"