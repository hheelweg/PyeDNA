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

FULL_FORCES="${NAME}_full_forces.nc"
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
# Construct the final force trajectories
#
# We produce:
#
#   1. Complete forces on CY3 and CY5 atoms:
#
#        F_dye,full = F_full restricted to CY3/CY5 atoms
#
#   2. Dye-DNA forces acting on DNA atoms:
#
#        F_DNA<-dye
#          = F_dye-DNA,nonbonded + F_dye-DNA,bonded
#
#        F_dye-DNA,nonbonded
#          = F_full - F_nonbond_off
#
#        F_dye-DNA,bonded
#          = F_full - F_bond_off
#
#        Therefore:
#
#        F_DNA<-dye
#          = (F_full - F_nonbond_off)
#            + (F_full - F_bond_off)
#
#          = 2 F_full - F_nonbond_off - F_bond_off
#
# The final expression is restricted to DNA atoms. On other atom types,
# F_full - F_nonbond_off may also contain removed dye-water, dye-ion, or
# dye-dye interactions.
# ----------------------------------------------------------------------------

DYE_FULL_FORCES="${NAME}_dye_full_forces.nc"
DNA_DYE_FORCES="${NAME}_dna_dye_forces.nc"

rm -f \
    "$DYE_FULL_FORCES" \
    "$DNA_DYE_FORCES"

echo
echo "Constructing complete dye forces and dye-DNA forces..."

python - \
    "$PRMTOP_ABS" \
    "$FULL_FORCES_ABS" \
    "$NONBOND_FORCES_ABS" \
    "$BOND_FORCES_ABS" \
    "$SUBMIT_DIR/$DYE_FULL_FORCES" \
    "$SUBMIT_DIR/$DNA_DYE_FORCES" <<'PY'
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
    dye_output_filename,
    dna_output_filename,
) = map(Path, sys.argv[1:])


DYE_RESIDUE_NAMES = {
    "CY3",
    "CY5",
}

DNA_RESIDUE_NAMES = {
    "DA", "DA3", "DA5",
    "DC", "DC3", "DC5",
    "DG", "DG3", "DG5",
    "DT", "DT3", "DT5",
    "DU", "DU3", "DU5",
}


def read_forces(filename):
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

    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected force-array shape in {filename}: "
            f"{forces.shape}"
        )

    return forces, time


def write_subset(
    filename,
    forces,
    time,
    topology,
    atom_indices,
    title,
    comment,
):
    nframes, natoms, nspatial = forces.shape

    atom_names = [
        topology.atoms[index].name
        for index in atom_indices
    ]

    residue_names = [
        topology.atoms[index].residue.name
        for index in atom_indices
    ]

    original_atom_indices = atom_indices + 1

    original_residue_indices = np.asarray(
        [
            topology.atoms[index].residue.idx + 1
            for index in atom_indices
        ],
        dtype=np.int32,
    )

    max_atom_name_length = max(
        4,
        max(len(name) for name in atom_names),
    )

    max_residue_name_length = max(
        4,
        max(len(name) for name in residue_names),
    )

    with netCDF4.Dataset(
        filename,
        "w",
        format="NETCDF4_CLASSIC",
    ) as output:

        output.createDimension("frame", None)
        output.createDimension("atom", natoms)
        output.createDimension("spatial", nspatial)

        output.createDimension(
            "atom_name_length",
            max_atom_name_length,
        )

        output.createDimension(
            "residue_name_length",
            max_residue_name_length,
        )

        spatial_variable = output.createVariable(
            "spatial",
            "S1",
            ("spatial",),
        )
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
        force_variable[:] = forces

        if time is not None:
            time_variable = output.createVariable(
                "time",
                "f8",
                ("frame",),
            )
            time_variable.units = "picosecond"
            time_variable[:] = time

        atom_index_variable = output.createVariable(
            "original_atom_index",
            "i4",
            ("atom",),
        )
        atom_index_variable.indexing = "one-based"
        atom_index_variable[:] = original_atom_indices

        residue_index_variable = output.createVariable(
            "original_residue_index",
            "i4",
            ("atom",),
        )
        residue_index_variable.indexing = "one-based"
        residue_index_variable[:] = original_residue_indices

        atom_name_variable = output.createVariable(
            "atom_name",
            "S1",
            ("atom", "atom_name_length"),
        )
        atom_name_variable[:] = netCDF4.stringtochar(
            np.asarray(
                atom_names,
                dtype=f"S{max_atom_name_length}",
            )
        )

        residue_name_variable = output.createVariable(
            "residue_name",
            "S1",
            ("atom", "residue_name_length"),
        )
        residue_name_variable[:] = netCDF4.stringtochar(
            np.asarray(
                residue_names,
                dtype=f"S{max_residue_name_length}",
            )
        )

        output.title = title
        output.application = "AMBER force decomposition"
        output.program = "force-generation SLURM script"
        output.Conventions = "AMBER"
        output.ConventionVersion = "1.0"
        output.comment = comment


topology = pmd.load_file(str(topology_filename))

full_forces, full_time = read_forces(full_filename)
nonbond_forces, nonbond_time = read_forces(nonbond_filename)
bond_forces, bond_time = read_forces(bond_filename)


if full_forces.shape != nonbond_forces.shape:
    raise RuntimeError(
        "Full and nonbonded-off force trajectories have different shapes: "
        f"{full_forces.shape} versus {nonbond_forces.shape}"
    )

if full_forces.shape != bond_forces.shape:
    raise RuntimeError(
        "Full and bonded-off force trajectories have different shapes: "
        f"{full_forces.shape} versus {bond_forces.shape}"
    )

if len(topology.atoms) != full_forces.shape[1]:
    raise RuntimeError(
        "Topology and force trajectory atom counts do not agree: "
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


dye_atom_indices = np.asarray(
    [
        atom.idx
        for atom in topology.atoms
        if atom.residue.name.upper() in DYE_RESIDUE_NAMES
    ],
    dtype=np.int64,
)

dna_atom_indices = np.asarray(
    [
        atom.idx
        for atom in topology.atoms
        if atom.residue.name.upper() in DNA_RESIDUE_NAMES
    ],
    dtype=np.int64,
)


if dye_atom_indices.size == 0:
    raise RuntimeError(
        "No atoms belonging to CY3 or CY5 were found."
    )

if dna_atom_indices.size == 0:
    residue_names = sorted(
        {residue.name for residue in topology.residues}
    )

    raise RuntimeError(
        "No recognized DNA residues were found. "
        f"Residue names present: {residue_names}"
    )


# Complete physical forces on all CY3/CY5 atoms.
dye_full_forces = full_forces[:, dye_atom_indices, :]


# Dye-DNA forces acting on DNA atoms:
#
# F_DNA<-dye
#   = (F_full - F_nonbond_off)
#     + (F_full - F_bond_off)
#
#   = 2 F_full - F_nonbond_off - F_bond_off
dna_dye_forces_all_atoms = (
    2.0 * full_forces
    - nonbond_forces
    - bond_forces
)

dna_dye_forces = dna_dye_forces_all_atoms[
    :,
    dna_atom_indices,
    :,
]


dye_comment = (
    "Complete forces acting on CY3 and CY5 atoms. "
    "The forces are taken directly from the unmodified force trajectory: "
    "F_dye_full = F_full restricted to CY3/CY5 atoms. "
    "These forces contain all force-field contributions acting on the dyes."
)

dna_comment = (
    "Dye-DNA interaction forces acting on DNA atoms. "
    "The nonbonded dye-DNA contribution is "
    "F_dye-DNA_nonbonded = F_full - F_nonbond_off. "
    "The cross-boundary bonded contribution is "
    "F_dye-DNA_bonded = F_full - F_bond_off. "
    "Therefore F_DNA<-dye = "
    "(F_full - F_nonbond_off) + (F_full - F_bond_off) = "
    "2*F_full - F_nonbond_off - F_bond_off. "
    "This expression is evaluated only for DNA atoms because the "
    "nonbonded-off topology can also remove dye-water, dye-ion, and "
    "dye-dye interactions on other atoms."
)


write_subset(
    filename=dye_output_filename,
    forces=dye_full_forces,
    time=full_time,
    topology=topology,
    atom_indices=dye_atom_indices,
    title="Complete forces on CY3 and CY5 atoms",
    comment=dye_comment,
)

write_subset(
    filename=dna_output_filename,
    forces=dna_dye_forces,
    time=full_time,
    topology=topology,
    atom_indices=dna_atom_indices,
    title="Dye-DNA forces acting on DNA atoms",
    comment=dna_comment,
)


print(
    f"Wrote {dye_output_filename} "
    f"with {dye_atom_indices.size} dye atoms."
)

print(
    f"Wrote {dna_output_filename} "
    f"with {dna_atom_indices.size} DNA atoms."
)
PY


# Verify the two derived trajectories.
for FORCE_FILE in \
    "$DYE_FULL_FORCES" \
    "$DNA_DYE_FORCES"
do
    if [[ ! -s "$FORCE_FILE" ]]; then
        echo "Error: derived force file '$FORCE_FILE' was not created."
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
echo "Force evaluation and decomposition completed for the first $NFRAMES frames:"
echo "  Full system:              $FULL_FORCES"
echo "  Nonbonded off:            $NONBOND_FORCES"
echo "  Bonded off:               $BOND_FORCES"
echo "  Complete CY3/CY5 forces:  $DYE_FULL_FORCES"
echo "  Dye-DNA forces on DNA:    $DNA_DYE_FORCES"