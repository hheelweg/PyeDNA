#!/bin/bash

#SBATCH --nodes=3
#SBATCH --partition=normal
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=forces
#SBATCH --output=slurm-%j.log

# USAGE:
#   sbatch this_script.sh <name>
#
# EXAMPLE:
#   sbatch this_script.sh dna_0nt

# Do not use `set -u` before sourcing config.sh because Conda's shell
# scripts may temporarily reference variables that have not been defined.
set -eo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <name>"
    echo "Example: sbatch $0 dna_0nt"
    exit 1
fi

NAME="$1"
NFRAMES=10

PRMTOP="${NAME}.prmtop"
TRAJECTORY="${NAME}.nc"

NONBOND_PRMTOP="${NAME}_nonbond.prmtop"
BOND_PRMTOP="${NAME}_bond.prmtop"

FULL_FORCES="${NAME}_forces.nc"
NONBOND_FORCES="${NAME}_nonbond_forces.nc"
BOND_FORCES="${NAME}_bond_forces.nc"

if [[ -z "${PYEDNA_HOME:-}" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

CONFIG_FILE="$PYEDNA_HOME/config.sh"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Source the environment before enabling nounset.
source "$CONFIG_FILE"

# Safe to enable stricter unset-variable handling for our own script now.
set -u


# ----------------------------------------------------------------------
# Generate modified topologies
# ----------------------------------------------------------------------

echo "Generating modified topologies..."

python -m modify_prmtop \
    "$PRMTOP" \
    "$NONBOND_PRMTOP" \
    "$BOND_PRMTOP"


# ----------------------------------------------------------------------
# Temporary working directory
#
# All intermediate files go here and are removed when the job finishes.
# ----------------------------------------------------------------------

WORKDIR="${SLURM_TMPDIR:-/tmp}/${USER}_forces_${SLURM_JOB_ID}"

mkdir -p "$WORKDIR"

cleanup() {
    rm -rf "$WORKDIR"
}

trap cleanup EXIT


SHORT_TRAJECTORY="$WORKDIR/${NAME}_first_${NFRAMES}.nc"
FIRST_RESTART="$WORKDIR/${NAME}_first_frame.rst7"
FORCE_INPUT="$WORKDIR/force.in"


# ----------------------------------------------------------------------
# Extract only the first NFRAMES frames.
#
# The restart supplies sander with coordinates and box information at
# startup. The actual evaluated coordinates are read from -y.
# ----------------------------------------------------------------------

echo "Extracting the first $NFRAMES frames..."

cpptraj "$PRMTOP" > "$WORKDIR/cpptraj.log" <<EOF
trajin $TRAJECTORY 1 $NFRAMES 1
trajout $SHORT_TRAJECTORY netcdf
trajout $FIRST_RESTART restart onlyframes 1
run
quit
EOF


# ----------------------------------------------------------------------
# sander trajectory force-evaluation input
#
# imin=5:
#   Read coordinates from an existing trajectory and evaluate energies
#   and forces without propagating molecular dynamics.
#
# ntb=1:
#   Use periodic boundary conditions with a fixed simulation box.
#
# ntwf=1 and -frc:
#   Write forces for every input trajectory frame.
#
# ioutfm=1:
#   Write binary NetCDF output rather than formatted text.
# ----------------------------------------------------------------------

cat > "$FORCE_INPUT" <<EOF
Trajectory force evaluation
&cntrl
    imin   = 5,
    ntx    = 1,
    irest  = 0,

    ntb    = 1,
    cut    = 10.0,

    ntc    = 1,
    ntf    = 1,

    ntpr   = 1,
    ntwx   = 0,
    ntwr   = 0,
    ntwf   = 1,

    ioutfm = 1,
/
EOF


# ----------------------------------------------------------------------
# Remove old force outputs so sander cannot append to stale files.
# ----------------------------------------------------------------------

rm -f \
    "$FULL_FORCES" \
    "$NONBOND_FORCES" \
    "$BOND_FORCES"


# ----------------------------------------------------------------------
# Run the three independent force evaluations concurrently.
#
# Each srun receives:
#   - one complete node
#   - one task
#   - one CPU
#
# Standard sander is essentially serial for this calculation, so the
# useful parallelism is across the three independent topologies, not
# by assigning many CPU cores to one sander process.
# ----------------------------------------------------------------------

echo "Starting three parallel force evaluations..."

# Remove any temporary sander output left from an earlier attempt.
rm -f \
    "$WORKDIR/full.out" \
    "$WORKDIR/nonbond.out" \
    "$WORKDIR/bond.out" \
    "$WORKDIR/full.stdout" \
    "$WORKDIR/nonbond.stdout" \
    "$WORKDIR/bond.stdout"


srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    -- \
    sander -O \
        -i "$FORCE_INPUT" \
        -o "$WORKDIR/full.out" \
        -p "$PRMTOP" \
        -c "$FIRST_RESTART" \
        -y "$SHORT_TRAJECTORY" \
        -frc "$FULL_FORCES" \
        > "$WORKDIR/full.stdout" 2>&1 &

PID_FULL=$!


srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    -- \
    sander -O \
        -i "$FORCE_INPUT" \
        -o "$WORKDIR/nonbond.out" \
        -p "$NONBOND_PRMTOP" \
        -c "$FIRST_RESTART" \
        -y "$SHORT_TRAJECTORY" \
        -frc "$NONBOND_FORCES" \
        > "$WORKDIR/nonbond.stdout" 2>&1 &

PID_NONBOND=$!


srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    -- \
    sander -O \
        -i "$FORCE_INPUT" \
        -o "$WORKDIR/bond.out" \
        -p "$BOND_PRMTOP" \
        -c "$FIRST_RESTART" \
        -y "$SHORT_TRAJECTORY" \
        -frc "$BOND_FORCES" \
        > "$WORKDIR/bond.stdout" 2>&1 &

PID_BOND=$!


# ----------------------------------------------------------------------
# Wait for all three sander calculations.
#
# Explicit status handling makes sure one failed calculation causes the
# whole SLURM job to fail.
# ----------------------------------------------------------------------

STATUS=0

wait "$PID_FULL" || {
    echo "Error: full-system force evaluation failed."
    cat "$WORKDIR/full.stdout"
    STATUS=1
}

wait "$PID_NONBOND" || {
    echo "Error: nonbonded-off force evaluation failed."
    cat "$WORKDIR/nonbond.stdout"
    STATUS=1
}

wait "$PID_BOND" || {
    echo "Error: bonded-off force evaluation failed."
    cat "$WORKDIR/bond.stdout"
    STATUS=1
}

if [[ "$STATUS" -ne 0 ]]; then
    exit "$STATUS"
fi


# ----------------------------------------------------------------------
# Verify output files
# ----------------------------------------------------------------------

for FORCE_FILE in \
    "$FULL_FORCES" \
    "$NONBOND_FORCES" \
    "$BOND_FORCES"
do
    if [[ ! -s "$FORCE_FILE" ]]; then
        echo "Error: force file '$FORCE_FILE' was not created."
        exit 1
    fi
done


echo
echo "Force evaluation completed for the first $NFRAMES frames:"
echo "  Full system:       $FULL_FORCES"
echo "  Nonbonded off:     $NONBOND_FORCES"
echo "  Cross-bonded off:  $BOND_FORCES"