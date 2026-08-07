#!/bin/bash
#SBATCH --job-name=haddock
#SBATCH --cpus-per-task=32
#SBATCH --output=haddock_slurm.out
#SBATCH --error=haddock_slurm.err

set -e

cd "$SLURM_SUBMIT_DIR"

if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

if [[ ! -f docking_config.cfg ]]; then
    echo "Error: docking_config.cfg not found."
    exit 1
fi

source "$PYEDNA_HOME/config.sh"

# TODO : only skip this while debugging, this definitely cannot be skipped in the actual workflow
SKIP_HADDOCK=false

RUN_DIR="haddock/run"

if [[ "$SKIP_HADDOCK" == false ]]; then
    rm -rf "$RUN_DIR"

    echo "Starting HADDOCK..."
    conda run -n haddock haddock3 docking_config.cfg
else
    if [[ ! -d "$RUN_DIR" ]]; then
        echo "Error: SKIP_HADDOCK=true but $RUN_DIR does not exist."
        exit 1
    fi

    echo "Using existing HADDOCK run in $RUN_DIR."
fi

echo "Selecting and processing HADDOCK structures..."
python -m postprocess_structure