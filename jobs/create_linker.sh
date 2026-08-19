#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=create_linker
#SBATCH --output=create_linker.log

# USAGE
# -----
# sbatch "$PYEDNA_HOME/jobs/create_linker.sh" [LINKER_CONFIG]
#
# LINKER_CONFIG defaults to linker.toml in the current directory.

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch $0 [LINKER_CONFIG]"
    exit 1
fi

LINKER_CONFIG="${1:-linker.toml}"

source "$PYEDNA_HOME/config.sh"

python "$PYEDNA_HOME/scripts/create_linker.py" --config "$LINKER_CONFIG"