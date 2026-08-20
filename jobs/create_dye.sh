#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=create_dye
#SBATCH --output=create_dye.log

# USAGE
# -----
# sbatch "$PYEDNA_HOME/jobs/create_dye.sh" [DYE_CONFIG]
#
# DYE_CONFIG defaults to dye.toml in the current directory.

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch $0 [DYE_CONFIG]"
    exit 1
fi

DYE_CONFIG="${1:-dye.toml}"

source "$PYEDNA_HOME/config.sh"

python "$PYEDNA_HOME/scripts/create_dye.py" --config "$DYE_CONFIG"
