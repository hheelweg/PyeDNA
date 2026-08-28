#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=create_dye
#SBATCH --output=create_dye.log

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch $0 [DYE_CONFIG]"
    exit 1
fi

DYE_CONFIG="${1:-dye.toml}"

pyedna components create-dye "$DYE_CONFIG"