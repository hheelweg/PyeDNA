#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=create_linker
#SBATCH --output=create_linker.log

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch $0 [LINKER_CONFIG]"
    exit 1
fi

LINKER_CONFIG="${1:-linker.toml}"

pyedna components create-linker "$LINKER_CONFIG"