#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dummy
#SBATCH --output=slurm-%j.log

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [MD_CONFIG]"
    exit 1
fi

MD_CONFIG="${1:-md.toml}"

if [[ ! -f "$MD_CONFIG" ]]; then
    echo "Error: MD configuration not found: $MD_CONFIG"
    exit 1
fi

JOB_NAME="$(basename "$MD_CONFIG" .toml)_md"

scontrol update JobID=$SLURM_JOB_ID Name=$JOB_NAME

pyedna md run "$MD_CONFIG"

NEW_OUTPUT="${JOB_NAME}.log"
mv "slurm-${SLURM_JOB_ID}.log" "$NEW_OUTPUT"