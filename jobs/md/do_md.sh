#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=do_md
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

echo "MD config: $MD_CONFIG"
echo "SLURM job: $SLURM_JOB_ID"
echo "SLURM tasks: ${SLURM_NTASKS:-unset}"
echo "CPU cores: ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

pyedna md run "$MD_CONFIG"

NEW_OUTPUT="${JOB_NAME}.log"
mv "slurm-${SLURM_JOB_ID}.log" "$NEW_OUTPUT"
