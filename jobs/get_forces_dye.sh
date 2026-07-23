#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=forces
#SBATCH --output=slurm-%j.log

Check if PYEDNA_HOME is set

if [[ -z “$PYEDNA_HOME” ]]; then
echo “Error: PYEDNA_HOME is not set. Please set it in shell.”
exit 1
fi

Load config.sh from the root of PyeDNA

CONFIG_FILE=”$PYEDNA_HOME/config.sh”

if [[ -f “$CONFIG_FILE” ]]; then
source “$CONFIG_FILE”
else
echo “Error: Configuration file ($CONFIG_FILE) not found!”
exit 1
fi

python -m modify_prmtop