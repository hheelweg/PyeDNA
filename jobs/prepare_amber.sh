#!/bin/bash

# Usage: bash "$PYEDNA_HOME/jobs/prepare_amber.sh" [STRUCTURE_CONFIG]
# STRUCTURE_CONFIG defaults to "structure.toml" and includes Amber settings.

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [STRUCTURE_CONFIG]"
    exit 1
fi

PYEDNA_STRUCTURE_CONFIG="${1:-structure.toml}"

if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

source "$PYEDNA_HOME/config.sh"

if [[ ! -f "$PYEDNA_STRUCTURE_CONFIG" ]]; then
    echo "Error: structure configuration not found: $PYEDNA_STRUCTURE_CONFIG"
    exit 1
fi

python "$PYEDNA_HOME/scripts/prepare_amber.py" \
    --config "$PYEDNA_STRUCTURE_CONFIG" > prepare_amber.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "Error: AMBER setup failed. See prepare_amber.log."
    exit 1
fi
