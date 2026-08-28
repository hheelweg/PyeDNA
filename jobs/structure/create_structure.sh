#!/bin/bash

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [STRUCTURE_CONFIG]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYEDNA_STRUCTURE_CONFIG="${1:-structure.toml}"

if [[ ! -f "$PYEDNA_STRUCTURE_CONFIG" ]]; then
    echo "Error: structure configuration not found: $PYEDNA_STRUCTURE_CONFIG"
    exit 1
fi

pyedna structure prepare "$PYEDNA_STRUCTURE_CONFIG" \
    > create_structure.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "Error: structure preparation failed. See create_structure.log."
    exit 1
fi

mkdir -p haddock

sbatch \
    --export=ALL,PYEDNA_STRUCTURE_CONFIG="$PYEDNA_STRUCTURE_CONFIG" \
    "$SCRIPT_DIR/run_haddock.sh"