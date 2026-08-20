#!/bin/bash

# USAGE
# -----
# bash "$PYEDNA_HOME/jobs/create_dyelnk.sh" [DYELNK_CONFIG]
#
# DYELNK_CONFIG defaults to dyelnk.toml in the current directory.

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [DYELNK_CONFIG]"
    exit 1
fi

DYELNK_CONFIG="${1:-dyelnk.toml}"

if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

source "$PYEDNA_HOME/config.sh"

python "$PYEDNA_HOME/scripts/create_dyelnk.py" --config "$DYELNK_CONFIG"