#!/bin/bash

if [[ -z "$PYEDNA_HOME" ]]; then
    echo "Error: PYEDNA_HOME is not set."
    exit 1
fi

source "$PYEDNA_HOME/config.sh"

if [[ ! -f amber.params ]]; then
    echo "Error: amber.params not found in current directory."
    exit 1
fi

python -m prepare_amber > amber_setup.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "Error: AMBER setup failed. See amber_setup.log."
    exit 1
fi