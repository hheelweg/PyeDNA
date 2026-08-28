#!/bin/bash

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [DYELNK_CONFIG]"
    exit 1
fi

DYELNK_CONFIG="${1:-dyelnk.toml}"

pyedna components create-dyelnk "$DYELNK_CONFIG"