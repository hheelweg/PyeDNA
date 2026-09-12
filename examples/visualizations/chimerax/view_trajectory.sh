#!/usr/bin/env bash
set -euo pipefail

style_file="chimera_style.cxc"
chimerax_cmd="${CHIMERAX:-chimerax}"

if (($# != 2)); then
    echo "Usage: $0 structure_xxx name" >&2
    exit 1
fi

structure_dir="$1"
structure_name="$2"

if [[ ! -d "$structure_dir" ]]; then
    echo "Error: missing trajectory directory: $structure_dir" >&2
    exit 1
fi

if [[ ! -f "$style_file" ]]; then
    echo "Error: missing $style_file in the current directory." >&2
    exit 1
fi

pdb_file="$structure_dir/${structure_name}_solvated.pdb"
nc_file="$structure_dir/${structure_name}.nc"

if [[ ! -f "$pdb_file" ]]; then
    echo "Error: missing solvated PDB reference: $pdb_file" >&2
    echo "ChimeraX loads Amber NetCDF trajectories by attaching them to the full solvated PDB model." >&2
    exit 1
fi

if [[ ! -f "$nc_file" ]]; then
    echo "Error: missing Amber NetCDF trajectory: $nc_file" >&2
    exit 1
fi

quote_chimerax_path() {
    local path="$1"
    path=${path//\\/\\\\}
    path=${path//\"/\\\"}
    printf '"%s"' "$path"
}

exec "$chimerax_cmd" \
    --cmd "open $(quote_chimerax_path "$pdb_file")" \
    --cmd "open $(quote_chimerax_path "$nc_file") format amber structureModel #1" \
    --cmd "hide solvent" \
    --cmd "hide ions" \
    --cmd "open $(quote_chimerax_path "$style_file")"
