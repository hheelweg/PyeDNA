#!/usr/bin/env bash
set -euo pipefail

style_file="chimera_style.cxc"
structures_dir="structures"
chimerax_cmd="${CHIMERAX:-chimerax}"

if [[ ! -f "$style_file" ]]; then
    echo "Error: missing $style_file in the current directory." >&2
    exit 1
fi

if [[ ! -d "$structures_dir" ]]; then
    echo "Error: missing $structures_dir/ directory in the current directory." >&2
    exit 1
fi

pdb_files=()
while IFS= read -r pdb_file; do
    pdb_files+=("$pdb_file")
done < <(find "$structures_dir" -maxdepth 1 -type f -name "*.pdb" -print | sort)

if ((${#pdb_files[@]} == 0)); then
    echo "Error: no PDB files found in $structures_dir/." >&2
    exit 1
fi

exec "$chimerax_cmd" "${pdb_files[@]}" "$style_file"
