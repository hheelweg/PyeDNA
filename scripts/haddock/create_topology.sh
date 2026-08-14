#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: bash create_topology.sh NAME CHARGE RESNAME SEGID"
    echo "Example: bash create_topology.sh CY3 0 CY3 Z"
    echo "Example: bash create_topology.sh PDI_dimer -10 PDI Y"
    exit 1
fi

NAME="$1"
CHARGE="$2"
RESNAME="$3"
SEGID="$4"

MOL2="${NAME}.mol2"
MOL2_SINGLE="${NAME}_${RESNAME}.mol2"
ACPYPE_DIR="${NAME}_${RESNAME}.acpype"
TMP_DIR=".acpype_tmp_${NAME}_${RESNAME}"
OUT_DIR="${NAME}"

[[ -f "$MOL2" ]] || { echo "Error: $MOL2 not found."; exit 1; }
[[ "$RESNAME" =~ ^[A-Za-z0-9]{1,3}$ ]] || {
    echo "Error: RESNAME must contain 1–3 letters or numbers."
    exit 1
}
[[ "$SEGID" =~ ^[A-Za-z0-9]$ ]] || {
    echo "Error: SEGID must be one letter or number."
    exit 1
}

awk -v resname="$RESNAME" '
/@<TRIPOS>ATOM/ {atom=1}
/@<TRIPOS>BOND/ {atom=0}
atom && NF>=9 {$7=1; $8=resname}
{print}
' "$MOL2" > "$MOL2_SINGLE"

rm -rf "$ACPYPE_DIR" "$TMP_DIR" "$OUT_DIR"

# Run ACPYPE in the dedicated HADDOCK environment
conda run --no-capture-output -n haddock \
    acpype -i "$MOL2_SINGLE" -o cns -a gaff -c user -n "$CHARGE"

TOP=$(find "$ACPYPE_DIR" -maxdepth 1 -name '*_CNS.top' -print -quit)
PAR=$(find "$ACPYPE_DIR" -maxdepth 1 -name '*_CNS.par' -print -quit)
PDB=$(find "$ACPYPE_DIR" -maxdepth 1 -name '*_NEW.pdb' -print -quit)

[[ -n "$TOP" && -n "$PAR" && -n "$PDB" ]] || {
    echo "Error: ACPYPE completed, but one or more expected files are missing."
    exit 1
}

mkdir -p "$OUT_DIR"
cp "$TOP" "$OUT_DIR/${NAME}_haddock.top"
cp "$PAR" "$OUT_DIR/${NAME}_haddock.par"

awk -v segid="$SEGID" '
/^(ATOM  |HETATM)/ {
    line=$0
    while (length(line)<76) line=line " "
    line=substr(line,1,21) segid substr(line,23,50) sprintf("%4s",segid) substr(line,77)
    print line
    next
}
{print}
' "$PDB" > "$OUT_DIR/${NAME}_haddock.pdb"

rm -rf "$ACPYPE_DIR" "$TMP_DIR"
rm -f "$MOL2_SINGLE"

echo
echo "Finished."
echo "Generated:"
echo "  $OUT_DIR/${NAME}_haddock.top"
echo "  $OUT_DIR/${NAME}_haddock.par"
echo "  $OUT_DIR/${NAME}_haddock.pdb"
echo "Residue name: $RESNAME"
echo "Chain/segid:  $SEGID"
