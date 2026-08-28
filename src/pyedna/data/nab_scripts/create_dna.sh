#!/bin/bash

# USAGE:
# bash this_script.sh nab_file.nab /path/to/AmberClassic


# check if a .nab file and AmberClassic directory are provided as arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename.nab> <amberclassic_dir>"
    exit 1
fi

NAB_FILE="$1"
AMBERCLASSIC_DIR="$2"

# ensure the provided file has a .nab extension
if [[ "$NAB_FILE" != *.nab ]]; then
    echo "Error: The file must have a .nab extension."
    exit 1
fi

# check if the NAB source file exists in the current working directory
if [ ! -f "$NAB_FILE" ]; then
    echo "Error: $NAB_FILE not found in the current directory $(pwd)."
    exit 1
fi

# source the AmberClassic environment setup script without changing directories
if [ -f "$AMBERCLASSIC_DIR/AmberClassic.sh" ]; then
    source "$AMBERCLASSIC_DIR/AmberClassic.sh"
else
    echo "Error: AmberClassic.sh not found in $AMBERCLASSIC_DIR."
    exit 1
fi

for lib_dir in \
    "${CONDA_PREFIX:-}/lib" \
    "${CONDA_PREFIX:-}/x86_64-conda-linux-gnu/lib" \
    "${AMBERHOME:-}/lib" \
    "$AMBERCLASSIC_DIR/lib"
do
    if [ -d "$lib_dir" ]; then
        export LIBRARY_PATH="$lib_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

# compile the NAB source file
nab "$NAB_FILE"

# give process time to finish
sleep 1

if [ ! -f "a.out" ]; then
    echo "Error: Compilation failed. a.out not generated."
    exit 1
fi

# run the compiled program
./a.out

# clean up generated files
rm -f a.out *.c tleap.out
