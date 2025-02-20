#!/bin/bash

# check if a .nab file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename.nab>"
    exit 1
fi

NAB_FILE="$1"

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

# define the path to the AmberClassic installation directory
AMBERCLASSIC_DIR="/home/hheelweg/opt/AmberClassic"

# source the AmberClassic environment setup script without changing directories
if [ -f "$AMBERCLASSIC_DIR/AmberClassic.sh" ]; then
    source "$AMBERCLASSIC_DIR/AmberClassic.sh"
else
    echo "Error: AmberClassic.sh not found in $AMBERCLASSIC_DIR."
    exit 1
fi

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