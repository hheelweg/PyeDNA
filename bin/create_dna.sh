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


# define the path to the AmberClassic installation directory
AMBERCLASSIC_DIR="/home/hheelweg/opt/AmberClassic"


# change to the AmberClassic directory
cd "$AMBERCLASSIC_DIR" || {
    echo "Error: Directory $AMBERCLASSIC_DIR does not exist. Check installation."
    exit 1
}

# source the AmberClassic environment setup script, which sets paths
if [ -f "AmberClassic.sh" ]; then
    source AmberClassic.sh
else
    echo "Error: AmberClassic.sh not found in $AMBERCLASSIC_DIR."
    exit 1
fi

# check if the NAB source file exists
if [ ! -f "$NAB_FILE" ]; then
    echo "Error: $NAB_FILE not found in $AMBERCLASSIC_DIR."
    exit 1
fi

# compile the NAB source file
nab "$NAB_FILE"
if [ ! -f "a.out" ]; then
    echo "Error: Compilation failed. a.out not generated."
    exit 1
fi

# Run the compiled program
./a.out
