#!/bin/sh
#SBATCH --nodes=1		
#SBATCH --ntasks=32
#SBATCH --job-name=dna_test
#SBATCH --output=dna_test.log

# path to AMBER
AMBERPATH="/home/hheelweg/.conda/envs/AmberTools23"

# Job Submission
srun $AMBERPATH/bin/sander -O -i dna_test_min1.in -o dna_test_min1.out -p dna_test.prmtop -c dna_test.rst7 -r dna_test_min1.ncrst -ref dna_test.rst7
srun $AMBERPATH/bin/sander -O -i dna_test_min2.in -o dna_test_min2.out -p dna_test.prmtop -c dna_test_min1.ncrst -r dna_test_min2.ncrst -ref dna_test_min1.ncrst
wait
