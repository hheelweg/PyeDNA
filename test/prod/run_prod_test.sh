#!/bin/sh
#SBATCH --nodes=1
#SBATCH --partition=gpu   		# change to the correct GPU partition name	
#SBATCH --nodelist=gpu001		# run on GPU node	
#SBATCH --ntasks=1			# only 1 task (as `pmemd.cuda` handles parallelism internally)
#SBATCH --gres=gpu:2           		# request # of GPUs (adjust based on cluster)
##SBATCH --cpus-per-task=8		# use 4-8 CPUs per GPU
#SBATCH --job-name=dna_test		# job name
#SBATCH --output=dna_test.log		# name of output log file

# path to AMBER MD executables
export AMBERHOME=/home/hheelweg/.conda/envs/AmberTools24/amber24
export PATH=$AMBERHOME/bin:$PATH
export LD_LIBRARY_PATH=$AMBERHOME/lib:$LD_LIBRARY_PATH

# Job Submission
srun pmemd.cuda -O -i dna_test_prod.in -o dna_test_prod.out -p dna_test.prmtop -c ../eq/dna_test_eq2.ncrst -r dna_test_prod.ncrst -x dna_test_prod.nc -ref ../min/dna_test_min2.ncrst
wait
