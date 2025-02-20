import os
import argparse
import time
import subprocess
import torch
import json
import numpy as np
from pyscf import gto, lib
import io
import sys
from joblib import dump, load

import pyedna

# Detect available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM allocation.")



def main():

    # TODO : write class for MD simulation
    params = []
    MDsim = pyedna.MDSimulation(params)

    # input path
    path = './'

    # trajectory raw data
    name_prmtop = 'dna_test.prmtop'
    name_nc = 'dna_test_prod.nc'                        # need to NetCDF3 and not NetCDF4 (use cpptraj to convert)
    name_out = 'dna_test_prod.out'

    # parameter file for trajectory analysis
    out_params = 'out.params'


    data = [name_prmtop, name_nc, name_out]
    dt = 10                                             # specify time step (ps)

    # define Trajectory object
    test = pyedna.Trajectory(
                            MDsim, path, data, dt,
                            output_params = out_params
                            )

    # define donor and acceptor molecules
    # TODO : put this into *.params file
    donor = [9]
    acceptor = [14]
    molecules = [donor, acceptor]


    # time slices we are interested in
    time_slice = [0, 0]
    test.initMolecules(molecules)
    test.loopTrajectory(time_slice)
    
    



if __name__ == "__main__":

    # run main
    main()

