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

# import custom modules
path_to_modules = '/home/hheelweg/Cy3Cy5/PyCY'
sys.path.append(path_to_modules)
import quantumTools, structure
import trajectory as traj
import const
import utils
import fileProcessing as fp

# Detect available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM allocation.")



def main():

    # TODO: write class for MD simulation
    params = []
    MDsim = traj.MDSimulation(params)

    # input data
    path = './'
    name_prmtop = 'dna_test.prmtop'
    name_nc = 'dna_test_prod.nc'                        # need to NetCDF3 and not NetCDF4 (use cpptraj to convert)
    name_out = 'dna_test_prod.out'
    data = [name_prmtop,name_nc, name_out]
    dt = 10                                             # specify time step (ps)

    # define Trajectory object
    test = traj.Trajectory(MDsim, path, data, dt)

    # define donor and acceptor molecules
    donor = [9]
    acceptor = [14]
    molecules = [donor, acceptor]

    # which information do we wish to extract from trajectory
    # TODO : load that from a *.params file ?!
    traj_info = {'conversion': 'pyscf',
             'com': True}

    # time slices we are interested in
    time_slice = [0, 0]
    test.initMolecules(molecules)
    test.loopTrajectory(time_slice, **traj_info)
    
    




if __name__ == "__main__":

    # run main
    main()

