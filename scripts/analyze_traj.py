import torch
import numpy as np
from pyscf import gto, lib
from joblib import dump, load

# import PyeDNA
import pyedna

# Detect available GPUs 
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM \
                       allocation and adjust accordingly.")


def main():

    # TODO : write class for MD simulation
    # ideally, read params from file
    params = []
    MDsim = pyedna.MDSimulation(params)

    # trajectory raw data from AMBER MD
    # searches for files with specific ending in cwd (needs to be unique)
    name_prmtop = pyedna.utils.findFileWithExtension('.prmtop')
    name_nc = pyedna.utils.findFileWithExtension('.nc')
    name_out = pyedna.utils.findFileWithExtension('.out')

    # parameter file for trajectory analysis
    # TODO : add check that file exists and maybe call traj.params
    out_params = pyedna.utils.findFileWithName('traj.params')

    traj_data = [name_prmtop, name_nc, name_out]
    # TODO : ideally use some MDSim.dt thing in the future
    dt = 10                                             # specify time step (ps)

    # TODO : specify file name for output files that we get out of this analysis

    # define Trajectory object
    test = pyedna.Trajectory(
                             MDsim, traj_data, dt,
                             output_params_file = out_params
                            )

    # define donor and acceptor molecules
    # TODO : put this into *.params file
    donor = [9]
    acceptor = [14]
    molecules = [donor, acceptor]


    # time slices we are interested in
    # TODO : put this into *.params file
    time_slice = [0, 0]
    test.initMolecules(molecules)
    test.loopTrajectory(time_slice)
    
    



if __name__ == "__main__":

    # run main
    main()

