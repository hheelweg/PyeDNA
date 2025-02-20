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
    traj_data = [name_prmtop, name_nc, name_out]

    # parameter file for trajectory analysis
    traj_params = pyedna.utils.findFileWithName('traj.params')

    # parameter file for molecules (dyes)
    mols_params = pyedna.utils.findFileWithName('mols.params')

    # TODO : ideally use MDSim.dt thing in the future
    # specify time step (ps)
    dt = 10                                             

    # TODO : specify file name for output files that we get out of this analysis

    # define Trajectory object
    test = pyedna.Trajectory(
                             MDsim, traj_data, dt,
                             traj_params_file = traj_params
                            )


    # initialize (dye) molecules of interest
    test.initMolecules(mols_params)

    # loop through trajectory snapshots and analyze based on traj.params
    test.loopTrajectory()
    
    

if __name__ == "__main__":

    main()

