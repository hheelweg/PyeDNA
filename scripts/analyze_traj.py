import torch
import numpy as np
from pyscf import gto, lib
from joblib import dump, load
import pyedna

# detect available GPUs 
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM \
                       allocation and adjust accordingly.")


def main():
    """
    Execute the trajectory analysis workflow.

    This function performs the following steps:
    1. Initializes the molecular dynamics (MD) simulation with parameters.
    2. Identifies and loads necessary trajectory data files:
       - Parameter/topology file (`.prmtop`)
       - Trajectory file (`.nc`)
       - Output file (`.out`)
    3. Loads analysis parameters from 'traj.params'.
    4. Loads molecular parameters from 'mols.params'.
    5. Sets the simulation time step (`dt`).
    6. Creates a `Trajectory` object with the loaded data and parameters.
    7. Initializes molecules of interest for analysis.
    8. Iterates over trajectory snapshots to perform analysis as specified in 'traj.params'.

    Note:
    - Ensure that the current working directory contains the required files:
      'traj.params' and 'mols.params'.
    - The `findFileWithExtension` and `findFileWithName` utility functions are
      used to locate files in the current directory.
    - The time step (`dt`) is currently set to a default value of 10 ps; consider
      updating this to reflect the actual simulation parameters.

    Raises:
    - FileNotFoundError: If any of the required files are not found in the current directory.
    - ValueError: If multiple files with the expected extension are found, indicating ambiguity.

    TODO:
    - Implement the `MDSimulation` class to handle MD simulation initialization.
    - Modify the time step (`dt`) to be retrieved from the `MDSimulation` object
      once it's implemented.
    """


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

    # define Trajectory object
    trajectory = pyedna.Trajectory(
                             MDsim, traj_data, dt,
                             traj_params_file = traj_params
                            )


    # initialize (dye) molecules of interest
    trajectory.initMolecules(mols_params)

    # loop through trajectory snapshots and analyze based on traj.params
    trajectory.loopTrajectory()
    
    

if __name__ == "__main__":

    main()

