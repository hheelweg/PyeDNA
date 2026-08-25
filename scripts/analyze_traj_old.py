import torch
import numpy as np
import argparse
from joblib import dump, load
import pyedna

# # detect available GPUs 
# num_gpus = torch.cuda.device_count()
# if num_gpus < 2:
#     raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM \
#                        allocation and adjust accordingly.")


def main(config_file="md.toml"):
    """
    Execute the trajectory analysis workflow.

    This function performs the following steps:
    1. Loads the molecular dynamics (MD) configuration for timing metadata.
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
    - The trajectory time step (`dt`) is read from the MD TOML file.

    Raises:
    - FileNotFoundError: If any of the required files are not found in the current directory.
    - ValueError: If multiple files with the expected extension are found, indicating ambiguity.

    """

    # (1) load MD metadata used for trajectory timing
    MDsim = pyedna.MDConfig.from_file(config_file)

    # (2) trajectory raw data from AMBER MD
    # searches for files with specific ending in cwd (needs to be unique)
    name_prmtop = pyedna.utils.findFileWithExtension('.prmtop')
    name_nc = pyedna.utils.findFileWithExtension('.nc')
    name_out = pyedna.utils.findFileWithExtension('.out')
    traj_data = [name_prmtop, name_nc, name_out]

    # (3) parameter file for trajectory analysis
    traj_params = pyedna.utils.findFileWithName('traj.params')

    # (4) parameter file for molecules (dyes)
    mols_params = pyedna.utils.findFileWithName('mols.params')


    # (5) define Trajectory object
    trajectory = pyedna.Trajectory(
                                    MDsim, traj_data,
                                    traj_params_file = traj_params
                                  )


    # (6) initialize (dye) molecules of interest
    trajectory.initMolecules(mols_params)

    # (7) loop through trajectory snapshots and analyze based on traj.params
    trajectory.loopTrajectory()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an MD trajectory")
    parser.add_argument("--config", default="md.toml", help="MD TOML file")
    args = parser.parse_args()
    main(config_file=args.config)
