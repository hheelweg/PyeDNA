import numpy as np
from joblib import dump, load
import pyedna

def main():

    # (1) load structure parameters and define instance of MDSimulation class
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    MDsim = pyedna.MDSimulation(dna_params, 'md.params')

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

    # (7) perform ORCA quantum-mechanical analysis
    trajectory.testORCA()
    
    

if __name__ == "__main__":

    main()