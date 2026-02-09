import pyedna 
import argparse
import torch

# detect available GPUs 
num_gpus = torch.cuda.device_count()
if num_gpus < 1:
    raise RuntimeError("Error: Less than 1 GPU(s) detected! Check SLURM \
                       allocation and adjust accordingly.")


def main(args):
    
    # (1) parse structure parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    composite_params = pyedna.CompositeStructure.parseCompositeStructure('struc.params')
    

    # (2) locate topology and forcefield files
    prmtop_file = pyedna.utils.findFileWithExtension('.prmtop')
    rst7_file = pyedna.utils.findFileWithExtension('.rst7')

    
    # (3) define MDSimulation object and initialize simulation by feeding topology and forcefield files
    md = pyedna.MDSimulation(dna_params, 'md.params', sim_name = composite_params["structure_name"])
    md.initSimulation(prmtop_file=prmtop_file, rst7_file=rst7_file)


    # (4) run one of three simulation programs 
    if args.sim == 0:                               # minimization only
        # (4.1) check for necessary topology files
        pyedna.utils.checkFileWithName(f"{composite_params['structure_name']}.prmtop")
        pyedna.utils.checkFileWithName(f"{composite_params['structure_name']}.rst7")
        # (4.2) perform minimization
        md.runMinimization()                
    elif args.sim == 1:                             # equilibration and production only
        # (4.1) check for necessary topology files
        pyedna.utils.checkFileWithName(f"{composite_params['structure_name']}.prmtop")
        pyedna.utils.checkFileWithName(f"min_{composite_params['structure_name']}.ncrst")
        # (4.2) perform equilibration and production
        md.runEquilibration()
        md.runProduction()
    elif args.sim == 2:                             # minimization, equilibration and production
        # (4.1) check for necessary topology files
        pyedna.utils.checkFileWithName(f"{composite_params['structure_name']}.prmtop")
        pyedna.utils.checkFileWithName(f"{composite_params['structure_name']}.rst7")
        # (4.2) perform minimization, equilibration, and production
        md.runMinimization()
        md.runEquilibration()
        md.runProduction()


    # (5) clean directory (set clean to 3 in order to keep everything)
    md.cleanFiles(args.clean)
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Molecular Dynamics Simulation")
    parser.add_argument("--sim", type=int, choices=[0, 1, 2], required=True, help="Simulation type (0-2)")
    parser.add_argument("--clean", type=int, choices=[0, 1, 2, 3], required=True, help="File verbosity (0-3)")
    args = parser.parse_args()

    main(args)
