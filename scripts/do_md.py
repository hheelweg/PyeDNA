import pyedna 
import argparse
import torch

# # detect available GPUs 
# num_gpus = torch.cuda.device_count()
# if num_gpus < 2:
#     raise RuntimeError("Error: Less than 2 GPUs detected! Check SLURM \
#                        allocation and adjust accordingly.")


def main(args):
    
    # parse structure parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    composite_params = pyedna.CompositeStructure.parseCompositeStructure('struc.params')
    

    # locate topology and forcefield
    prmtop_file = pyedna.utils.findFileWithExtension('.prmtop')
    rst7_file = pyedna.utils.findFileWithExtension('.rst7')

    
    # load MDSimulation object and initialize simulation by feeding topology and forcefield files
    md = pyedna.MDSimulation(dna_params, 'md.params', sim_name = composite_params["structure_name"])
    md.initSimulation(prmtop_file=prmtop_file, rst7_file=rst7_file)

    # TODO : based on args.sim, add a checkpoint function here that checks whether required files are available!
    # use function checkInputFiles for this

    # perform minimization, equilibration, production run with parameters specified in 'md.params'
    # run one of various simulation programs 
    if args.sim == 0:                               # minimization only
        md.runMinimization()                
    elif args.sim == 1:                             # equilibration only             
        md.runEquilibration()
    elif args.sim == 2:                             # production only
        md.runProduction()
    elif args.sim == 3:                             # minimization, equilibration and production
        md.runMinimization()
        md.runEquilibration()
        md.runProduction()

    # clean directory (set clean to 3 in order to keep everything)
    md.cleanFiles(args.clean)
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Molecular Dynamics Simulation")
    parser.add_argument("--sim", type=int, choices=[0, 1, 2, 3], required=True, help="Simulation type (0-3)")
    parser.add_argument("--clean", type=int, choices=[0, 1, 2, 3], required=True, help="File verbosity (0-3)")
    args = parser.parse_args()

    main(args)
