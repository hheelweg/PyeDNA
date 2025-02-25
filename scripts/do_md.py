import pyedna 
import argparse


# function to run MD simulation on .prmtop and .rst7 input 
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

    print('test dt, traj_steps ', md.dt, md.traj_steps)

    # do check on parameter parsing
    debug = pyedna.MDSimulation.parseInputParams(dna_params, 'md.params')
    print('test print MD parameters ', debug)

    # perform minimization, equilibration, production run with parameters specified in 'md.params'
    
    print(f"Running MD simulation")
    print(f"Simulation type selected: {args.sim}")

    
    
    # test : perform minimization
    #md.runMinimization(delete_ins=False, delete_outs=False)
    #md.runEquilibration(delete_ins=False, delete_outs=False)
    #md.runProduction(delete_ins=False, delete_outs=False)
    
    

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Molecular Dynamics Simulation")
    parser.add_argument("--sim", type=int, choices=[0, 1, 2, 3], required=True, help="Simulation type (0-3)")

    args = parser.parse_args()

    main(args)
