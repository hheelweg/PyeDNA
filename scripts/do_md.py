import pyedna 
import os


# function to run MD simulation on .prmtop and .rst7 input 
def main():
    
    # parse structure parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    composite_params = pyedna.CompositeStructure.parseCompositeStructure('struc.params')
    print(composite_params)
    
    # parse input parameters for minimization/MD
    md_params = pyedna.MDSimulation.parseInputParams(dna_params = dna_params, file='md.params')
    print('md_params', md_params) 


    # load MDSimulation object
    md = pyedna.MDSimulation(dna_params, 'md.params', sim_name = composite_params["structure_name"])
    print(md.simulation_name)
    
    


if __name__ == '__main__':

    main()
