import pyedna 
import os


# function to run MD simulation on .prmtop and .rst7 input 
def main():
    
    # parse structure parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    # parse input parameters for minimization/MD
    md_params = pyedna.MDSimulation.parseInputParams(dna_params = dna_params, file='md.params')
    print('min_params', md_params) 


    # load MDSimulation object
    md = pyedna.MDSimulation(dna_params, 'md.params')
    md.writeMinimizationInput()
    


if __name__ == '__main__':

    main()
