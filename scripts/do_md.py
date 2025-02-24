import pyedna 
import os


# function to run MD simulation on .prmtop and .rst7 input 
def main():
    
    # parse simulation parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    print('DNA test', dna_params)
    min_params = pyedna.MDSimulation.parseMinimizationParams(dna_params = dna_params, file='md.params')
    print('min_params', min_params)


    # load MDSimulation object
    md = pyedna.MDSimulation(dna_params, 'md.params')
    


if __name__ == '__main__':

    main()
