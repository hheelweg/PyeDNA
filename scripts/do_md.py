import pyedna 
import os


# function to run MD simulation on .prmtop and .rst7 input 
def main():
    
    # parse structure parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    # parse input parameters for minimization/MD
    md_params = pyedna.MDSimulation.parseInputParams(dna_params = dna_params, file='md.params')
    print('md_params', md_params) 


    # load MDSimulation object
    md = pyedna.MDSimulation(dna_params, 'md.params')
    md.writeAMBERInput(md_params, input_type='min2')
    


if __name__ == '__main__':

    main()
