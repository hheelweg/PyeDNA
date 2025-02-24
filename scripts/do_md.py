import pyedna 
import os


# function to run MD simulation on .prmtop and .rst7 input 
def main():
    
    # parse simulation parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    print(dna_params, flush = True)


    # load MDSimulation object
    md = pyedna.MDSimulation(dna_params)
    


if __name__ == '__main__':

    main()
