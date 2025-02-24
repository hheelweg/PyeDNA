import pyedna 
import os


# function to run MD simulation on .prmtop and .rst7 input 
def main():
    
    # parse structure parameters
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    composite_params = pyedna.CompositeStructure.parseCompositeStructure('struc.params')
    
    
    # locate topology and forcefield
    prmtop_file = pyedna.utils.findFileWithExtension('*.prmtop')
    rst7_file = pyedna.utils.findFileWithExtension('*.rst7')

    
    # load MDSimulation object
    md = pyedna.MDSimulation(dna_params, 'md.params', sim_name = composite_params["structure_name"])
    print(md.simulation_name)
    
    


if __name__ == '__main__':

    main()
