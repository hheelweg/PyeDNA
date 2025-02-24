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

    
    # load MDSimulation object and initialize simulation
    md = pyedna.MDSimulation(dna_params, 'md.params', sim_name = composite_params["structure_name"])
    md.initSimulation(prmtop_file=prmtop_file, rst7_file=rst7_file)
    
    # test perform minimization
    md.runMinimization()
    
    


if __name__ == '__main__':

    main()
