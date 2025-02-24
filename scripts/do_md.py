import pyedna 
import os


# function to run MD simulation on .prmtop and .rst7 input 
def main():
    
    # parse simulation parameters
    struc_params = pyedna.CreateDNA('struc.params')
    print(struc_params)



    # load MDSimulation object
    md = pyedna.MDSimulation()
    


if __name__ == '__main__':

    main()
