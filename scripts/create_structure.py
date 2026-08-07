import pyedna 
import os

from pyedna.structure_config import StructureConfig

# TODO : make this a function that feeds in information about the molecules (dyes) and 
# about the DNA sequence and returns a DNA+dye structure (maybe .pdb) and also the necessary
# inputs for molecular dynamics. make this such that the orientation of dyes to DNA is correctly specified. 
def main():
    

    # (0) Read in parameters for DNA strcture creation and dye attachment from .params file
    config = StructureConfig.from_file("struc.params")

    print(config)
    print(config.dockings)




if __name__ == "__main__":

    main()