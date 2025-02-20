import pyedna 
import os

# TODO : make this a function that feeds in information about the molecules (dyes) and 
# about the DNA sequence and returns a DNA+dye structure (maybe .pdb) and also the necessary
# inputs for molecular dynamics. make this such that the orientation of dyes to DNA is correctly specified. 
def main():
    

    # NOTE : the current implementation starts from the pdb structures of the DNA and the dyes
    # we want to load the dye information from some bib/lib directory that we have yet to implement
    dna_pdb = pyedna.utils.findFileWithName('dna.pdb')
    print(dna_pdb)

    # dye names we want to attach to the DNA
    dye_names = ['CY5', 'CY3']
    # look for dyes in specified structure library
    dye_dir = os.getenv("DYE_DIR")
    print(pyedna.utils.findSubdirWithName(dye_dir, dye_names[0]))



    pass



if __name__ == "__main__":

    main()