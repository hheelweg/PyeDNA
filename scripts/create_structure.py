import pyedna 
import os

# TODO : make this a function that feeds in information about the molecules (dyes) and 
# about the DNA sequence and returns a DNA+dye structure (maybe .pdb) and also the necessary
# inputs for molecular dynamics. make this such that the orientation of dyes to DNA is correctly specified. 
def main():
    

    # NOTE : the current implementation starts from the pdb structures of the DNA and the dyes
    # we want to load the dye information from some bib/lib directory that we have yet to implement
    path = './'
    dna_pdb = pyedna.utils.findFileWithName('dna.pdb', path)


    # set up composite structure starting from DNA
    composite = pyedna.CompositeStructure(dna_pdb, path)
    print(composite.dna.res_names)


    # dye names we want to attach to the DNA
    dyes = ['CY5', 'CY3']
    # look for dyes in specified structure library
    dye_base_dir = os.getenv("DYE_DIR")
    for dye in dyes:
        # get directory name for dye
        dye_dir = pyedna.utils.findSubdirWithName(dye, dir=dye_base_dir)
        print(dye_dir)
        dye_pdb = pyedna.utils.findFileWithName(dye + ".pdb", dir=dye_dir)
        print(dye_pdb)
    
    # TODO : bring this into agreement with current implementation in structure.py




    pass



if __name__ == "__main__":

    main()