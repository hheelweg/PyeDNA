import pyedna 
import os

# TODO : make this a function that feeds in information about the molecules (dyes) and 
# about the DNA sequence and returns a DNA+dye structure (maybe .pdb) and also the necessary
# inputs for molecular dynamics. make this such that the orientation of dyes to DNA is correctly specified. 
def main():
    

    # NOTE : the current implementation starts from the pdb structures of the DNA and the dyes
    # we want to load the dye information from some bib/lib directory that we have yet to implement


    # NOTE : this rn assumes that the DNA .pdb file is located in the current directory
    # TODO : we want to read in ideally a nucleotide string and crearte the DNA "on-the-fly"
    dna_pdb = pyedna.utils.findFileWithName('dna.pdb')


    # set up composite structure starting from DNA
    composite = pyedna.CompositeStructure(dna_pdb)


    # dye names we want to attach to the DNA, these need to exist
    dyes = ['CY5', 'CY3']

    # where do we want to attach the dyes
    attach_residues = [3, 8]

    # name of DNA+sye-structure to create
    name = 'dna_test'

    # look for dyes in specified structure library
    dye_base_dir = os.getenv("DYE_DIR")
    for i, dye in enumerate(dyes):
        # get directory name for dye
        dye_dir = pyedna.utils.findSubdirWithName(dye, dir=dye_base_dir)
        # perform attachment
        composite.prepareAttachment(dye_dir, dye, attach_residues[i], orientation=-1)
    print('tytst', composite.chromophore_list[0].path)
    print('tytst', composite.chromophore_list[1].path)
    

    # write Amber input
    composite.writeAMBERinput(file_name = name)



    pass



if __name__ == "__main__":

    main()