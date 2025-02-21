import pyedna 
import os

# TODO : make this a function that feeds in information about the molecules (dyes) and 
# about the DNA sequence and returns a DNA+dye structure (maybe .pdb) and also the necessary
# inputs for molecular dynamics. make this such that the orientation of dyes to DNA is correctly specified. 
def main():
    
    # TODO : read these parameters from some struc.params file for example 
    # dye names we want to attach to the DNA, these need to exist
    dyes = ['CY5', 'CY3']
    # where do we want to attach the dyes
    attach_residues = [3, 8]
    # name of DNA+sye-structure to create
    name = 'dna_3nt'


    # NOTE : this rn assumes that the DNA .pdb file is located in the current directory
    # TODO : we want to read in ideally a nucleotide string and crearte the DNA "on-the-fly"
    dna_pdb = pyedna.utils.findFileWithName('dna.pdb')

    # NOTE : (alternatively) can set up DNA_DIR with DNA structure of interest


    # set up composite structure starting from DNA
    composite = pyedna.CompositeStructure(dna_pdb)


    # load dye library with specified location
    # TODO : might want to add some sample/default dye library to PyeDNA
    dye_base_dir = os.getenv("DYE_DIR")

    # loop through dyes and perform attachment
    for i, dye in enumerate(dyes):
        # get directory name for dye
        dye_dir = pyedna.utils.findSubdirWithName(dye, dir=dye_base_dir)
        # perform attachment (TODO : enable different orientations in the future)
        composite.prepareAttachment(dye_dir, dye, attach_residues[i], orientation=-1)

    # write AMBER input (.rst and .prmtop files)
    # TODO : write this s.t. we can select whether we want to have .pdb output file etc. 
    composite.writeAMBERinput(file_name = name)




if __name__ == "__main__":

    main()