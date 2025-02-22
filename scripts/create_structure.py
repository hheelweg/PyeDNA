import pyedna 
import os

# TODO : make this a function that feeds in information about the molecules (dyes) and 
# about the DNA sequence and returns a DNA+dye structure (maybe .pdb) and also the necessary
# inputs for molecular dynamics. make this such that the orientation of dyes to DNA is correctly specified. 
def main():
    
    # # TODO : read these parameters from some struc.params file for example 
    # # dye names we want to attach to the DNA, these need to exist
    # dyes = ['CY5', 'CY3']
    # # where do we want to attach the dyes
    # attach_residues = [3, 8]
    # # name of DNA+sye-structure to create
    # name = 'dna_3nt'

    # (0) read in parameters for DNA strcture creation and dye attachment from .params file
    dna_params = pyedna.CreateDNA.parseDNAStructure('struc.params')
    composite_params = pyedna.CompositeStructure.parseCompositeStructure('struc.params')
    dyes = composite_params["dyes"]
    dye_positions = composite_params["dye_postions"]


    create = pyedna.CreateDNA(name = dna_params["dna_name"], type = dna_params["dna_type"])
    create.feedDNAseq(DNA_sequence = dna_params["dna_sequence"])
    create.createDNA()
    # NOTE : this creates DNA structure on-the-fly (alternatively) can set up DNA_DIR with DNA structure of interest


    # (2) Set up composite structure starting from DNA
    composite = pyedna.CompositeStructure(f"{dna_params['dna_name']}.pdb")


    # load dye library with specified location
    # TODO : might want to add some sample/default dye library to PyeDNA
    dye_base_dir = os.getenv("DYE_DIR")

    # loop through dyes and perform attachment
    for i, dye in enumerate(dyes):
        # get directory name for dye
        dye_dir = pyedna.utils.findSubdirWithName(dye, dir=dye_base_dir)
        # perform attachment (TODO : enable different orientations in the future)
        composite.prepareAttachment(dye_dir, dye, dye_positions[i], orientation=-1)

    # write AMBER input (.rst and .prmtop files)
    # TODO : write this s.t. we can select whether we want to have .pdb output file etc. 
    composite.writeAMBERinput(file_name = composite_params["structure_name"])




if __name__ == "__main__":

    main()