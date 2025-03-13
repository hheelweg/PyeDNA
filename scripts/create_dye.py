import pyedna
import MDAnalysis as mda

# function to create geometry optimized dye structures
def main():

    # (0) set settings for QM (DFT/TDDFT) calculation (TODO : maybe load this from somewhere else)
    settings_dft, _ = pyedna.Trajectory.setQMSettings('qm.params')

    # need to have dye_name.cdx in file to perform geometry optimization on it
    # TODO : read in from command line
    dye_name = 'CY3'

    # (0) do forcefield preoptimization with Open Babel from ChemDraw input structure
    # returns .pdb of dye moecule with forcefield-optimized coordinates (without constraint)
    pyedna.quanttools.optimizeStructureFF(dye_name = dye_name,
                                          suffix = 'ff'
                                          )
    

    # (1) perform geometry optimization with DFT and return tmp.pdb 
    # this constraint is for phosphate groups linking to double_helix DNA where P-P distance is 6.49 Angstrom
    constraint = ['P', 'distance', 6.49]
    pyedna.quanttools.geometryOptimization_gpu(f"{dye_name}_ff.pdb",
                                               dye_name = dye_name,
                                               constraint = constraint,
                                               **settings_dft
                                               )

    # TODO : delete dye_name_ff.pdb file

    # (2) write attachment information of dye
    dye = pyedna.Chromophore(mda.Universe(f"{dye_name}.pdb", format = "PDB"))
    pyedna.Chromophore.writeAttachmentInfo(dye.chromophore_u,
                                           dye_name = dye_name,
                                           linker_atoms = ['P1', 'P2'],
                                           linker_group = 'phosphate'
                                           )
    
    # (3) parse attachment information for Chromophore object
    dye.storeSourcePath('./')
    dye.parseAttachment(change_atom_names = False)

    # (4) create forcefield parameters .frcmod and .mol2 for dye
    #dye.createFF()




if __name__ == '__main__':
    main()