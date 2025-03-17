import pyedna
import MDAnalysis as mda

# function to create geometry optimized dye structures
def main():

    # (0) set settings for QM (DFT/TDDFT) calculation (TODO : maybe load this from somewhere else)
    settings_dft, _ = pyedna.Trajectory.setQMSettings('qm.params')

    # need to have dye_name.cdx in file to perform geometry optimization on it
    # TODO : read in from command line
    dye_name = 'CY5'
    # this constraint is for phosphate groups linking to double_helix DNA where P-P distance is 6.49 Angstrom
    constraint = ['P', 'distance', 6.49]
    linking_atoms = ['P1', 'P2']
    symmetry_group = "C2"

    # TODO : make check for dye_name.cdx file

    # (1) do forcefield preoptimization with Open Babel from ChemDraw input structure
    # returns .pdb of dye molecule with forcefield-optimized coordinates (without constraint)
    pyedna.quanttools.optimizeStructureFF(dye_name = dye_name,
                                          suffix = 'ff'
                                          )
    
    # (2) (optional) classical force-field optimization subject to point group symmetry of molecule
    # return .pdb of dye molecule with forcefield-optimized coordinates (with symmetry and constraint)
    if symmetry_group is not None:
        if symmetry_group == "C2":
            pyedna.quanttools.optimizeStructureFFSymmetry(in_pdb_file = f"{dye_name}_ff.pdb", 
                                                        out_pdb_file = f"{dye_name}_ff.pdb",
                                                        constraint = constraint, 
                                                        point_group = symmetry_group
                                                        )

    # (3) perform geometry optimization with DFT and return dye_name.pdb as geometry-optimized dye+linker file
    pyedna.quanttools.geometryOptimizationDFT_gpu(f"{dye_name}_ff.pdb",
                                               dye_name = dye_name,
                                               constraint = constraint,
                                               **settings_dft
                                               )

    # TODO : (optional) delete dye_name_ff.pdb file

    # (4) write attachment information of dye
    dye = pyedna.Chromophore(mda.Universe(f"{dye_name}.pdb", format = "PDB"))
    pyedna.Chromophore.writeAttachmentInfo(dye.chromophore_u,
                                           dye_name = dye_name,
                                           linker_atoms = linking_atoms,
                                           linker_group = 'phosphate'
                                           )
    
    # (5) parse attachment information for Chromophore object and attachmen to DNA
    dye.storeSourcePath('./')
    dye.parseAttachment(change_atom_names = False)

    # (6) create forcefield parameters .frcmod and .mol2 for dye
    # NOTE : this creates coordinate files for the 
    dye.createFF(charge = 0)




if __name__ == '__main__':
    main()