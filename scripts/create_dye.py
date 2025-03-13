import pyedna
import MDAnalysis as mda

# function to create geometry optimized dye structures
def main():

    # (0) set settings for QM (DFT/TDDFT) calculation
    settings_dft, _ = pyedna.Trajectory.setQMSettings('qm.params')

    # need to have dye.pdb in file to perform geometry optimization on it
    # TODO : read in from command line
    pdb_file = 'cy3_unopt.pdb'
    test_out = 'cy3.pdb'
    dye_name = 'CY3'

    # (0) TODO : do preoptimization with Open Babel
    # input : .cdx, output : unoptimized.pdb
    pyedna.quanttools.optimizeStructureFF(moleculeName = dye_name)
    

    # (1) perform geometry optimization with DFT and return tmp.pdb 
    # this constraint is for phosphate groups linking to double_helix DNA where P-P distance is 6.49 Angstrom
    constraint = ['P1', 'P2', 'distance', 6.49]
    # pyedna.quanttools.geometryOptimization_gpu(pdb_file, test_out, constraint=constraint, **settings_dft)

    # clean outputted tmp.pdb file
    pyedna.structure.cleanPDB('tmp.pdb', test_out, res_code = dye_name)

    # (2) write information attachment of dye
    dye = pyedna.Chromophore(mda.Universe(test_out, format = "PDB"))
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