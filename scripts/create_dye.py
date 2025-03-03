import pyedna

# function to create geometry optimized dye structures
def main():

    # (0) set settings for QM (DFT/TDDFT) calculation
    settings_dft, settings_tddft = pyedna.Trajectory.setQMSettings('qm.params')

    # need to have dye.pdb in file to perform geometry optimization on it
    pdb_file = 'cy3_unopt.pdb'
    test_out = 'test.pdb'

    # (1) perform geometry optimization
    pyedna.quanttools.geometryOptimization_gpu(pdb_file, test_out, **settings_dft)



if __name__ == '__main__':
    main()