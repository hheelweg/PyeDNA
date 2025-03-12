import pyedna

# function to create geometry optimized dye structures
def main():

    # (0) set settings for QM (DFT/TDDFT) calculation
    settings_dft, _ = pyedna.Trajectory.setQMSettings('qm.params')
    print(settings_dft)

    # need to have dye.pdb in file to perform geometry optimization on it
    # TODO : read in from command line
    pdb_file = 'cy3_unopt.pdb'
    test_out = 'cy3.pdb'

    # (0) TODO : do preoptimization with Open Babel
    # input : .cdx, output : unoptimized.pdb

    # (1) perform geometry optimization with DFT
    constraint = ['P1', 'P2', 'distance', 6.49]

    pyedna.quanttools.geometryOptimization_gpu(pdb_file, test_out, constraint=constraint, **settings_dft)

    # (2) create force field for dye


if __name__ == '__main__':
    main()