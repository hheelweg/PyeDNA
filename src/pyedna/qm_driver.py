import argparse
from joblib import load, dump

from . import quanttools as qm
from . import trajectory as traj
from . import utils



def main(molecule_id):

    # (0) set settings for QM (DFT/TDDFT) calculation
    settings_dft, settings_tddft = traj.Trajectory.setQMSettings('qm.params')

    # (0) load output information to see which outputs we need to store
    output_params_file = utils.findFileWithName('traj.params')
    output_keys = traj.Trajectory.parseParameters(output_params_file)
    # intialize value dict for storing outputs
    values = {key: None for key in output_keys}

    # (1) load chromophore pyscf input from cache as DFT/TDDFT input
    chromophore_conv = load(f"input_{molecule_id}.joblib")

    # 

    # (1) load fragment information
    # TODO : only load this if specified
    # chromophore_fragments = load(f"fragments_{molecule_id}.joblib")

    # (2) perform DFT/TDDFT calculation and store outputs
    values['mol'], values['mf'], values['occ'], values['virt'], values['orbit_enrgs'] = qm.doDFT_gpu(chromophore_conv, molecule_id, **settings_dft)
    if settings_tddft.pop("do_tddft", False):
        # TODO : only load this if chromophore_fragments specified
        #tddft_output = qm.doTDDFT_gpu(values['mol'], values['mf'], values['occ'], values['virt'], output_keys, fragments = chromophore_fragments, **settings_tddft)
        tddft_output = qm.doTDDFT_gpu(values['mol'], values['mf'], values['occ'], values['virt'], output_keys, fragments = None, **settings_tddft)
        values.update(tddft_output)
    
    # (3) output quantities of interest
    # TODO : might want to add that specific outputs are only possible if do_tddft is set to True
    output = {key: values[key] for key, value in output_keys.items() if value}
    for key in output:
        dump(values[key], f"{key}_{molecule_id}.joblib")



if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser(description="Run DFT and optional TDDFT simulations on molecule")
    parser.add_argument("molecule_id", type=int, help="Molecule ID (integer)")                              # specifies residue name of molecule
    args = parser.parse_args()

    # run main
    main(args.molecule_id)


