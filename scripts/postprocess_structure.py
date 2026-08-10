import os
from pathlib import Path

from pyedna.haddock import (load_prepared_dye_instances, reformat_docked_models, select_best_models)
from pyedna.structure_config import StructureConfig


def main():
    workdir = Path.cwd()
    config = StructureConfig.from_file(workdir / "struc_gen.params")

    # (1) Select best HADDOCK models
    structure_dir = workdir / "structures"
    select_best_models(
        run_dir=workdir / "haddock" / "run",
        output_dir=structure_dir,
        top=config.top_models)

    # (2) Reconstruct prepared dye-instance information
    instances = load_prepared_dye_instances(
        dockings=config.dockings,
        dye_dir=os.environ["DYE_DIR"],
        workdir=workdir)

    # (3) Restore original DNA/dye formatting in selected structures
    reformat_docked_models(instances=instances,
                           dna_template=workdir / "haddock" / f"{config.dna_name}_haddock.pdb",
                           bonding_csv=workdir / "haddock" / f"{config.dna_name}_bonding.csv",
                           structure_dir=structure_dir,
                           bond_file=workdir / "haddock" / "bonds.csv")


if __name__ == "__main__":
    main()