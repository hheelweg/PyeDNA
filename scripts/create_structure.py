import os
from pathlib import Path

import pyedna 
from pyedna.structure_config import StructureConfig
from pyedna.structure import prepare_dna
from pyedna.dye import load_dye_definitions, create_dye_instances
from pyedna.haddock import prepare_dye_topologies


def main():

    
    # (0) Read in parameters for DNA strcture creation and dye attachment from .params file
    config = StructureConfig.from_file("struc.params")

    # (1) Prepare / locate DNA structure
    dna_pdb = prepare_dna(config=config,
                          dna_dir=os.environ["DNA_DIR"],)

    print(f"DNA structure: {dna_pdb}")

    # (2) Resolve dye definitions from user dye library
    dye_definitions = load_dye_definitions(
        dockings=config.dockings,
        dye_dir=os.environ["DYE_DIR"],
    )

    print("\nDye definitions:")
    for dye in dye_definitions.values():
        print(f"  {dye.name}")
        print(f"    MOL2:   {dye.mol2}")
        print(f"    attach: {dye.attach}")


    dye_instances = create_dye_instances(config.dockings, dye_definitions)
    print("\nDye instances:")
    for dye in dye_instances:
        print(f"  {dye.name}: dye={dye.definition.name}, residues={dye.residues}, segid={dye.segid}")

    # (3) Prepare HADDOCK topology files for all dye instances
    topology_script = Path(os.environ["PYEDNA_HOME"]) / "scripts" / "haddock" / "create_topology.sh"
    dye_topologies = prepare_dye_topologies(dye_instances, workdir=Path.cwd(), script=topology_script)


if __name__ == "__main__":

    main()