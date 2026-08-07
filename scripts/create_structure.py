import pyedna 
import os

from pyedna.structure_config import StructureConfig
from pyedna.structure import prepare_dna
from pyedna.dye import load_dye_definitions


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


if __name__ == "__main__":

    main()