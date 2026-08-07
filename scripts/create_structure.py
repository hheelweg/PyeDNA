import pyedna 
import os

from pyedna.structure_config import StructureConfig
from pyedna.structure import prepare_dna


def main():
    

    # (0) Read in parameters for DNA strcture creation and dye attachment from .params file
    config = StructureConfig.from_file("struc_gen.params")

    # (1) Prepare / locate DNA structure
    dna_pdb = prepare_dna(
        config=config,
        dna_dir=os.environ["DNA_DIR"],
    )

    print(f"DNA structure: {dna_pdb}")


if __name__ == "__main__":

    main()