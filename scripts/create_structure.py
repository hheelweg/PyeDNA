import os
from pathlib import Path


import pyedna 
from pyedna.structure_config import StructureConfig
from pyedna.structure import prepare_dna
from pyedna.dye import load_dye_definitions, create_dye_instances
from pyedna.haddock import prepare_dye_topologies, combine_ligand_topologies, prepare_dna_for_haddock, write_bond_restraints, write_docking_config, prepare_dna_for_haddock


def main():

    
    # (0) Read in parameters for DNA strcture creation and dye attachment from .params file
    workdir = Path.cwd()
    config = StructureConfig.from_file(workdir / "struc.params")

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

    # (4) Combine topology/parameter files for unique dye types
    top_file, par_file = combine_ligand_topologies(dye_instances, workdir=Path.cwd())

    # (5) Prepare DNA structure for HADDOCK
    haddock_dna_pdb, bonding_csv = prepare_dna_for_haddock(dna_pdb=dna_pdb, instances=dye_instances, workdir=Path.cwd())

    # (6) Generate HADDOCK bond restraints
    restraint_file, bond_file = write_bond_restraints(dye_instances, haddock_dna_pdb,
                                                      output=Path.cwd() / "haddock" / "bond_restraint.tbl",
                                                      bond_output=Path.cwd() / "haddock" / "bonds.csv")

    # (7) Generate HADDOCK configuration
    user_docking_config = Path.cwd() / "user_docking_config.toml"
    if not user_docking_config.exists():
        user_docking_config = None

    docking_config = write_docking_config(
        dna_pdb=haddock_dna_pdb, instances=dye_instances, top_file=top_file,
        par_file=par_file, restraint_file=restraint_file, workdir=Path.cwd(),
        user_config=user_docking_config)


if __name__ == "__main__":

    main()