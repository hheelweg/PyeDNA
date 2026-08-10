from pathlib import Path

from pyedna.amber import AmberSetup


def main():
    workdir = Path.cwd()

    setup = AmberSetup.from_file(workdir / "amber.params", workdir=workdir)
    setup.load_structure_data()
    setup.validate()
    setup.prepare_input()
    setup.write_tleap_input()

    print(f"Input structure : {setup.input_pdb}")
    print(f"Bond file       : {setup.bond_file}")
    print(f"Dyes            : {', '.join(setup.dye_definitions)}")
    print(f"Bonds           : {len(setup.bonds)}")

    for dye in setup.dye_definitions.values():
        print(f"  {dye.name}:")
        print(f"    MOL2    : {dye.mol2}")
        for frcmod in dye.frcmods:
            print(f"    frcmod  : {frcmod}")


if __name__ == "__main__":
    main()