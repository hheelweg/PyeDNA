from pathlib import Path

from pyedna.amber import AmberSetup


def main():
    workdir = Path.cwd()
    setup = AmberSetup.from_file(workdir / "amber.params", workdir=workdir)

    print(f"Input structure : {setup.input_pdb}")
    print(f"Output directory: {setup.output_dir}")
    print(f"Water model     : {setup.water_model}")
    print(f"Padding         : {setup.solvent_padding} Å")
    print(f"Ions            : {setup.positive_ion}, {setup.negative_ion}")
    print(f"Neutralize      : {setup.neutralize}")


if __name__ == "__main__":
    main()