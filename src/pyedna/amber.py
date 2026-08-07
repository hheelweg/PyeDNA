from pathlib import Path

from . import fileproc as fp


class AmberSetup:
    def __init__(self, input_pdb, output_name, workdir=".", water_model="TIP3P",
                 solvent_padding=20.0, positive_ion="Na+", negative_ion="Cl-", neutralize=True):
        self.workdir = Path(workdir)
        self.input_pdb = Path(input_pdb)
        self.output_name = output_name
        self.water_model = water_model
        self.solvent_padding = solvent_padding
        self.positive_ion = positive_ion
        self.negative_ion = negative_ion
        self.neutralize = neutralize

        if not self.input_pdb.exists():
            raise FileNotFoundError(f"Input structure not found: {self.input_pdb}")

        self.output_dir = self.workdir / "amber" / self.output_name

    @classmethod
    def from_file(cls, path, workdir="."):
        params = fp.readParams(path)
        workdir = Path(workdir)

        structure = params.get("structure")
        output_name = params.get("output_name")

        if not structure:
            raise ValueError("'structure' must be specified in amber.params")

        if not output_name:
            output_name = Path(structure).stem

        input_pdb = workdir / "structures" / structure

        return cls(
            input_pdb=input_pdb,
            output_name=output_name,
            workdir=workdir,
            water_model=params.get("water_model", "TIP3P"),
            solvent_padding=params.get("solvent_padding", 20.0),
            positive_ion=params.get("positive_ion", "Na+"),
            negative_ion=params.get("negative_ion", "Cl-"),
            neutralize=params.get("neutralize", True),
        )