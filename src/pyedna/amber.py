from pathlib import Path
import pandas as pd
import shutil
import os

from . import fileproc as fp
from .dye import load_dye_definitions
from .structure_config import StructureConfig


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

        self.bond_file = self.workdir / "structures" / "bonds.csv"
        self.structure_config = None
        self.dye_definitions = {}
        self.bonds = None

    def load_structure_data(self):
        if not self.bond_file.exists():
            raise FileNotFoundError(f"Bond file not found: {self.bond_file}")

        struc_params = self.workdir / "struc.params"
        if not struc_params.exists():
            raise FileNotFoundError(f"Structure parameter file not found: {struc_params}")

        self.structure_config = StructureConfig.from_file(struc_params)
        self.dye_definitions = load_dye_definitions(
            self.structure_config.dockings, os.environ["DYE_DIR"])
        self.bonds = pd.read_csv(self.bond_file)

        return self

    def validate(self):
        if self.bonds is None:
            raise RuntimeError("Structure data has not been loaded")

        required = {"resid1", "atom1", "resid2", "atom2"}
        missing = required - set(self.bonds.columns)
        if missing:
            raise ValueError(f"{self.bond_file}: missing columns {sorted(missing)}")

        return self

    def prepare_input(self):
        output_pdb = self.workdir / f"{self.output_name}.pdb"
        shutil.copy2(self.input_pdb, output_pdb)
        self.amber_pdb = output_pdb
        return self

    

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