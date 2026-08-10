from pathlib import Path
import pandas as pd
import shutil
import os

from . import fileproc as fp
from .dye import load_dye_definitions
from .structure_config import StructureConfig


class AmberSetup:
    def __init__(self, input_pdb, output_name, workdir=".", water_model="TIP3P",
                solvent_padding=20.0, positive_ion="Na+", negative_ion="Cl-",
                neutralize=True, dna_forcefield="leaprc.DNA.OL15",
                dye_forcefield="leaprc.gaff", water_forcefield="leaprc.water.tip3p"):
        self.workdir = Path(workdir)
        self.input_pdb = Path(input_pdb)
        self.output_name = output_name
        self.water_model = water_model
        self.solvent_padding = solvent_padding
        self.positive_ion = positive_ion
        self.negative_ion = negative_ion
        self.neutralize = neutralize
        self.dna_forcefield = dna_forcefield
        self.dye_forcefield = dye_forcefield
        self.water_forcefield = water_forcefield

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

    def amber_atom_name(self, source, atom):
        if source == "DNA":
            return atom

        dye_name = source.rsplit("_", 1)[0]
        dye = self.dye_definitions[dye_name]
        mappings = {mapping.atom: mapping.name for mapping in dye.read_amber_mapping()}

        return mappings.get(atom, atom)

    def prepare_input(self):
        output_pdb = self.workdir / f"{self.output_name}.pdb"
        lines = self.input_pdb.read_text().splitlines()

        amber_mappings = {
            dye.name: {mapping.atom: mapping.name for mapping in dye.read_amber_mapping()}
            for dye in self.dye_definitions.values()}

        output = []
        for line in lines:
            if not line.startswith(("ATOM  ", "HETATM")):
                output.append(line)
                continue

            resname = line[17:20].strip()
            atom_name = line[12:16].strip()

            if resname in amber_mappings and atom_name in amber_mappings[resname]:
                new_name = amber_mappings[resname][atom_name]
                line = line[:12] + f"{new_name:>4s}" + line[16:]

            output.append(line)

        output_pdb.write_text("\n".join(output) + "\n")
        self.amber_pdb = output_pdb

        print(f"Prepared AMBER PDB: {output_pdb}")
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
            dna_forcefield=params.get("dna_forcefield", "leaprc.DNA.bsc1"),
            dye_forcefield=params.get("dye_forcefield", "leaprc.gaff2"),
        )


    def write_tleap_input(self):
        if self.bonds is None:
            raise RuntimeError("Structure data has not been loaded")
        if not hasattr(self, "amber_pdb"):
            raise RuntimeError("AMBER input structure has not been prepared")

        tleap_file = self.workdir / f"{self.output_name}_tleap.in"

        lines = [f"source {self.dna_forcefield}",
                 f"source {self.dye_forcefield}",
                 f"source {self.water_forcefield}",
                 "",]

        # Load dye templates and parameters
        for dye in self.dye_definitions.values():
            lines.append(f"# {dye.name}")

            for mol2 in dye.mol2_templates:
                template_name = mol2.stem
                lines.append(f"{template_name} = loadMol2 {mol2}")

            for frcmod in dye.frcmods:
                lines.append(f"loadAmberParams {frcmod}")

            for mapping in dye.read_amber_mapping():
                lines.append(f"set {mapping.resname}.1.{mapping.atom} type {mapping.type}")
                lines.append(f"set {mapping.resname}.1.{mapping.atom} name {mapping.name}")

            lines.append("")

        # Load final cleaned DNA+dye PDB
        lines += [
            f"mol = loadPdb {self.amber_pdb}",
            "",
            "# DNA-dye and dye-dye bonds",
        ]

        for _, bond in self.bonds.iterrows():
            atom1 = self.amber_atom_name(bond["source1"], bond["atom1"])
            atom2 = self.amber_atom_name(bond["source2"], bond["atom2"])

            lines.append(
                f"bond mol.{int(bond['resid1'])}.{atom1} "
                f"mol.{int(bond['resid2'])}.{atom2}"
            )

        water_box = {"TIP3P": "TIP3PBOX"}.get(self.water_model)
        if water_box is None:
            raise ValueError(f"Unsupported water model: {self.water_model!r}")

        lines += [
            "",
            f"solvateBox mol {water_box} {self.solvent_padding}",
        ]

        if self.neutralize:
            lines += [
                f"addIons mol {self.positive_ion} 0",
                f"addIons mol {self.negative_ion} 0",
            ]

        lines += [
            "",
            f"saveAmberParm mol {self.output_name}.prmtop {self.output_name}.rst7",
            f"savePdb mol {self.output_name}_solvated.pdb",
            "quit",
        ]

        tleap_file.write_text("\n".join(lines) + "\n")
        self.tleap_file = tleap_file

        print(f"Wrote {tleap_file}")
        return tleap_file