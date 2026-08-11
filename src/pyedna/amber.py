from pathlib import Path
import pandas as pd
import shutil
import os

from . import fileproc as fp
from .dye import load_dye_definitions
from .structure_config import StructureConfig


class AmberSetup:
    """Prepare a docked DNA-dye model for AMBER topology generation."""
    def __init__(self, input_pdb, output_name, workdir=".", water_model="TIP3P",
                solvent_padding=20.0, positive_ion="Na+", negative_ion="Cl-",
                neutralize=True, dna_forcefield="leaprc.DNA.OL15",
                dye_forcefield="leaprc.gaff", water_forcefield="leaprc.water.tip3p"):
        """Store AMBER preparation settings and validate the input structure."""
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

    def final_resid_for_mapping(self, source, mapping):
        """Resolve a dye-library residue to its residue ID in the final PDB."""
        matches = set()

        left = self.bonds[
            (self.bonds["source1"] == source)
            & (self.bonds["resname1"] == mapping.resname)
            & (self.bonds["original_resid1"].astype(int) == mapping.resid)
        ]
        matches.update(left["resid1"].astype(int))

        right = self.bonds[
            (self.bonds["source2"] == source)
            & (self.bonds["resname2"] == mapping.resname)
            & (self.bonds["original_resid2"].astype(int) == mapping.resid)
        ]
        matches.update(right["resid2"].astype(int))

        if len(matches) != 1:
            raise ValueError(
                f"{source}: expected one final residue for "
                f"{mapping.resname} {mapping.resid}, found {sorted(matches)}"
            )

        return matches.pop()

    def load_structure_data(self):
        """Load structure metadata, dye definitions, and explicit bonds."""
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
        """Validate that the bond table contains all columns AMBER needs."""
        if self.bonds is None:
            raise RuntimeError("Structure data has not been loaded")

        required = {
            "resid1", "resname1", "original_resid1", "atom1", "source1",
            "resid2", "resname2", "original_resid2", "atom2", "source2",
        }

        missing = required - set(self.bonds.columns)
        if missing:
            raise ValueError(f"{self.bond_file}: missing columns {sorted(missing)}")

        return self

    def amber_atom_name(self, source, resname, original_resid, atom):
        """Return the post-mapping atom name used by LEaP."""
        if source == "DNA":
            return atom

        dye_name = source.rsplit("_", 1)[0]
        dye = self.dye_definitions[dye_name]

        matches = [
            mapping for mapping in dye.read_amber_mapping()
            if mapping.resname == resname
            and mapping.resid == int(original_resid)
            and mapping.atom == atom
        ]

        if len(matches) > 1:
            raise ValueError(
                f"{source}: ambiguous AMBER mapping for "
                f"{resname} {original_resid} {atom}"
            )

        return matches[0].name if matches else atom

    def prepare_input(self):
        """Apply configured AMBER atom renames to a copy of the docked PDB."""
        output_pdb = self.workdir / f"{self.output_name}.pdb"
        rename = {}

        sources = set(self.bonds["source1"]) | set(self.bonds["source2"])
        sources.discard("DNA")

        for source in sources:
            dye_name = source.rsplit("_", 1)[0]
            dye = self.dye_definitions[dye_name]

            for mapping in dye.read_amber_mapping():
                final_resid = self.final_resid_for_mapping(source, mapping)
                rename[(final_resid, mapping.atom)] = mapping.name

        output = []

        for line in self.input_pdb.read_text().splitlines():
            if line.startswith(("ATOM  ", "HETATM")):
                resid = int(line[22:26])
                atom = line[12:16].strip()
                new_name = rename.get((resid, atom))

                if new_name is not None:
                    line = line[:12] + f"{new_name:>4s}" + line[16:]

            output.append(line)

        output_pdb.write_text("\n".join(output) + "\n")
        self.amber_pdb = output_pdb

        print(f"Prepared AMBER PDB: {output_pdb}")
        return self

    
    @classmethod
    def from_file(cls, path, workdir="."):
        """Create an AMBER setup from an ``amber.params`` file."""
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
        """Write a complete LEaP input using dye metadata and explicit bonds."""
        if self.bonds is None:
            raise RuntimeError("Structure data has not been loaded")
        if not hasattr(self, "amber_pdb"):
            raise RuntimeError("AMBER input structure has not been prepared")

        tleap_file = self.workdir / f"{self.output_name}_tleap.in"

        lines = [
            f"source {self.dna_forcefield}",
            f"source {self.dye_forcefield}",
            f"source {self.water_forcefield}",
            "",
        ]

        for dye in self.dye_definitions.values():
            lines.append(f"# {dye.name}")

            for mol2 in dye.mol2_templates:
                lines.append(f"{mol2.stem} = loadMol2 {mol2}")

            for frcmod in dye.frcmods:
                lines.append(f"loadAmberParams {frcmod}")

            for mapping in dye.read_amber_mapping():
                lines.append(f"set {mapping.resname}.1.{mapping.atom} type {mapping.type}")
                lines.append(f"set {mapping.resname}.1.{mapping.atom} name {mapping.name}")

            lines.append("")

        lines += [
            f"mol = loadPdb {self.amber_pdb}",
            "",
            "# DNA-dye, dye-dye and internal composite-dye bonds",
        ]

        for _, bond in self.bonds.iterrows():
            atom1 = self.amber_atom_name(
                bond["source1"], bond["resname1"],
                bond["original_resid1"], bond["atom1"])

            atom2 = self.amber_atom_name(
                bond["source2"], bond["resname2"],
                bond["original_resid2"], bond["atom2"])

            lines.append(
                f"bond mol.{int(bond['resid1'])}.{atom1} "
                f"mol.{int(bond['resid2'])}.{atom2}"
            )

        water_box = {"TIP3P": "TIP3PBOX"}.get(self.water_model)
        if water_box is None:
            raise ValueError(f"Unsupported water model: {self.water_model!r}")

        lines += [
            "",
            "check mol",
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
