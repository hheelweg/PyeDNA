"""Prepare finalized DNA-dye structures for Amber with tleap."""

from pathlib import Path
import pandas as pd
import re
import subprocess

from pyedna.config import amber_environment, amber_executable, get_config
from pyedna.structure.config import DyePlacement
from pyedna.structure.attachments import (
    DyeLinkerConfig,
    forcefield_id,
    load_dye_definitions,
    tleap_source,
)


class AmberSetup:
    """Prepare a finalized DNA–dye structure and tleap input for Amber."""

    def __init__(self, input_pdb, output_name, workdir=".", source_workdir=None,
                bond_file=None, water_model=None, water_box=None,
                solvent_padding=20.0, positive_ion="Na+", negative_ion="Cl-",
                neutralize=True, dna_forcefield="OL15",
                dye_forcefield="gaff2", water_forcefield="tip3p",
                structure_config=None, dye_dir=None, lnk_dir=None):

        self.workdir = Path(workdir)
        self.source_workdir = Path(source_workdir) if source_workdir is not None else self.workdir
        self.input_pdb = Path(input_pdb)
        self.output_name = output_name
        self.water_model = water_model
        self.water_box = water_box
        self.solvent_padding = solvent_padding
        self.positive_ion = positive_ion
        self.negative_ion = negative_ion
        self.neutralize = neutralize
        self.dna_forcefield = dna_forcefield
        self.dye_forcefield = dye_forcefield
        self.water_forcefield = water_forcefield

        if not self.input_pdb.exists():
            raise FileNotFoundError(f"Input structure not found: {self.input_pdb}")

        self.bond_file = Path(bond_file) if bond_file is not None else self.source_workdir / "structures" / "bonds.csv"
        self.structure_config = structure_config
        self.dye_dir = Path(dye_dir) if dye_dir is not None else None
        self.lnk_dir = Path(lnk_dir) if lnk_dir is not None else None
        self.dye_definitions = {}
        self.bonds = None

    def _generated_workdir(self):
        """Return where structure generation stored linked Amber metadata."""

        amber_dir = self.source_workdir / "structures" / "amber"
        return amber_dir if amber_dir.exists() else self.source_workdir

    def _final_resid_for_mapping(self, source, mapping):
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

    def _load_structure_data(self):
        if not self.bond_file.exists():
            raise FileNotFoundError(f"Bond file not found: {self.bond_file}")

        if self.structure_config is None:
            placements = self._placements_from_bonds()
            generated = self._generated_from_bonds()
        else:
            placements = self.structure_config.dyes
            generated = {
                attachment.name: DyeLinkerConfig.from_names(
                    attachment.dye,
                    attachment.linker,
                    dye_forcefield=self.dye_forcefield,
                    dna_forcefield=self.dna_forcefield,
                    dye_dir=self.dye_dir,
                    lnk_dir=self.lnk_dir,
                )
                for attachment in self.structure_config.attachments
            }

        if self.dye_dir is None and not generated:
            self.dye_dir = get_config().libraries.dye_dir

        self.dye_definitions = load_dye_definitions(
            placements,
            self.dye_dir,
            generated=generated,
            workdir=self._generated_workdir(),
            dye_forcefield=self.dye_forcefield,
        )
        self.bonds = pd.read_csv(self.bond_file)

        return self

    def _placements_from_bonds(self):
        sources = self._dye_sources_from_bonds()
        return [DyePlacement(name=source, sites=[1]) for source in sources]

    def _generated_from_bonds(self):
        generated = {}
        for name in self._dye_sources_from_bonds():
            if self.dye_dir is not None and (Path(self.dye_dir) / name).exists():
                continue
            if "_" not in name:
                continue
            dye, linker = name.split("_", 1)
            try:
                generated[name] = DyeLinkerConfig.from_names(
                    dye,
                    linker,
                    dye_forcefield=self.dye_forcefield,
                    dna_forcefield=self.dna_forcefield,
                    dye_dir=self.dye_dir,
                    lnk_dir=self.lnk_dir,
                )
            except FileNotFoundError:
                continue
        return generated

    def _dye_sources_from_bonds(self):
        if not self.bond_file.exists():
            raise FileNotFoundError(f"Bond file not found: {self.bond_file}")
        bonds = pd.read_csv(self.bond_file)
        required = {"source1", "source2"}
        missing = required - set(bonds.columns)
        if missing:
            raise ValueError(f"{self.bond_file}: missing columns {sorted(missing)}")
        sources = set(bonds["source1"]) | set(bonds["source2"])
        sources.discard("DNA")
        return sorted({str(source).rsplit("_", 1)[0] for source in sources})

    def _validate(self):
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

    def _amber_atom_name(self, source, resname, original_resid, atom):
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

    def _prepare_input(self):
        output_pdb = self.workdir / f"{self.output_name}.pdb"
        rename = {}

        sources = set(self.bonds["source1"]) | set(self.bonds["source2"])
        sources.discard("DNA")

        for source in sources:
            dye_name = source.rsplit("_", 1)[0]
            dye = self.dye_definitions[dye_name]

            for mapping in dye.read_amber_mapping():
                final_resid = self._final_resid_for_mapping(source, mapping)
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
    def from_md_config(cls, md_config, structure, workdir=".", source_workdir=".",
                       dye_dir=None, lnk_dir=None):
        """Create an Amber setup for one structure selected in md.toml."""

        source_workdir = Path(source_workdir)
        input_pdb = source_workdir / md_config.system.structure_path(structure)
        amber = md_config.amber

        return cls(
            input_pdb=input_pdb,
            output_name=md_config.system.name,
            workdir=workdir,
            source_workdir=source_workdir,
            bond_file=source_workdir / md_config.system.structure_directory / "bonds.csv",
            water_model=amber.water_model,
            water_box=amber.water_box,
            solvent_padding=amber.solvent_padding,
            positive_ion=amber.positive_ion,
            negative_ion=amber.negative_ion,
            neutralize=amber.neutralize,
            dna_forcefield=amber.dna_forcefield,
            dye_forcefield=amber.dye_forcefield,
            water_forcefield=amber.water_forcefield,
            dye_dir=dye_dir,
            lnk_dir=lnk_dir,
        )

    @classmethod
    def from_config(cls, structure_config, workdir=".", dye_dir=None, lnk_dir=None):
        """Compatibility constructor for callers that still pass StructureConfig."""

        workdir = Path(workdir)
        amber = structure_config.amber
        legacy_structure = getattr(amber, "model", 1)
        output_name = getattr(amber, "output_name", None) or structure_config.name
        input_pdb = workdir / "structures" / f"{structure_config.name}_{legacy_structure}.pdb"

        return cls(
            input_pdb=input_pdb,
            output_name=output_name,
            workdir=workdir,
            source_workdir=workdir,
            water_model=amber.water_model,
            solvent_padding=amber.solvent_padding,
            positive_ion=amber.positive_ion,
            negative_ion=amber.negative_ion,
            neutralize=amber.neutralize,
            dna_forcefield=amber.dna_forcefield,
            dye_forcefield=amber.dye_forcefield,
            water_forcefield=amber.water_forcefield,
            structure_config=structure_config,
            dye_dir=dye_dir,
            lnk_dir=lnk_dir,
        )

    def prepare(self, run_tleap=True, cleanup_intermediates=True):
        """Prepare Amber inputs and run tleap by default to create MD input files."""

        self._load_structure_data()
        self._validate()
        self._prepare_input()
        self._write_tleap_input()
        if run_tleap:
            self.run_tleap()
            if cleanup_intermediates:
                self.cleanup_intermediates()
        return self

    def cleanup_intermediates(self):
        """Remove generated intermediates after successful tleap."""
        removed = []

        for dye in self.dye_definitions.values():
            for path in dye.linked_intermediates():
                if path.exists():
                    path.unlink()
                    removed.append(path)

        for path in (self.tleap_file, self.workdir / "leap.log"):
            if path.exists():
                path.unlink()
                removed.append(path)

        for path in removed:
            print(f"Removed intermediate: {path}")

        return removed

    def _write_tleap_input(self):
        if self.bonds is None:
            raise RuntimeError("Structure data has not been loaded")
        if not hasattr(self, "amber_pdb"):
            raise RuntimeError("AMBER input structure has not been prepared")

        tleap_file = self.workdir / f"{self.output_name}_tleap.in"

        lines = [
            f"source {tleap_source(self.dna_forcefield, 'dna')}",
            f"source {tleap_source(self.dye_forcefield, 'small')}",
            f"source {tleap_source(self.water_forcefield, 'water')}",
            "",
        ]

        for dye in self.dye_definitions.values():
            lines.append(f"# {dye.name}")

            for mol2 in dye.mol2_templates:
                lines.append(f"{mol2.stem} = loadMol2 {self._tleap_read_path(mol2)}")

            for frcmod in dye.frcmods:
                lines.append(f"loadAmberParams {self._tleap_read_path(frcmod)}")

            for mapping in dye.read_amber_mapping():
                lines.append(f"set {mapping.resname}.1.{mapping.atom} type {mapping.type}")
                lines.append(f"set {mapping.resname}.1.{mapping.atom} name {mapping.name}")

            lines.append("")

        lines += [
            f"mol = loadPdb {self._tleap_read_path(self.amber_pdb)}",
            "",
            "# DNA-dye, dye-dye and internal composite-dye bonds",
        ]

        for _, bond in self.bonds.iterrows():
            atom1 = self._amber_atom_name(
                bond["source1"], bond["resname1"],
                bond["original_resid1"], bond["atom1"])

            atom2 = self._amber_atom_name(
                bond["source2"], bond["resname2"],
                bond["original_resid2"], bond["atom2"])

            lines.append(
                f"bond mol.{int(bond['resid1'])}.{atom1} "
                f"mol.{int(bond['resid2'])}.{atom2}"
            )

        water_box = {"TIP3P": "TIP3PBOX"}.get(self.water_model)
        if water_box is None and self.water_box is not None:
            water_box = self.water_box
        if water_box is None:
            water = str(self.water_forcefield).removeprefix("leaprc.water.").lower()
            water_box = {"tip3p": "TIP3PBOX"}.get(water)
        if water_box is None:
            raise ValueError(f"Unsupported water force field: {self.water_forcefield!r}")

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

    @staticmethod
    def _tleap_read_path(path):
        """Return an absolute path for tleap inputs read from per-model cwd."""

        return Path(path).resolve()

    def _connect_frcmod_path(self):
        """Return expected shared DNA-linker compatibility frcmod path."""
        if self.lnk_dir is None:
            return None
        dye_ff = forcefield_id(self.dye_forcefield)
        dna_ff = forcefield_id(self.dna_forcefield)
        return self.lnk_dir / "connect" / dye_ff / dna_ff / "connectparams.frcmod"

    def _tleap_failure_reasons(self, output, missing):
        """Return fatal tleap issues detected in stdout/stderr/log text."""
        reasons = []
        match = re.search(r"Exiting LEaP:\s*Errors\s*=\s*(\d+)", output)
        if match and int(match.group(1)):
            reasons.append(f"LEaP reported {match.group(1)} errors")

        checks = {
            "FATAL:": "LEaP reported a fatal error",
            "Parameter file was not saved.": "LEaP did not save the parameter file",
            "No torsion terms for atom types": "missing torsion parameters",
            "No angle terms for atom types": "missing angle parameters",
            "No bond terms for atom types": "missing bond parameters",
            "Could not find": "LEaP could not find required data",
            "Unknown atom type": "unknown atom type",
        }
        reasons.extend(reason for text, reason in checks.items() if text in output)
        if missing:
            reasons.append(f"missing expected output files: {missing}")
        return sorted(set(reasons))

    def run_tleap(self):
        """Execute tleap and verify that topology and coordinate files were created."""

        if not hasattr(self, "tleap_file"):
            raise RuntimeError("tleap input has not been prepared")

        leap_log = self.workdir / "leap.log"
        if leap_log.exists():
            leap_log.unlink()

        result = subprocess.run(
            [str(amber_executable("tleap")), "-f", str(self.tleap_file.resolve())],
            cwd=self.workdir,
            text=True,
            capture_output=True,
            env=amber_environment("tleap"),
        )
        output = result.stdout + result.stderr
        if leap_log.exists():
            output += "\n\n# leap.log\n" + leap_log.read_text()

        self.tleap_log = self.workdir / "tleap_amber.log"
        self.tleap_log.write_text(output)

        self.prmtop_file = self.workdir / f"{self.output_name}.prmtop"
        self.rst7_file = self.workdir / f"{self.output_name}.rst7"
        self.solvated_pdb = self.workdir / f"{self.output_name}_solvated.pdb"

        missing = [
            path for path in (self.prmtop_file, self.rst7_file, self.solvated_pdb)
            if not path.exists()
        ]
        reasons = self._tleap_failure_reasons(output, missing)
        if result.returncode != 0:
            reasons.append(f"tleap exited with status {result.returncode}")
        if reasons:
            message = [
                "tleap failed during MD Amber preparation.",
                "",
                "Detected issues:",
                *[f"  - {reason}" for reason in sorted(set(reasons))],
                "",
                f"tleap input: {self.tleap_file}",
                f"tleap log:   {self.tleap_log}",
            ]
            connect_frcmod = self._connect_frcmod_path()
            if connect_frcmod is not None:
                message += [
                    "",
                    "If the failure reports missing DNA-linker bond/angle/dihedral "
                    "terms, add them to the manually curated compatibility file:",
                    f"  {connect_frcmod}",
                ]
            raise RuntimeError("\n".join(message))

        return self
