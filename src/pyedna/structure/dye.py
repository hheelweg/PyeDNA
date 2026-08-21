"""Represent dye-library definitions and prepared docking instances."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import fileproc as fp


@dataclass(frozen=True)
class AttachmentAtom:
    resname: str
    resid: int
    atom: str


@dataclass(frozen=True)
class AmberAtomMapping:
    resname: str
    resid: int
    atom: str
    name: str
    type: str


@dataclass(frozen=True)
class DyeDefinition:
    name: str
    directory: Path
    mol2: Path
    mol2_templates: list[Path]
    frcmods: list[Path]
    attach: Optional[Path]
    attachment: Optional[dict[str, AttachmentAtom]] = None

    @classmethod
    def from_library(cls, name, dye_dir):
        directory = Path(dye_dir) / name
        mol2 = directory / f"{name}.mol2"
        attach = directory / f"{name}.attach"
        params_file = directory / "dye.params"

        for path in (mol2, attach, params_file):
            if not path.exists():
                raise FileNotFoundError(f"Missing dye input: {path}")

        params = fp.readParams(params_file)
        mol2_templates = [directory / filename for filename in params.get("mol2_templates", [])]
        frcmods = [directory / filename for filename in params.get("frcmods", [])]

        if not mol2_templates:
            raise ValueError(f"{params_file}: no mol2_templates specified")
        if not frcmods:
            raise ValueError(f"{params_file}: no frcmods specified")

        missing = [str(path) for path in mol2_templates + frcmods if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing AMBER files: {missing}")

        return cls(name=name, directory=directory, mol2=mol2,
                mol2_templates=mol2_templates, frcmods=frcmods, attach=attach)

    @classmethod
    def from_generated(cls, name, dyelnk, workdir):
        """Load a dye-linker definition generated in the structure workdir."""
        workdir = Path(workdir)
        mol2 = workdir / f"{name}_linked.mol2"
        linked_frcmod = workdir / f"{name}_linked.frcmod"
        mol2_templates = [
            dyelnk.linker5_mol2,
            dyelnk.dye_mol2,
            dyelnk.linker3_mol2,
        ]
        frcmods = [path.with_suffix(".frcmod") for path in mol2_templates]
        frcmods.append(dyelnk.linker_connect_frcmod)
        frcmods.append(linked_frcmod)
        attachment = {
            end: AttachmentAtom(resname=resname, resid=resid, atom=atom)
            for end, (resname, resid, atom)
            in dyelnk.structure_attachment_records().items()
        }

        missing = [str(path) for path in [mol2] + mol2_templates + frcmods
                   if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing generated dye files: {missing}")

        return cls(
            name=name,
            directory=workdir,
            mol2=mol2,
            mol2_templates=mol2_templates,
            frcmods=frcmods,
            attach=None,
            attachment=attachment,
        )


    def read_attachment(self):
        if self.attachment is not None:
            return self.attachment
        if self.attach is None:
            raise FileNotFoundError(f"{self.name}: no attachment metadata available")

        data = {}

        for line in self.attach.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            fields = line.split()
            if fields[0] not in {"5'", "3'"}:
                continue
            if len(fields) != 4:
                raise ValueError(f"{self.attach}: invalid attachment line: {line}")

            end, resname, resid, atom = fields
            data[end] = AttachmentAtom(resname=resname, resid=int(resid), atom=atom)

        if set(data) != {"5'", "3'"}:
            raise ValueError(f"{self.attach}: must define exactly 5' and 3'")

        return data


    def write_attachment(self, output_file):
        """Write DNA-facing attachment metadata for this definition."""
        data = self.read_attachment()
        text = "".join(
            f"{end} {atom.resname} {atom.resid} {atom.atom}\n"
            for end, atom in data.items()
        )
        Path(output_file).write_text(text)
        return output_file

    def read_amber_mapping(self):
        if self.attach is None:
            return []

        mappings = []

        for line in self.attach.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            fields = line.split()
            if fields[0] != "AMBER":
                continue
            if len(fields) != 6:
                raise ValueError(f"{self.attach}: invalid AMBER mapping line: {line}")

            _, resname, resid, atom, name, atom_type = fields
            mappings.append(AmberAtomMapping(
                resname=resname, resid=int(resid), atom=atom,
                name=name, type=atom_type))

        return mappings

    def read_inter_residue_bonds(self):
        atoms, bonds = {}, []
        section = None

        for line in self.mol2.read_text().splitlines():
            if line.startswith("@<TRIPOS>ATOM"):
                section = "atoms"
                continue
            if line.startswith("@<TRIPOS>BOND"):
                section = "bonds"
                continue
            if line.startswith("@<TRIPOS>"):
                section = None
                continue
            if not line.strip():
                continue

            fields = line.split()

            if section == "atoms":
                atoms[int(fields[0])] = {
                    "atom": fields[1],
                    "resid": int(fields[6]),
                    "resname": fields[7],
                }

            elif section == "bonds":
                atom1, atom2 = atoms[int(fields[1])], atoms[int(fields[2])]

                if atom1["resid"] != atom2["resid"]:
                    bonds.append({
                        "resname1": atom1["resname"], "resid1": atom1["resid"], "atom1": atom1["atom"],
                        "resname2": atom2["resname"], "resid2": atom2["resid"], "atom2": atom2["atom"],
                    })

        return bonds


@dataclass
class DyeInstance:
    """Represent one physical dye and its prepared HADDOCK artifacts."""

    definition: DyeDefinition
    residues: list[int]
    name: str
    segid: str
    charge: Optional[int] = None
    resname: Optional[str] = None
    directory: Optional[Path] = None
    pdb: Optional[Path] = None
    top: Optional[Path] = None
    par: Optional[Path] = None
    attach: Optional[Path] = None
    mapping: Optional[Path] = None

    def set_prepared_paths(self, workdir):
        """Populate the predictable artifact paths for this dye instance."""

        self.directory = Path(workdir) / "haddock" / self.name
        self.pdb = self.directory / f"{self.name}_haddock.pdb"
        self.top = self.directory / f"{self.name}_haddock.top"
        self.par = self.directory / f"{self.name}_haddock.par"
        self.attach = self.directory / f"{self.name}.attach"
        self.mapping = self.directory / f"{self.name}_mapping.csv"
        return self


def load_dye_definitions(placements, dye_dir, generated=None, workdir="."):
    """Load each unique dye definition requested by the docking configuration."""

    names = dict.fromkeys(placement.name for placement in placements)
    generated = generated or {}

    return {
        name: (
            DyeDefinition.from_generated(name, generated[name], workdir)
            if name in generated else DyeDefinition.from_library(name, dye_dir)
        )
        for name in names
    }


def create_dye_instances(placements, definitions):
    """Create ordered, uniquely named dye instances for the requested docking sites."""

    counts = {}
    segids = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    instances = []

    if len(placements) > len(segids):
        raise ValueError(f"At most {len(segids)} dye instances are supported")

    for i, placement in enumerate(placements):
        counts[placement.name] = counts.get(placement.name, 0) + 1
        name = f"{placement.name}_{counts[placement.name]}"
        instances.append(DyeInstance(
            definition=definitions[placement.name],
            residues=placement.sites,
            name=name,
            segid=segids[i],
        ))

    return instances
