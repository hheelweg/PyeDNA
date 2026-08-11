from dataclasses import dataclass
from pathlib import Path
from . import fileproc as fp

@dataclass(frozen=True)
class AttachmentAtom:
    """Identify one attachment atom in the assembled dye definition."""
    resname: str
    resid: int
    atom: str


@dataclass(frozen=True)
class AmberAtomMapping:
    """Describe an optional LEaP atom name and type conversion."""
    resname: str
    resid: int
    atom: str
    name: str
    type: str


@dataclass(frozen=True)
class DyeDefinition:
    """Represent the files and metadata for one reusable dye."""
    name: str
    directory: Path
    mol2: Path
    mol2_templates: list[Path]
    frcmods: list[Path]
    attach: Path

    @classmethod
    def from_library(cls, name, dye_dir):
        """Load and validate a dye definition from the dye library."""
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


    def read_attachment(self):
        """Read the required 5' and 3' external attachment atoms."""
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


    def read_amber_mapping(self):
        """Read optional AMBER name/type mappings from the attachment file."""
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
        """Extract bonds crossing residue boundaries in the assembled MOL2."""
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
    """Track one placed occurrence of a dye during structure generation."""
    definition: DyeDefinition
    residues: list[int]
    name: str
    segid: str
    

def load_dye_definitions(dockings, dye_dir):
    """Load each distinct dye referenced by the docking specifications."""
    names = dict.fromkeys(docking.dye for docking in dockings)

    return {name: DyeDefinition.from_library(name, dye_dir) for name in names}


def create_dye_instances(dockings, definitions):
    """Create uniquely named and segmented dye instances in docking order."""
    counts = {}
    segids = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    instances = []

    if len(dockings) > len(segids):
        raise ValueError(f"At most {len(segids)} dye instances are supported")

    for i, docking in enumerate(dockings):
        counts[docking.dye] = counts.get(docking.dye, 0) + 1
        name = f"{docking.dye}_{counts[docking.dye]}"
        instances.append(DyeInstance(definition=definitions[docking.dye], residues=docking.residues, name=name, segid=segids[i]))

    return instances



