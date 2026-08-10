from dataclasses import dataclass
from pathlib import Path
from . import fileproc as fp

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
    frcmods: list[Path]
    attach: Path

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
        frcmods = [directory / filename for filename in params.get("frcmods", [])]

        if not frcmods:
            raise ValueError(f"{params_file}: no frcmods specified")

        missing = [str(path) for path in frcmods if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing frcmod files: {missing}")

        return cls(name=name, directory=directory, mol2=mol2,
                frcmods=frcmods, attach=attach)


    def read_attachment(self):
        data = {}

        for line in self.attach.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            fields = line.split()
            if fields[0] not in {"5'", "3'"}:
                continue

            end, resname, resid, atom = fields
            data[end] = AttachmentAtom(resname=resname, resid=int(resid), atom=atom)

        if set(data) != {"5'", "3'"}:
            raise ValueError(f"{self.attach}: must define exactly 5' and 3'")

        return data


    def read_amber_mapping(self):
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


@dataclass
class DyeInstance:
    definition: DyeDefinition
    residues: list[int]
    name: str
    segid: str
    

def load_dye_definitions(dockings, dye_dir):
    names = dict.fromkeys(docking.dye for docking in dockings)

    return {name: DyeDefinition.from_library(name, dye_dir) for name in names}


def create_dye_instances(dockings, definitions):
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




