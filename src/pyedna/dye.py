from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DyeDefinition:
    name: str
    directory: Path
    mol2: Path
    frcmod: Path
    attach: Path

    @classmethod
    def from_library(cls, name, dye_dir):
        directory = Path(dye_dir) / name
        mol2 = directory / f"{name}.mol2"
        attach = directory / f"{name}.attach"
        frcmod = directory / f"{name}.frcmod"

        if not directory.is_dir():
            raise FileNotFoundError(f"Dye directory not found: {directory}")

        if not mol2.exists():
            raise FileNotFoundError(f"Dye MOL2 file not found: {mol2}")

        if not attach.exists():
            raise FileNotFoundError(f"Dye attachment file not found: {attach}")

        if not frcmod.exists():
            raise FileNotFoundError(f"Dye frcmod file not found: {frcmod}")

        return cls(name=name,
                   directory=directory,
                   mol2=mol2,
                   frcmod=frcmod,
                   attach=attach,)


    def read_attachment(self):
        data = {}

        for line in self.attach.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            end, resname, resid, atom = line.split()

            if end not in {"5'", "3'"}:
                raise ValueError(f"{self.attach}: unknown end {end!r}")

            data[end] = AttachmentAtom(resname=resname, resid=int(resid), atom=atom)

        if set(data) != {"5'", "3'"}:
            raise ValueError(f"{self.attach}: must define exactly 5' and 3'")

        return data

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


@dataclass(frozen=True)
class AttachmentAtom:
    resname: str
    resid: int
    atom: str

