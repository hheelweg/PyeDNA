from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DyeDefinition:
    name: str
    directory: Path
    mol2: Path
    attach: Path

    @classmethod
    def from_library(cls, name, dye_dir):
        directory = Path(dye_dir) / name
        mol2 = directory / f"{name}.mol2"
        attach = directory / f"{name}.attach"

        if not directory.is_dir():
            raise FileNotFoundError(f"Dye directory not found: {directory}")

        if not mol2.exists():
            raise FileNotFoundError(f"Dye MOL2 file not found: {mol2}")

        if not attach.exists():
            raise FileNotFoundError(f"Dye attachment file not found: {attach}")

        return cls(
            name=name,
            directory=directory,
            mol2=mol2,
            attach=attach,
        )

def load_dye_definitions(dockings, dye_dir):
    names = dict.fromkeys(docking.dye for docking in dockings)

    return {name: DyeDefinition.from_library(name, dye_dir) for name in names}