from dataclasses import dataclass
from typing import Optional

from . import fileproc as fp


@dataclass
class DockingSpec:
    dye: str
    residues: list[int]



@dataclass
class StructureConfig:
    dna_source: str
    dna_name: str
    structure_name: str
    dockings: list[DockingSpec]
    dna_sequence: Optional[str] = None
    dna_type: Optional[str] = None

    @classmethod
    def from_file(cls, path):
        params = fp.readParams(path)

        dna_source = params.get("dna_source", "generate")
        dna_name = params.get("dna_name")
        structure_name = params.get("structure_name")
        dna_sequence = params.get("dna_sequence")
        dna_type = params.get("dna_type")

        dyes = params.get("dyes", [])
        dye_sites = params.get("dye_sites", [])

        if len(dyes) != len(dye_sites):
            raise ValueError(
                f"'dyes' and 'dye_sites' must have the same length "
                f"({len(dyes)} != {len(dye_sites)})"
            )

        dockings = [
            DockingSpec(dye=dye, residues=list(residues))
            for dye, residues in zip(dyes, dye_sites)
        ]

        return cls(
            dna_source=dna_source,
            dna_name=dna_name,
            structure_name=structure_name,
            dockings=dockings,
            dna_sequence=dna_sequence,
            dna_type=dna_type,
        )