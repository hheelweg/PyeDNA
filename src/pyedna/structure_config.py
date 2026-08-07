from dataclasses import dataclass

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
    dna_sequence: str | None = None
    dna_type: str | None = None

    @classmethod
    def from_file(cls, path):
        params = fp.readParams(path)

        dna_source = params.get("dna_source", "generate")
        dna_name = params.get("dna_name")
        structure_name = params.get("structure_name")
        dna_sequence = params.get("dna_sequence")
        dna_type = params.get("dna_type")

        dyes = params.get("dyes", [])
        dye_residues = params.get("dye_residues", [])

        if len(dyes) != len(dye_residues):
            raise ValueError(
                f"'dyes' and 'dye_residues' must have the same length "
                f"({len(dyes)} != {len(dye_residues)})"
            )

        dockings = [
            DockingSpec(dye=dye, residues=list(residues))
            for dye, residues in zip(dyes, dye_residues)
        ]

        return cls(
            dna_source=dna_source,
            dna_name=dna_name,
            structure_name=structure_name,
            dockings=dockings,
            dna_sequence=dna_sequence,
            dna_type=dna_type,
        )