from dataclasses import dataclass
from typing import Optional

from . import fileproc as fp


@dataclass
class DockingSpec:
    """Pair a dye name with the consecutive DNA residues it replaces."""
    dye: str
    residues: list[int]


@dataclass
class StructureConfig:
    """Store and validate structure-generation inputs."""
    dna_source: str
    dna_name: str
    structure_name: str
    dockings: list[DockingSpec]
    dna_sequence: Optional[str] = None
    dna_type: Optional[str] = None
    top_models: int = 5

    def __post_init__(self):
        """Validate DNA settings and non-overlapping docking ranges."""
        if self.dna_source not in {"generate", "library"}:
            raise ValueError(f"Unknown dna_source {self.dna_source!r}; "
                             "expected 'generate' or 'library'")

        if not self.dna_name:
            raise ValueError("'dna_name' must be specified")

        if not self.structure_name:
            raise ValueError("'structure_name' must be specified")

        if self.dna_source == "generate":
            if not self.dna_sequence:
                raise ValueError("'dna_sequence' must be specified when dna_source='generate'")
            if not self.dna_type:
                raise ValueError("'dna_type' must be specified when dna_source='generate'")

        occupied = set()

        for docking in self.dockings:
            if not docking.residues:
                raise ValueError(f"{docking.dye}: at least one DNA residue must be specified")

            residues = sorted(set(docking.residues))

            if residues != list(range(residues[0], residues[-1] + 1)):
                raise ValueError(f"{docking.dye}: residues must be consecutive: {residues}")

            overlap = occupied.intersection(residues)
            if overlap:
                raise ValueError(f"{docking.dye}: DNA residues already assigned: {sorted(overlap)}")

            occupied.update(residues)

        if self.top_models < 1:
            raise ValueError("'top_models' must be at least 1")

    @classmethod
    def from_file(cls, path):
        """Build a validated configuration from a PyeDNA parameter file."""
        params = fp.readParams(path)

        dna_source = params.get("dna_source", "generate")
        dna_name = params.get("dna_name")
        structure_name = params.get("structure_name")
        dna_sequence = params.get("dna_sequence")
        dna_type = params.get("dna_type")

        dyes = params.get("dyes", [])
        dye_sites = params.get("dye_sites", [])
        top_models = params.get("top_models", 5)

        if len(dyes) != len(dye_sites):
            raise ValueError(
                f"'dyes' and 'dye_sites' must have the same length "
                f"({len(dyes)} != {len(dye_sites)})"
            )

        dockings = [
            DockingSpec(dye=dye, residues=list(residues))
            for dye, residues in zip(dyes, dye_sites)
        ]

        return cls(dna_source=dna_source, dna_name=dna_name, structure_name=structure_name,
                   dockings=dockings, dna_sequence=dna_sequence, dna_type=dna_type,
                   top_models=top_models)
