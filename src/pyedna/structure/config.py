"""Configuration models for structure-generation workflows."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass(frozen=True)
class DNAConfig:
    """Define whether DNA is generated from a sequence or copied from a library."""

    source: str
    name: str
    sequence: Optional[str] = None
    type: Optional[str] = None

    def __post_init__(self):
        if self.source not in {"generate", "library"}:
            raise ValueError("'dna.source' must be 'generate' or 'library'")
        if not self.name:
            raise ValueError("'dna.name' must be specified")
        if self.source == "generate" and (not self.sequence or not self.type):
            raise ValueError(
                "'dna.sequence' and 'dna.type' are required when dna.source='generate'"
            )

    def as_parameters(self):
        """Return the DNA mapping expected by MD and trajectory classes."""

        return {
            "dna_sequence": self.sequence,
            "dna_type": self.type,
            "dna_name": self.name,
        }


@dataclass(frozen=True)
class DyePlacement:
    """Describe one dye and the consecutive DNA residues it replaces."""

    name: str
    sites: list[int]


@dataclass(frozen=True)
class AttachmentConfig:
    """Describe one dye/linker attachment requested by structure TOML."""

    dye: str
    linker: str
    residue: int

    @property
    def name(self):
        return f"{self.dye}_{self.linker}"

    def as_placement(self):
        return DyePlacement(name=self.name, sites=[self.residue])


@dataclass(frozen=True)
class HaddockConfig:
    """Store HADDOCK model selection and optional parameter overrides."""

    engine: str = "haddock3"
    top_models: int = 5
    overrides: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.engine != "haddock3":
            raise ValueError("'docking.engine' must be 'haddock3'")
        if self.top_models < 1:
            raise ValueError("'docking.top_models' must be at least 1")
        if not isinstance(self.overrides, dict):
            raise ValueError("'docking.overrides' must contain TOML sections")
        if any(not isinstance(values, dict) for values in self.overrides.values()):
            raise ValueError("Each 'docking.overrides' section must contain key-value pairs")


@dataclass(frozen=True)
class AmberConfig:
    """Store tleap force-field, solvation, and output options."""

    model: int = 1
    output_name: Optional[str] = None
    dna_forcefield: str = "OL15"
    dye_forcefield: str = "gaff2"
    water_forcefield: str = "leaprc.water.tip3p"
    water_model: str = "TIP3P"
    solvent_padding: float = 20.0
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    neutralize: bool = True

    def __post_init__(self):
        if self.model < 1:
            raise ValueError("'amber.model' must be at least 1")


@dataclass(frozen=True)
class WorkflowConfig:
    """Store optional cross-stage workflow behavior."""

    prepare_amber: bool = False

    def __post_init__(self):
        if not isinstance(self.prepare_amber, bool):
            raise ValueError("'workflow.prepare_amber' must be true or false")


@dataclass(frozen=True)
class StructureConfig:
    """Store and validate all structure-generation and Amber settings."""

    name: str
    dna: DNAConfig
    dyes: list[DyePlacement]
    attachments: list[AttachmentConfig] = field(default_factory=list)
    haddock: HaddockConfig = field(default_factory=HaddockConfig)
    amber: AmberConfig = field(default_factory=AmberConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    def __post_init__(self):
        if not self.name:
            raise ValueError("'system.name' must be specified")
        if self.amber.model > self.haddock.top_models:
            raise ValueError("'amber.model' cannot exceed 'docking.top_models'")
        if self.attachments and self.dyes != [a.as_placement() for a in self.attachments]:
            raise ValueError("Do not mix legacy [[dyes]] with [[attachments]]")

        occupied = set()
        for dye in self.dyes:
            if not dye.name or not dye.sites:
                raise ValueError("Each dye requires a name and at least one site")
            sites = sorted(set(dye.sites))
            if sites != list(range(sites[0], sites[-1] + 1)):
                raise ValueError(f"{dye.name}: sites must be consecutive: {sites}")
            overlap = occupied.intersection(sites)
            if overlap:
                raise ValueError(
                    f"{dye.name}: DNA sites already assigned: {sorted(overlap)}"
                )
            occupied.update(sites)

    @classmethod
    def from_file(cls, path):
        """Load a complete structure configuration from a TOML file."""

        path = Path(path)
        with path.open("rb") as handle:
            data = tomllib.load(handle)

        system = data.get("system", data.get("structure"))
        dna = data.get("dna")
        if system is None:
            raise ValueError(f"{path}: missing [system] section")
        if dna is None:
            raise ValueError(f"{path}: missing [dna] section")

        try:
            legacy_dyes = data.get("dyes", [])
            attachments_data = data.get("attachments", [])
            if "components" in data:
                raise ValueError(f"{path}: use [[attachments]] instead of [[components]]")

            if legacy_dyes and attachments_data:
                raise ValueError(
                    f"{path}: do not mix [[attachments]] and [[dyes]]"
                )

            attachments = [
                AttachmentConfig(**entry)
                for entry in attachments_data
            ]

            dyes = (
                [attachment.as_placement() for attachment in attachments]
                if attachments else [DyePlacement(**dye) for dye in legacy_dyes]
            )
            docking = dict(data.get("haddock", {}))
            docking.update(data.get("docking", {}))
            workflow = dict(data.get("workflow", {}))

            amber = dict(data.get("amber", {}))
            forcefield = data.get("forcefield", {})
            if "components" in forcefield:
                raise ValueError(
                    f"{path}: use forcefield.attachments instead of forcefield.components"
                )
            if "dna" in forcefield:
                amber["dna_forcefield"] = forcefield["dna"]
            if "attachments" in forcefield:
                amber["dye_forcefield"] = forcefield["attachments"]
            if "water" in forcefield:
                amber["water_forcefield"] = forcefield["water"]

            return cls(
                name=system["name"],
                dna=DNAConfig(**dna),
                dyes=dyes,
                attachments=attachments,
                haddock=HaddockConfig(**docking),
                amber=AmberConfig(**amber),
                workflow=WorkflowConfig(**workflow),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: invalid configuration field: {exc}") from exc
