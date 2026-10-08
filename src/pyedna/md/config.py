"""Configuration models for Amber MD workflows."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


STAGE_GROUPS = ("prepare", "minimize", "equilibrate", "production")
CLEANUP_LEVELS = ("minimal", "standard", "restart", "all")
RESTRAINT_TARGETS = ("none", "terminal", "structure", "custom")
AMBER_MD_ENGINES = ("auto", "pmemd", "pmemd.MPI", "pmemd.cuda", "sander")
MINIMIZATION_ENGINES = AMBER_MD_ENGINES
NPT_CLEANUP_LEVELS = ("previous", "all", "none")


@dataclass(frozen=True)
class SystemConfig:
    """Store the MD system name and selected finalized structures."""

    name: str
    structure_directory: str = "structures"
    structures: list[int] = field(default_factory=lambda: [1])
    prmtop: str | None = None
    rst7: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("'system.name' must be specified")
        if (self.prmtop is None) != (self.rst7 is None):
            raise ValueError(
                "'system.prmtop' and 'system.rst7' must be given together"
            )
        if not self.structures:
            raise ValueError("'system.structures' must contain at least one structure")
        invalid = [
            structure for structure in self.structures
            if not isinstance(structure, int) or structure < 1
        ]
        if invalid:
            raise ValueError(
                f"'system.structures' contains invalid structure numbers: {invalid}"
            )
        duplicates = sorted({
            structure for structure in self.structures
            if self.structures.count(structure) > 1
        })
        if duplicates:
            raise ValueError(
                f"'system.structures' contains duplicate structure numbers: {duplicates}"
            )

    def structure_path(self, structure):
        return Path(self.structure_directory) / f"{self.name}_{structure}.pdb"


@dataclass(frozen=True)
class WorkflowConfig:
    """Store user-facing MD workflow stages."""

    stages: list[str] = field(
        default_factory=lambda: ["prepare", "minimize", "equilibrate", "production"]
    )

    def __post_init__(self):
        invalid = [stage for stage in self.stages if stage not in STAGE_GROUPS]
        if invalid:
            raise ValueError(f"'workflow.stages' contains invalid stages: {invalid}")


@dataclass(frozen=True)
class SimulationConfig:
    """Store common simulation control parameters."""

    temperature: float = 300.0
    pressure: float = 1.0
    timestep: float = 0.002
    cutoff: float = 8.0
    initial_temperature: float = 0.0
    iwrap: int = 1
    ntb: int = 1
    ntc: int = 2
    ntf: int = 2
    ntp: int = 2
    ioutfm: int = 1


@dataclass(frozen=True)
class StageRestraintConfig:
    """Store restraint settings for one Amber MD stage."""

    target: str = "none"
    strength: Optional[float] = None

    def __post_init__(self):
        if self.target not in RESTRAINT_TARGETS:
            raise ValueError(f"'target' must be one of {RESTRAINT_TARGETS}")
        if self.target != "none" and self.strength is None:
            raise ValueError("'strength' is required when restraint target is not 'none'")


@dataclass(frozen=True)
class MinimizationRestraintConfig:
    """Store restraint settings for the two minimization substages."""

    stage1: StageRestraintConfig = field(default_factory=StageRestraintConfig)
    stage2: StageRestraintConfig = field(default_factory=StageRestraintConfig)

    @classmethod
    def from_mapping(cls, data):
        return cls(
            stage1=StageRestraintConfig(**data.get("stage1", {})),
            stage2=StageRestraintConfig(**data.get("stage2", {})),
        )


@dataclass(frozen=True)
class EquilibrationRestraintConfig:
    """Store restraint settings for the two equilibration substages."""

    stage1: StageRestraintConfig = field(default_factory=StageRestraintConfig)
    stage2: StageRestraintConfig = field(default_factory=StageRestraintConfig)

    @classmethod
    def from_mapping(cls, data):
        return cls(
            stage1=StageRestraintConfig(**data.get("stage1", {})),
            stage2=StageRestraintConfig(**data.get("stage2", {})),
        )


@dataclass(frozen=True)
class MinimizationConfig:
    """Store minimization step counts and restraint settings."""

    max_steps: int = 1000
    steepest_descent_steps: int = 500
    engine: str = "pmemd"
    cutoff: float | None = None
    ntmin: int = 1
    restraints: MinimizationRestraintConfig = field(
        default_factory=MinimizationRestraintConfig
    )

    def __post_init__(self):
        _validate_md_engine("minimization.engine", self.engine)

    @classmethod
    def from_mapping(cls, data):
        data = dict(data)
        restraints = MinimizationRestraintConfig.from_mapping(
            data.pop("restraints", {})
        )
        return cls(restraints=restraints, **data)


@dataclass(frozen=True)
class EquilibrationConfig:
    """Store equilibration step counts, output intervals, and restraints."""

    engine: str = "auto"
    cutoff: float | None = None
    heating_cutoff: float | None = None
    npt_cutoff: float | None = None
    heating_steps: int = 10000
    npt_steps: int = 50000
    npt_chunks: int = 1
    npt_cleanup: str = "previous"
    ntpr: int = 5000
    ntwx: int = 5000
    ntwr: int = 5000
    restraints: EquilibrationRestraintConfig = field(
        default_factory=EquilibrationRestraintConfig
    )

    def __post_init__(self):
        _validate_md_engine("equilibration.engine", self.engine)
        if self.npt_chunks < 1:
            raise ValueError("'equilibration.npt_chunks' must be at least 1")
        if self.npt_steps % self.npt_chunks != 0:
            raise ValueError(
                "'equilibration.npt_steps' must be divisible by "
                "'equilibration.npt_chunks'"
            )
        if self.npt_cleanup not in NPT_CLEANUP_LEVELS:
            raise ValueError(
                f"'equilibration.npt_cleanup' must be one of {NPT_CLEANUP_LEVELS}"
            )

    @classmethod
    def from_mapping(cls, data):
        data = dict(data)
        restraints = EquilibrationRestraintConfig.from_mapping(
            data.pop("restraints", {})
        )
        return cls(restraints=restraints, **data)


@dataclass(frozen=True)
class ProductionConfig:
    """Store production MD length, output intervals, and restraints."""

    engine: str = "auto"
    cutoff: float | None = None
    steps: int = 1000000
    log_interval: int = 5000
    trajectory_interval: int = 5000
    restart_interval: int = 50000
    force_interval: int = 0
    restraints: StageRestraintConfig = field(default_factory=StageRestraintConfig)

    def __post_init__(self):
        _validate_md_engine("production.engine", self.engine)

    @classmethod
    def from_mapping(cls, data):
        data = dict(data)
        restraints = StageRestraintConfig(**data.pop("restraints", {}))
        _rename_legacy_field(data, "ntpr", "log_interval")
        _rename_legacy_field(data, "ntwx", "trajectory_interval")
        _rename_legacy_field(data, "ntwr", "restart_interval")
        _rename_legacy_field(data, "ntwf", "force_interval")
        return cls(restraints=restraints, **data)


def _validate_md_engine(field_name, engine):
    if engine not in AMBER_MD_ENGINES:
        raise ValueError(f"'{field_name}' must be one of {AMBER_MD_ENGINES}")


@dataclass(frozen=True)
class ThermostatConfig:
    """Store thermostat settings."""

    type: str = "langevin"
    gamma: float = 5.0
    seed: int = -1

    @property
    def amber_ntt(self):
        if self.type != "langevin":
            raise ValueError("'thermostat.type' must currently be 'langevin'")
        return 3


@dataclass(frozen=True)
class BarostatConfig:
    """Store barostat settings."""

    tau: float = 2.0


@dataclass(frozen=True)
class AmberPreparationConfig:
    """Store tleap force-field and solvation settings owned by MD."""

    dna_forcefield: str = "OL15"
    dye_forcefield: str = "gaff2"
    water_forcefield: str = "tip3p"
    water_model: str | None = None
    solvent_padding: float = 20.0
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    neutralize: bool = True

    @property
    def water_box(self):
        """Return the tleap solvent box selected by the water force field."""

        water = self.water_model or self.water_forcefield
        water = str(water).removeprefix("leaprc.water.").lower()
        boxes = {
            "tip3p": "TIP3PBOX",
        }
        if water not in boxes:
            raise ValueError(f"Unsupported water force field: {self.water_forcefield!r}")
        return boxes[water]


@dataclass(frozen=True)
class OutputConfig:
    """Store runtime output and cleanup behavior."""

    directory: str = "md"
    cleanup: str = "standard"

    def __post_init__(self):
        if self.cleanup not in CLEANUP_LEVELS:
            raise ValueError(f"'output.cleanup' must be one of {CLEANUP_LEVELS}")


@dataclass(frozen=True)
class MDConfig:
    """Store and validate all Amber MD settings."""

    system: SystemConfig
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    minimization: MinimizationConfig = field(default_factory=MinimizationConfig)
    equilibration: EquilibrationConfig = field(default_factory=EquilibrationConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    thermostat: ThermostatConfig = field(default_factory=ThermostatConfig)
    barostat: BarostatConfig = field(default_factory=BarostatConfig)
    amber: AmberPreparationConfig = field(default_factory=AmberPreparationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_file(cls, path):
        """Load an MD configuration from a TOML file."""

        path = Path(path)
        data = _load_toml(path)

        system = data.get("system")
        if system is None:
            raise ValueError(f"{path}: missing [system] section")
        if "restraints" in data:
            raise ValueError(
                f"{path}: move [restraints.*] sections under their stages, "
                "for example [minimization.restraints.stage1]"
            )

        try:
            system = dict(system)
            if "models" in system:
                if "structures" in system:
                    raise ValueError(
                        f"{path}: use either system.structures or legacy "
                        "system.models, not both"
                    )
                system["structures"] = system.pop("models")

            return cls(
                system=SystemConfig(**system),
                workflow=WorkflowConfig(**data.get("workflow", {})),
                simulation=SimulationConfig(**data.get("simulation", {})),
                minimization=MinimizationConfig.from_mapping(
                    data.get("minimization", {})
                ),
                equilibration=EquilibrationConfig.from_mapping(
                    data.get("equilibration", {})
                ),
                production=ProductionConfig.from_mapping(data.get("production", {})),
                thermostat=ThermostatConfig(**data.get("thermostat", {})),
                barostat=BarostatConfig(**data.get("barostat", {})),
                amber=AmberPreparationConfig(**data.get("amber", {})),
                output=OutputConfig(**data.get("output", {})),
            )
        except TypeError as exc:
            raise ValueError(f"{path}: invalid configuration field: {exc}") from exc

    @property
    def traj_dt(self):
        return self.production.trajectory_interval * self.simulation.timestep

    @property
    def total_time(self):
        return self.production.steps * self.simulation.timestep

    @property
    def traj_steps(self):
        return self.production.steps // self.production.trajectory_interval


def _load_toml(path):
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)

    data = {}
    current = data
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = data
            for part in line.strip("[]").split("."):
                current = current.setdefault(part, {})
            continue

        key, value = line.split("=", 1)
        current[key.strip()] = _parse_toml_value(value.strip())

    return data


def _parse_toml_value(value):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip('"').strip("'")


def _rename_legacy_field(data, old_name, new_name):
    if old_name not in data:
        return
    if new_name in data:
        raise ValueError(
            f"Use either '{new_name}' or legacy '{old_name}', not both"
        )
    data[new_name] = data.pop(old_name)
