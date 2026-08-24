"""Configuration models for Amber MD workflows."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


STAGE_GROUPS = ("minimize", "equilibrate", "production")
CLEANUP_LEVELS = ("minimal", "standard", "restart", "all")


@dataclass(frozen=True)
class SystemConfig:
    """Store the MD system name and prepared Amber inputs."""

    name: str
    prmtop: Optional[str] = None
    rst7: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("'system.name' must be specified")

    @property
    def prmtop_path(self):
        return self.prmtop or f"{self.name}.prmtop"

    @property
    def rst7_path(self):
        return self.rst7 or f"{self.name}.rst7"


@dataclass(frozen=True)
class WorkflowConfig:
    """Store user-facing MD workflow stages."""

    stages: list[str] = field(
        default_factory=lambda: ["minimize", "equilibrate", "production"]
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
class MinimizationConfig:
    """Store minimization step counts."""

    max_steps: int = 1000
    steepest_descent_steps: int = 500


@dataclass(frozen=True)
class EquilibrationConfig:
    """Store equilibration step counts and output intervals."""

    heating_steps: int = 10000
    npt_steps: int = 50000
    ntpr: int = 5000
    ntwx: int = 5000
    ntwr: int = 5000


@dataclass(frozen=True)
class ProductionConfig:
    """Store production MD length and output intervals."""

    steps: int = 1000000
    ntpr: int = 5000
    ntwx: int = 5000
    ntwr: int = 50000
    ntwf: int = 0


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
class RestraintForceConfig:
    """Store stage-specific restraint force constants."""

    min1: float = 500.0
    min2: float = 5.0
    eq1: float = 10.0
    eq2: float = 10.0
    production: float = 5.0


@dataclass(frozen=True)
class RestraintConfig:
    """Store Amber restraint settings independent from structure.toml."""

    mode: str = "none"
    start: int = 1
    end: Optional[int] = None
    mask: Optional[str] = None
    force: RestraintForceConfig = field(default_factory=RestraintForceConfig)

    def __post_init__(self):
        if self.mode not in {"terminal", "none"}:
            raise ValueError("'restraints.mode' must be 'terminal' or 'none'")
        if self.start < 1:
            raise ValueError("'restraints.start' must be at least 1")
        if self.end is not None and self.end < self.start:
            raise ValueError("'restraints.end' must be greater than or equal to start")
        if self.mode == "terminal" and (self.end is None or not self.mask):
            raise ValueError(
                "'restraints.end' and 'restraints.mask' are required for "
                "terminal restraints until restraint inference is implemented"
            )

    @classmethod
    def from_mapping(cls, data):
        data = dict(data)
        force = RestraintForceConfig(**data.pop("force", {}))
        return cls(force=force, **data)


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
    restraints: RestraintConfig = field(default_factory=RestraintConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_file(cls, path):
        """Load an MD configuration from a TOML file."""

        path = Path(path)
        data = _load_toml(path)

        system = data.get("system")
        if system is None:
            raise ValueError(f"{path}: missing [system] section")

        try:
            return cls(
                system=SystemConfig(**system),
                workflow=WorkflowConfig(**data.get("workflow", {})),
                simulation=SimulationConfig(**data.get("simulation", {})),
                minimization=MinimizationConfig(**data.get("minimization", {})),
                equilibration=EquilibrationConfig(**data.get("equilibration", {})),
                production=ProductionConfig(**data.get("production", {})),
                thermostat=ThermostatConfig(**data.get("thermostat", {})),
                barostat=BarostatConfig(**data.get("barostat", {})),
                restraints=RestraintConfig.from_mapping(data.get("restraints", {})),
                output=OutputConfig(**data.get("output", {})),
            )
        except TypeError as exc:
            raise ValueError(f"{path}: invalid configuration field: {exc}") from exc

    @property
    def traj_dt(self):
        return self.production.ntwx * self.simulation.timestep

    @property
    def total_time(self):
        return self.production.steps * self.simulation.timestep

    @property
    def traj_steps(self):
        return self.production.steps // self.production.ntwx


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
