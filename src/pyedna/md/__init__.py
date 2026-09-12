"""Run Amber molecular dynamics simulations."""

import importlib

from .config import (
    BarostatConfig,
    EquilibrationConfig,
    EquilibrationRestraintConfig,
    MDConfig,
    AmberPreparationConfig,
    MinimizationRestraintConfig,
    MinimizationConfig,
    OutputConfig,
    ProductionConfig,
    SimulationConfig,
    StageRestraintConfig,
    SystemConfig,
    ThermostatConfig,
    WorkflowConfig,
)

__all__ = [
    "AmberPreparationConfig",
    "AmberSetup",
    "BarostatConfig",
    "EquilibrationConfig",
    "EquilibrationRestraintConfig",
    "MDConfig",
    "MDSimulation",
    "MinimizationRestraintConfig",
    "MinimizationConfig",
    "OutputConfig",
    "ProductionConfig",
    "SimulationConfig",
    "StageRestraintConfig",
    "SystemConfig",
    "ThermostatConfig",
    "WorkflowConfig",
]


def __getattr__(name):
    if name == "AmberSetup":
        module = importlib.import_module("pyedna.md.preparation")
    elif name == "MDSimulation":
        module = importlib.import_module("pyedna.md.simulation")
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    value = getattr(module, name)
    globals()[name] = value
    return value
