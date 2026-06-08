# schemas/__init__.py
"""Shared domain schemas for the DLens scientific workflow.

Currently the simulation-stage schemas (``sim_config`` / ``sim_output`` of the
ExperimentRun). Future stages (model design, training, analysis) add their
schemas here.
"""

from dlens.schemas._simulation import (
    CosmologyParams,
    SimConfig,
    SimModelConfig,
    SimOutput,
    SubstructureType,
)
from dlens.schemas._model_design import (
    ArchFamily,
    ArchitectureSpec,
    DatasetCharacteristics,
    TaskType,
    TrainingConfig,
    characteristics_from_sim_outputs,
)
from dlens.schemas._experiment import (
    ExperimentRun,
    ExperimentState,
    PlannerAction,
    PlannerDecision,
)

__all__ = [
    # simulation
    "SubstructureType",
    "SimModelConfig",
    "CosmologyParams",
    "SimConfig",
    "SimOutput",
    # model design
    "TaskType",
    "ArchFamily",
    "DatasetCharacteristics",
    "ArchitectureSpec",
    "TrainingConfig",
    "characteristics_from_sim_outputs",
    # experiment state / planner (WIP)
    "PlannerAction",
    "PlannerDecision",
    "ExperimentRun",
    "ExperimentState",
]
