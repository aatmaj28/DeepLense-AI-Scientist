# schemas/_experiment.py
"""Experiment-state schemas — the typed interface for the Experiment Planner.

These mirror the ``ExperimentState`` / ``ExperimentRun`` / ``PlannerDecision`` data
model defined in docs/WORKFLOW_DESIGN.md (canonical). Fields for agents not yet
implemented (training, inference, analysis) are optional placeholders.

NOTE: the Experiment Planner that *consumes/produces* these is a WIP stub — see
docs/EXPERIMENT_PLANNER.md. No real planning logic is implemented.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
from dlens.schemas._simulation import SimConfig, SimOutput


class PlannerAction(str, Enum):
    """Where the planner sends control next (mirrors WORKFLOW_DESIGN decision space)."""

    SIMULATE = "simulate"          # loop back to agent 1 (Simulation)
    DESIGN_MODEL = "design_model"  # loop back to agent 2 (Model Design)
    TRAIN = "train"                # loop back to agent 3 (Training)
    REPORT = "report"             # proceed to agent 7 (Report)
    STOP = "stop"                  # terminate without a full report


class PlannerDecision(BaseModel):
    """A planner decision: which stage to go to next, and why."""

    action: PlannerAction = Field(description="Next stage to route to.")
    rationale: str = Field(description="Why this decision was made.")
    updated_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameter overrides for the next stage."
    )


class ExperimentRun(BaseModel):
    """One iteration of the experiment loop (accumulated in ExperimentState.runs)."""

    iteration: int = Field(description="0-based iteration index.")
    sim_config: Optional[SimConfig] = None
    sim_output: Optional[SimOutput] = None
    architecture: Optional[ArchitectureSpec] = None
    training_config: Optional[TrainingConfig] = None
    # Populated by future agents (training/inference/analysis) — see WORKFLOW_DESIGN.
    training_logs: Optional[dict[str, Any]] = None
    predictions: Optional[dict[str, Any]] = None
    analysis: Optional[dict[str, Any]] = None
    planner_decision: Optional[PlannerDecision] = None


class ExperimentState(BaseModel):
    """Shared state threaded through the pipeline, accumulating run history."""

    hypothesis: str = Field(description="Original problem statement / hypothesis.")
    current_iteration: int = Field(default=0, description="Current iteration index.")
    max_iterations: int = Field(default=3, ge=1, description="Iteration budget (guardrail).")
    runs: list[ExperimentRun] = Field(default_factory=list, description="Full run history.")
