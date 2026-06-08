"""Experiment Planner is a WIP stub: verify the typed interface + wiring only.

These tests pin the *interface* (no-op decide, agent wiring, run() not implemented)
so the real planner can be filled in later without breaking the contract.
"""

from __future__ import annotations

import pytest

from dlens.agents import ExperimentPlanner
from dlens.schemas import ExperimentState, PlannerAction, PlannerDecision


def test_decide_is_noop_placeholder():
    planner = ExperimentPlanner()
    state = ExperimentState(hypothesis="Can a ResNet separate cdm vs vortex?")
    decision = planner.decide(state)
    assert isinstance(decision, PlannerDecision)
    assert decision.action == PlannerAction.STOP
    assert "WIP" in decision.rationale  # explicitly not real planning


def test_decide_honors_iteration_budget_guardrail():
    planner = ExperimentPlanner()
    state = ExperimentState(hypothesis="h", current_iteration=3, max_iterations=3)
    assert planner.decide(state).action == PlannerAction.REPORT


def test_agent_wiring():
    sim, design = object(), object()
    planner = ExperimentPlanner(simulation_agent=sim, model_design_agent=design)
    assert planner.agent_for(PlannerAction.SIMULATE) is sim
    assert planner.agent_for(PlannerAction.DESIGN_MODEL) is design
    # Unbuilt stages route to nothing yet.
    assert planner.agent_for(PlannerAction.TRAIN) is None
    assert planner.agent_for(PlannerAction.REPORT) is None


def test_run_is_not_implemented():
    import asyncio

    planner = ExperimentPlanner()
    with pytest.raises(NotImplementedError):
        asyncio.run(planner.run("some hypothesis"))
