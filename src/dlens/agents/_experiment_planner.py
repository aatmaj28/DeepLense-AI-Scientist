# agents/_experiment_planner.py
"""ExperimentPlanner — agent #6 of the DLens pipeline.

=============================================================================
WIP / FOR DISCUSSION — SKELETON ONLY.
Real planning logic (reasoning over experiment history to decide what to change)
is intentionally NOT implemented. See docs/EXPERIMENT_PLANNER.md for the proposed
design (ReAct loop vs agentic tree search) and docs/WORKFLOW_DESIGN.md for the
canonical role, I/O, decision space, and termination conditions.
=============================================================================

This stub fixes the *typed interface* and *wiring* so the planner can be filled in
collaboratively later:
  * it holds references to the subagents it would orchestrate (Simulation,
    Model Design — the ones that exist today), and
  * ``decide()`` is a no-op placeholder returning a STOP/REPORT decision.

The real planner is expected to be a ReAct-style agent backed by a larger model
(via API), per WORKFLOW_DESIGN.md.
"""

from __future__ import annotations

from typing import Any, Optional

from dlens.schemas._experiment import ExperimentState, PlannerAction, PlannerDecision

# Mapping from a planner action to the pipeline agent that handles it (those not
# yet implemented are None placeholders).
_ACTION_AGENT_ATTR: dict[PlannerAction, str | None] = {
    PlannerAction.SIMULATE: "simulation_agent",
    PlannerAction.DESIGN_MODEL: "model_design_agent",
    PlannerAction.TRAIN: None,   # Training Agent (agent #3) — not implemented yet
    PlannerAction.REPORT: None,  # Report Agent (agent #7) — not implemented yet
    PlannerAction.STOP: None,
}


class ExperimentPlanner:
    """WIP skeleton for the experiment-planning loop (no real planning logic)."""

    def __init__(
        self,
        *,
        simulation_agent: Any | None = None,
        model_design_agent: Any | None = None,
        model: Any | None = None,
    ) -> None:
        # Subagents this planner would orchestrate (loop back to).
        self.simulation_agent = simulation_agent
        self.model_design_agent = model_design_agent
        # Reserved for the future ReAct reasoning model (a larger API model).
        self._model = model

    def decide(self, state: ExperimentState) -> PlannerDecision:
        """NO-OP placeholder. Does NOT reason about results.

        The real implementation will synthesize the hypothesis, full run history,
        and failure-mode analysis into a routing decision (see EXPERIMENT_PLANNER.md).
        For now it only honors the iteration-budget guardrail and otherwise stops.
        """
        if state.current_iteration >= state.max_iterations:
            return PlannerDecision(
                action=PlannerAction.REPORT,
                rationale="WIP stub: iteration budget reached; would proceed to Report.",
            )
        return PlannerDecision(
            action=PlannerAction.STOP,
            rationale="WIP stub: planning logic not implemented (see EXPERIMENT_PLANNER.md).",
        )

    def agent_for(self, action: PlannerAction) -> Optional[Any]:
        """Return the wired subagent for an action (None if not implemented yet)."""
        attr = _ACTION_AGENT_ATTR.get(action)
        return getattr(self, attr) if attr else None

    async def run(self, hypothesis: str) -> ExperimentState:  # pragma: no cover - WIP
        """The experiment loop. NOT IMPLEMENTED — design only.

        Will: initialize ExperimentState(hypothesis), then loop {decide -> dispatch
        to agent_for(decision.action) -> record ExperimentRun} until REPORT/STOP or
        the iteration budget is hit. Left unimplemented on purpose so the planning
        strategy can be decided collaboratively (see EXPERIMENT_PLANNER.md).
        """
        raise NotImplementedError(
            "ExperimentPlanner.run is a WIP stub; see docs/EXPERIMENT_PLANNER.md."
        )
