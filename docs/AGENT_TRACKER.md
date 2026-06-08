# Agent Tracker
**Last updated: 2026-06-06**

Registry of agents in the DLens pipeline. Pipeline topology, roles, and I/O are
canonical in [WORKFLOW_DESIGN.md](./WORKFLOW_DESIGN.md).

## Agents Overview

| Agent | Path | Status |
|-------|------|--------|
| DataSimulationAgent (#1) | `src/dlens/agents/_data_simulation.py` | ✅ Implemented |
| ModelDesignAgent (#2) | `src/dlens/agents/_model_design.py` | ✅ Implemented |
| Training Agent (#3) | — | ⬜ Planned |
| Inference Agent (#4) | — | ⬜ Planned |
| Analysis Agent (#5) | — | ⬜ Planned |
| Experiment Planner (#6) | `src/dlens/agents/_experiment_planner.py` | 🟡 WIP stub (design + interface only) |
| Report Agent (#7) | — | ⬜ Planned |

### Notes

- **DataSimulationAgent** — `DLensConversationalAgent` wrapping DeepLenseSim
  (API-wrapping, not code-gen). Two human-in-the-loop gates: typed
  `SimClarification`, and plan approval via `run_simulation`
  (`requires_approval=True` → `DeferredToolRequests`). Real `DeepLensBackend` +
  deterministic `MockBackend` fallback (offline). Schemas: `SimConfig`/`SimOutput`.
- **ModelDesignAgent** — stateless `DLensBaseAgent`, a thin wrapper around the
  deterministic `recommend_architecture` tool; emits `ArchitectureSpec` +
  `TrainingConfig`. Bridge: `characteristics_from_sim_outputs()`.
- **Experiment Planner** — WIP skeleton (typed interface + no-op `decide()`,
  `run()` not implemented). Design proposal in
  [EXPERIMENT_PLANNER.md](./EXPERIMENT_PLANNER.md).

## Planned Agents

Training, Inference, Analysis, and Report agents (#3–#5, #7) are defined in
WORKFLOW_DESIGN.md and not yet implemented. The framework supports:

- **Stateless agents** (`DLensBaseAgent`) — single-turn tasks and sub-agents.
- **Conversational agents** (`DLensConversationalAgent`) — multi-turn with session memory.
- **E2E orchestration** (`AbstractBaseAgent`) — composing multi-agent workflows.
