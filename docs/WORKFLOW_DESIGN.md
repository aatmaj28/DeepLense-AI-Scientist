# DLens Workflow Design
**Last updated: 2026-06-05**

## Overview

DLens orchestrates an end-to-end scientific workflow for gravitational lensing research. A human provides a problem statement or hypothesis, and the system autonomously runs simulation, modeling, evaluation, and experimentation loops, producing a final report with findings.

## Pipeline

```
                    ┌─────────────────────────┐
                    │   Human Input            │
                    │   Problem / Hypothesis   │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  1. Simulation Agent     │
                    │                         │
                    │  - Configures params    │
                    │  - Generates code       │
                    │  - Submits job          │
                    │  - Polls for completion │
                    └───────────┬─────────────┘
                                │ simulated dataset
                                ▼
                    ┌─────────────────────────┐
                    │  2. Model Design Agent   │
                    │                         │
                    │  - Designs architecture │
                    │  - Selects loss/optim   │
                    │  - Writes training code │
                    └───────────┬─────────────┘
                                │ model + training config
                                ▼
                    ┌─────────────────────────┐
                    │  3. Training Agent       │
                    │                         │
                    │  - Runs training        │
                    │  - Monitors metrics     │
                    │  - Handles checkpoints  │
                    └───────────┬─────────────┘
                                │ trained model
                                ▼
                    ┌─────────────────────────┐
                    │  4. Inference Agent      │
                    │                         │
                    │  - Runs on validation   │
                    │  - Collects predictions │
                    │  - Stores outputs       │
                    └───────────┬─────────────┘
                                │ predictions
                                ▼
                    ┌─────────────────────────┐
                    │  5. Analysis Agent       │
                    │                         │
                    │  - Computes metrics     │
                    │  - Error analysis       │
                    │  - Identifies failure   │
                    │    modes & patterns     │
                    └───────────┬─────────────┘
                                │ structured analysis
                                ▼
              ┌───────────────────────────────────────┐
              │  6. Experiment Planner                 │
              │                                       │
              │  Inputs:                              │
              │  - Original hypothesis                │
              │  - Current + past experiment results  │
              │  - Failure mode analysis              │
              │                                       │
              │  Decides:                             │
              │  - Adjust simulation params? → go to 1│
              │  - Change architecture?     → go to 2 │
              │  - Retrain with new config? → go to 3 │
              │  - Hypothesis validated?    → go to 7 │
              │  - Max iterations reached?  → go to 7 │
              └──────────┬────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │ loop                │ done
              ▼                     ▼
         back to 1/2/3    ┌─────────────────────────┐
                          │  7. Report Agent         │
                          │                         │
                          │  - Summarizes findings  │
                          │  - Compares experiments │
                          │  - Documents decisions  │
                          │  - Generates report     │
                          └─────────────────────────┘
```

## Agents

### 1. Simulation Agent

Wraps DeepLenseSim to generate gravitational lensing datasets.

**Inputs:** Simulation parameters (lens model, source properties, noise profiles, sample size)
**Outputs:** Simulated dataset path, generation metadata
**Tools:** DeepLenseSim code generation, job submission (SLURM/server), job status polling

The agent translates high-level requirements (e.g., "generate 10k strong lensing images with SIE lens model and varying axis ratios") into DeepLenseSim configuration and executable code, submits the job, and waits for completion.

### 2. Model Design Agent

Designs the neural network architecture for the downstream task (classification or regression).

**Inputs:** Task description, dataset characteristics, analysis from prior experiments (if looping)
**Outputs:** Architecture specification, training configuration (loss function, optimizer, schedule)

This is where architectural decisions are made. The agent should reason about the problem structure, not just default to standard architectures. Given domain-appropriate feedback from the analysis agent, the design should evolve across experiment iterations, potentially incorporating physics-informed components (e.g., equivariant layers, physics-based loss terms, or hybrid architectures like LensPINN) rather than converging on generic architectures.

### 3. Training Agent

Executes model training and monitors convergence.

**Inputs:** Model code, training config, dataset path
**Outputs:** Trained model checkpoint, training logs (loss curves, metric history)
**Tools:** Job submission, log monitoring, checkpoint management

### 4. Inference Agent

Runs the trained model on held-out validation data.

**Inputs:** Model checkpoint, validation dataset path
**Outputs:** Predictions, confidence scores, per-sample results

### 5. Analysis Agent

Evaluates model performance and identifies failure patterns.

**Inputs:** Predictions, ground truth, dataset metadata
**Outputs:** Structured analysis report

The quality of this agent's output directly determines how well the experiment planner can reason about next steps. The analysis should go beyond aggregate metrics and surface:
- **Failure modes:** Which subpopulations does the model fail on? (e.g., high-eccentricity substructures, low SNR images)
- **Error correlations:** Do residuals correlate with physical parameters? (e.g., Einstein radius, source redshift)
- **Comparison with prior runs:** What improved, what regressed, and why?
- **Physical consistency:** Are predictions physically plausible?

### 6. Experiment Planner

The core reasoning agent. Decides what to change and why based on experiment history.

**Inputs:** Original hypothesis, full experiment history (all past runs, analyses, decisions)
**Outputs:** Decision (which agent to loop back to, or proceed to report), rationale, updated parameters

**Decision space:**
- **Loop to Simulation (1):** Data is insufficient, need different parameter coverage, more samples, or different noise profiles
- **Loop to Model Design (2):** Architecture is the bottleneck, need structural changes (e.g., add physics priors, change backbone, adjust capacity)
- **Loop to Training (3):** Architecture is fine but training failed to converge, need different hyperparameters
- **Proceed to Report (7):** Hypothesis is validated/refuted with sufficient evidence, or max experiment budget reached

**Termination conditions:**
- Target metric threshold achieved
- Hypothesis conclusively validated or refuted
- Maximum iteration count reached
- Diminishing returns detected across successive runs

This agent likely benefits from a larger model (via API) using ReAct-style reasoning, since it needs to synthesize information across multiple experiment runs and make nuanced scientific judgments.

### 7. Report Agent

Generates a structured report summarizing the full experimental campaign.

**Inputs:** Complete experiment history, all analyses, planner decisions
**Outputs:** Final report document

**Report contents:**
- Original hypothesis and problem statement
- Experimental methodology (agent pipeline, simulation setup)
- Results per experiment iteration (architecture, metrics, key findings)
- Evolution of approach across iterations (why changes were made)
- Final conclusions and recommendations

## Inter-Agent Data Flow

### Experiment State

A shared experiment state object is passed through the pipeline and accumulates history across loop iterations:

```
ExperimentState:
  hypothesis: str                    # Original problem statement
  current_iteration: int
  max_iterations: int
  runs: List[ExperimentRun]          # Full history

ExperimentRun:
  iteration: int
  sim_config: SimConfig              # What was simulated
  sim_output: SimOutput              # Dataset metadata
  architecture: ArchitectureSpec     # Model design
  training_config: TrainingConfig    # Hyperparameters
  training_logs: TrainingLogs        # Loss curves, etc.
  predictions: PredictionOutput      # Raw inference results
  analysis: AnalysisReport           # Metrics + failure modes
  planner_decision: PlannerDecision  # What to change and why
```

### Data Dependencies

```
Agent              Reads                         Writes
─────────────────────────────────────────────────────────────
Simulation         sim_config                    sim_output
Model Design       task desc, past analyses      architecture, training_config
Training           architecture, training_config training_logs, checkpoint
Inference          checkpoint, val dataset        predictions
Analysis           predictions, ground truth     analysis report
Exp. Planner       full experiment history       decision + updated params
Report             full experiment history       final report
```

## Implementation Notes

- **Agent types:** Simulation, Training, Inference agents are tool-heavy and can use smaller local models. Model Design and Analysis need moderate reasoning. Experiment Planner needs the strongest reasoning capability.
- **Experiment state** will need a persistent store beyond `InMemorySessionStore` since runs may span hours or days on a cluster.
- **The simulation-to-training handoff** is asynchronous. The simulation agent submits a job and the pipeline waits for completion before proceeding.
- **Guardrails:** The planner should have a budget (max iterations, max compute time) to prevent unbounded loops.
