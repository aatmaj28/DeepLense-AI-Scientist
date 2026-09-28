# Groq-planner seed reference: cnn_medium_s4 vs resnet34, 5 seeds

**Date: 2026-09-28** · Runner: `scripts/run_multi_seed.py` (Experiment A only) · Records:
`docs/groq_seed_reference/`

> **SEPARATE EXPERIMENT, FOR REFERENCE ONLY. NOT FOR PUBLICATION.**
> This uses a **different planner** (`openai/gpt-oss-120b` served by Groq) on a **newly
> generated dataset** (seeded 2026-09-28), on different hardware. It is **not comparable**
> to the recorded gpt-5.2 results in `docs/PAPER_RUN_RESULTS.md`,
> `docs/MULTI_SEED_RESULTS.md` or `docs/BUDGET_FAIRNESS.md`, and no comparison with them
> is made here.

## What differs from the recorded runs

| | Recorded paper runs | This experiment |
|---|---|---|
| Planner LLM | gpt-5.2 (OpenAI) | `openai/gpt-oss-120b` via Groq's OpenAI-compatible API (free tier) |
| Dataset | Jul 20 generation, **unseeded**, not reproducible | seeded 2026-09-28 generation, SHA-256 manifest (`groq_seed_reference/dataset/`) |
| Test set | Jul 27, seeds 777/778/779 | seeds 2001/2002/2003, 0 hash overlap with the pool |
| Training subset | `train_small`, 1,800 images, selection rule not recorded | `train_small` = the first 600 files per class of the seeded split |
| Hardware | Apple M2 Pro | Apple M5, 16 GB |
| Code | various commits | `bc40d0e`, branch `exp/groq-seed-reference` |

## Protocol

It is the same configuration as the paper run, with the same tuning settings as
`run_multi_seed.py`:

- **Arms:** `cnn_medium_s4` (cnn, depths [2,3,3,4], widths [32,64,128,256], 2.539 M) and
  `resnet34` (21.280 M).
- **Data:** 1,800 training / 1,800 validation images, 150×150, 3 classes.
- **Training settings:** adamw, lr 3e-4, batch 32, weight decay 1e-4, cosine schedule.
- **Tuning:** 25 epochs per iteration, at most 4 ReAct tuning iterations.
- **Selection:** the best-validation iteration is reported.
- **Test set:** evaluated **exactly once per model** after tuning. The manifest records
  `test_evaluations: 10`.
- **Seeds:** 0–4 for both arms, run one after another in a single process.

The effective model was printed at the top of every run's log and recorded in each run's
`config.json`: `llm: openai/gpt-oss-120b`, `llm_base_url: https://api.groq.com/openai/v1/`.

```bash
DLENS_LLM=groq:openai/gpt-oss-120b uv run python scripts/run_multi_seed.py \
  --data-root ~/GSoC/deeplense_data/seeded_2026-09-28/model1 --train-dir train_small \
  --val-dir val --test-root ~/GSoC/deeplense_data/seeded_2026-09-28/model1_test \
  --artifact-root ~/GSoC/deeplense_artifacts --tag groq_gpt-oss-120b_seeded_2026-09-28 \
  --seeds 0,1,2,3,4 --skip-b --tune-epochs 25 --tune-iterations 4
```

Total time was 175.4 min. There were 0 HTTP retries and 0 errors.

### Structured-output gate (run before any data was generated)

`openai/gpt-oss-120b` **passed the strict gate 20/20** (`scripts/check_structured_output.py`,
`groq_seed_reference/gate_gpt-oss-120b.json`). The gate treats every silent-repair path
as a failure: a judge ranking that production would repair, a planner key that `_apply`
would ignore, and an action that the loop would run as a plain retrain (see
`docs/KNOWN_ISSUES.md`).

I applied the same checks afterwards to all **18 LLM planner decisions made in the 10
runs**. None had an ignored key and none had a non-`train` action.

## Results

### Per seed

| Arm | Seed | Val acc | Test acc | Test AUC | Iters | Best iter | Stopped by |
|---|---|---|---|---|---|---|---|
| `cnn_medium_s4` | 0 | 0.8328 | 0.8178 | 0.9407 | 3 | 1 | guard: val stalled |
| `cnn_medium_s4` | 1 | 0.8672 | 0.8611 | 0.9588 | 3 | 2 | guard: gap closed |
| `cnn_medium_s4` | 2 | 0.8456 | 0.8378 | 0.9466 | 4 | 1 | guard: val stalled |
| `cnn_medium_s4` | 3 | 0.8206 | 0.8272 | 0.9398 | 3 | 0 | guard: val stalled |
| `cnn_medium_s4` | 4 | 0.8417 | 0.8272 | 0.9518 | 3 | 2 | guard: gap closed |
| `resnet34` | 0 | 0.7511 | 0.7433 | 0.8842 | 3 | 0 | guard: val stalled |
| `resnet34` | 1 | 0.8372 | 0.8383 | 0.9394 | 2 | 1 | guard: gap closed |
| `resnet34` | 2 | 0.7256 | 0.7411 | 0.8802 | 3 | 0 | guard: gap closed |
| `resnet34` | 3 | 0.7956 | 0.7994 | 0.9155 | 2 | 1 | guard: gap closed |
| `resnet34` | 4 | 0.7939 | 0.7778 | 0.9050 | 2 | 1 | guard: gap closed |

Accuracies are correct counts out of 1,800, rounded to 4 dp; the raw values are in
`groq_seed_reference/runs/*/metrics.json`.

### Per arm (n = 5; SD is the sample SD; range is max − min)

| Arm | | Mean | Range | SD |
|---|---|---|---|---|
| `cnn_medium_s4` | val | 0.8416 | 0.8206–0.8672 (0.0467) | 0.0173 |
| `cnn_medium_s4` | test | **0.8342** | 0.8178–0.8611 (0.0433) | **0.0166** |
| `resnet34` | val | 0.7807 | 0.7256–0.8372 (0.1117) | 0.0433 |
| `resnet34` | test | **0.7800** | 0.7411–0.8383 (0.0972) | **0.0408** |

Test AUC means: `cnn_medium_s4` 0.9476 (SD 0.0080), `resnet34` 0.9049 (SD 0.0242).

### Paired differences (`cnn_medium_s4` − `resnet34`, same seed)

| | s0 | s1 | s2 | s3 | s4 | Mean | SD | Sign |
|---|---|---|---|---|---|---|---|---|
| Val | +0.0817 | +0.0300 | +0.1200 | +0.0250 | +0.0478 | +0.0609 | 0.0398 | 5/5 positive |
| **Test** | +0.0744 | +0.0228 | +0.0967 | +0.0278 | +0.0494 | **+0.0542** | **0.0313** | **5/5 positive** |

As a supplementary figure, the 95% t-interval on the mean paired test difference
(df = 4) is **[+0.0153, +0.0931]**.

**Reading, within this experiment only:**

- `cnn_medium_s4` came out ahead of `resnet34` on held-out test accuracy in every seed.
  The smallest gap was +0.0228 and the mean was +0.0542.
- The interval on the mean excludes zero.
- `resnet34` was about 2.5x as variable across seeds (test SD 0.0408 vs 0.0166). Two of
  its five runs (seeds 0 and 2) finished near 0.74.
- With n = 5, one dataset generation, one planner model and 1,800 training images, this
  is a single reference point. It is not a general claim, and none of these numbers
  should be combined with or set against the recorded gpt-5.2 results.

### How the planner behaved in the runs

- **First interventions.** `dropout` appeared in 7 of the 10 first interventions and
  `augment` in 4. The exact first moves were:
  - 5× `dropout 0.2 + early_stop_patience 5`
  - 2× `augment + early_stop_patience 5`
  - 1× `augment + dropout 0.2`
  - 1× `dropout 0.3 + early_stop_patience 5`
  - 1× `augment`
- **Every run ended on the loop guard, never on a planner decision:** 4 because
  validation accuracy stalled, 6 because the train/val gap closed. As in
  `docs/AGENT_RELIABILITY.md` §0, the stopping decision in this experiment was not the
  agent's.
- **Three runs reported iteration 0**, the untuned baseline, because nothing the
  planner tried improved validation accuracy: `cnn_medium_s4` seed 3 and `resnet34` seeds
  0 and 2.
- **In `cnn_medium_s4` seed 3, the planner's changes made the model collapse.** The first
  change (`dropout 0.2 + early stop`) dropped validation accuracy from 0.821 to 0.337,
  which is chance level. The next change (`dropout 0.1`) left it at 0.347. The loop
  guard then stopped the run on a stall, and the reported number is iteration 0's.

## E1 follow-up: 20 trials on the underfit probe (reported separately)

This follows E1's protocol exactly: the same harness (`scripts/run_planner_reliability.py`,
changed only to build the model through `build_llm`), the same `underfit` state, 20
trials, and the same scoring rule (`judge_quality`). No training is involved.

The underfit state is a **constructed probe** (train 0.6500 / val 0.6294), not a recorded
observation (`docs/AGENT_RELIABILITY.md` §1). The scoring rule counts a decision as
"harmful" if it adds regularization **without** also raising capacity (epochs, learning
rate, widths or depths).

| | gpt-5.2 (recorded E1, 2026-08-11) | gpt-oss-120b via Groq (2026-09-28) |
|---|---|---|
| Actions | `train` 20/20 | `train` 20/20 |
| E1 quality | **19 harmful, 1 ok** | **18 ok, 2 harmful** |
| Raised epochs | 0/20 | **18/20** |
| Raised learning rate | 1/20 | 2/20 |
| Added augment / dropout / weight decay | 19/20 | 2/20 |
| Added early stopping | 16/20 | 11/20 (10 of them together with more epochs) |
| Modal knob set | `augment + early_stop_patience` (15/20) | `epochs + early_stop_patience` (9/20); `epochs` alone 7/20 |
| Knob-set entropy | 0.992 bits | 1.913 bits |

On this fixed probe, gpt-oss-120b nearly reverses gpt-5.2's recorded behaviour. It raised
capacity in 18 of 20 trials, where gpt-5.2 added regularization in 19 of 20. It is also
less stereotyped: knob-set entropy was about twice as high. Like gpt-5.2, it never chose
`report` or `stop`.

Caveats:

- It is one constructed state.
- The two runs are six weeks apart on different providers.
- 19 HTTP 429 retries occurred, because Groq's free tier allows 8K tokens per minute.
  All succeeded; retries do not change a trial's content.

Records: `groq_seed_reference/e1_underfit_gpt-oss-120b.{json,log}`.

## Limitations

- **Selection bias on validation.** Validation accuracy was used to pick the reported
  iteration, so it is optimistically biased. Test accuracy is the unbiased figure.
- **The training subset rule differs from the paper's.** `train_small` here is a
  documented rule (the first 600 per class of a seeded shuffle). The paper run's subset
  rule was never recorded.
- **The served model can change.** Groq serves `openai/gpt-oss-120b` under a name, not a
  pinned build. Groq-side updates would not be visible in these records.
- **The run banner overstates two things.** `run_multi_seed.py` hard-codes "full dataset"
  and "pipeline runs: 5" in its header. For this experiment the data was the 1,800-image
  subset and Experiment B was skipped (`--skip-b`). Each run's `config.json` records the
  real sizes.
- **Runtime was much higher than estimated:** about 7 minutes per tuning iteration, and
  most runs used 3–4 iterations. Swap grew to 5.8 GB during the runs, so memory pressure
  on this 16 GB machine may have contributed; that was not confirmed.

## Records

| Path | Contents |
|---|---|
| `groq_seed_reference/gate_gpt-oss-120b.{json,log}` | strict structured-output gate, 20/20 PASS |
| `groq_seed_reference/dataset/` | seeds, SHA-256 manifest, split map, image id, pip freeze, generation logs |
| `groq_seed_reference/runs/` | `manifest.json`, and per run: `config.json`, `metrics.json` (full tuning trajectory), `train_log.txt` |
| `groq_seed_reference/e1_underfit_gpt-oss-120b.{json,log}` | the 20-trial E1 underfit probe |

Checkpoints (1.1 GB) and prediction files stay outside the repository, in
`~/GSoC/deeplense_artifacts/groq_gpt-oss-120b_seeded_2026-09-28/`.
