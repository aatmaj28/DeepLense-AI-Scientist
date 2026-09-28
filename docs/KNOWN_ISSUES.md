# Known issues: silent-repair paths in the agent loop

**Recorded: 2026-09-28** · Found while building a strict structured-output check for a
second LLM provider (see §4).

These belong to the same class as the guard-attribution finding in
`docs/AGENT_RELIABILITY.md` §0. In each case the code quietly substitutes for, or
discards, part of what the LLM returned, and the recorded artifacts still look
normal. None of them is known to have changed a recorded result (the evidence is
given per issue), but each one would let a misbehaving model produce a run that
looks successful.

None of §1–§3 has been fixed. The code at the cited lines is unchanged.

---

## 1. Judge ranking repair: omissions are hidden, duplicates are kept

**Where:** `src/dlens/agents/_architecture_search.py:253-256`

```python
ranking = [i for i in verdict.output.ranking if 0 <= i < len(candidates)]
# The judge may omit indices; append anything it left out so the
# shortlist is always well defined.
ranking += [i for i in range(len(candidates)) if i not in ranking]
```

**Behaviour**

- **Out-of-range indices** are dropped without any message.
- **Omitted candidates** are appended in ascending index order. That silently ranks
  them last. The append line was added in `b2cc03d` (2026-08-15). Before that, only
  the range filter existed.
- **Duplicates are not removed.** A duplicated index takes two top-k slots, so that
  candidate is shortlisted and trained twice and a different candidate loses its
  place. With 5 candidates, `[3, 3, 11, 0]` becomes `[3, 3, 0, 1, 2, 4]`, and top-4 is
  `[3, 3, 0, 1]`.
- **The raw judge output is never saved.** The printed `judge ranking:` line (:260)
  and the recorded `judge_ranking` (:281) both come after the repair. So nobody can
  tell afterwards, from the artifacts alone, whether a repair happened.

**Evidence in the recorded artifacts**

- `docs/ai_scientist_run.json` (2026-07-25, `gpt-5.6-luna`, commit `f450b2b`)
  records `judge_ranking [3, 5, 2, 7, 6, 0, 4, 8, 9]` for **10 proposed candidates,
  0 unbuildable**. Candidate 1 (`cnn_small_deep`) was omitted by the judge. At that
  commit only the range filter existed, so the omission shows. The candidate was
  never ranked and nothing flagged it. The top-4 shortlist was unaffected.
- No recorded ranking contains a duplicate.
- The 20 raw gpt-5.2 rankings in `docs/agent_reliability/e2_judge.json` are all
  exact permutations. That harness calls the judge directly and bypasses the repair.
- Every search run since 2026-08-15 records the ranking after the repair, including
  the broadened search at `docs/broader_search/run_partial.log:21,80,138`. So an
  omission there cannot be ruled out. A single omitted index would simply show up as
  the last element.

**Suggested fix (not applied):** require an exact permutation of
`range(len(candidates))`. On failure, raise, or ask the model again through a
pydantic-ai output validator (`ModelRetry`). Save the raw ranking next to whatever
is used.

---

## 2. `_apply` silently ignores planner parameters

**Where:** `src/dlens/agents/_experiment_loop.py:200-223` (`_apply`). The allowed keys
are at :46-50, and the call site is :178.

**Behaviour**

- **Unknown keys are dropped.** Only `_ARCH_KEYS` (`name`, `family`,
  `physics_informed`) and `_CFG_KEYS` (`loss`, `optimizer`, `learning_rate`,
  `batch_size`, `epochs`, `weight_decay`, `lr_scheduler`, `dropout`, `augment`,
  `early_stop_patience`) are applied. Anything else disappears without an error.
- **`depths` and `widths` are not in `_ARCH_KEYS`.** A planner decision to resize the
  network therefore does nothing, and the next iteration retrains the same
  architecture.
- **A `training` or `architecture` group whose value is not a dict** is dropped.
- **A decision with no applicable keys still triggers a retrain.** The trajectory
  records it as a planner intervention even though nothing changed.

**Related: actions the loop cannot carry out.** `_TERMINAL` is `{REPORT, STOP}`
(:44). Every other action falls through to `_apply` plus a retrain (:174-178). That
includes `simulate` and `design_model`. The loop has no simulation or redesign
stage, so those decisions are silently executed as ordinary retrains.

**Evidence in the recorded artifacts:** I checked all 87 non-terminal planner
decisions recorded under `docs/**/*.json`. **0 contain ignored keys, and all 87 are
`train`.** So no recorded result was affected. The behaviour only matters for a
planner that emits other keys or actions, such as a different LLM or a changed
prompt.

**Suggested fix (not applied):**

- Make `_apply` strict: raise on unknown keys and on non-dict groups.
- Either add `depths`/`widths` to `_ARCH_KEYS` or reject them explicitly.
- In loops that have no simulation or redesign stage, reject actions other than
  `train` that aren't terminal.
- Log the settings that actually changed at each iteration.

---

## 3. `PlannerDecision` can never be decoded under strict mode

**Where:** `src/dlens/schemas/_experiment.py:37-39`, `updated_params: dict[str, Any]`

**Behaviour:** strict (constrained) decoding needs every object to be closed
(`additionalProperties: false`) and every field to be required. An open dict cannot
meet that, so pydantic-ai never asks for strict mode on the planner's output tool.
The model generates the JSON on a best-effort basis, and only pydantic validation
afterwards (with `retries=2` re-asks) enforces its shape.

Nothing enforces what `updated_params` contains, and that field is exactly where §2's
silent ignores happen. So a `PlannerDecision` that validates proves only that the
JSON parsed. What it asks for has to be checked separately. The strict gate
(`scripts/check_structured_output.py`) does that check; production does not.

**This is not specific to any provider, and it applies to the recorded gpt-5.2
runs.** I captured the request payloads offline through a mock transport. The
pydantic-ai version was 2.1.0, which `uv.lock` has pinned since 2026-06-29, so it
covers every recorded run. The planner and judge schemas were unchanged over that
period (checked at `f450b2b`).

| Path | Planner output tool | Judge output tool |
|---|---|---|
| gpt-5.2 (Chat Completions) | `strict=None` | `strict=True` |
| gpt-5.6-luna (Responses API) | `strict=False` | `strict=True` |
| Groq (OpenAI-compatible endpoint) | `strict=None` | `strict=True` (Groq documents `strict` only for `response_format`, not for tools) |

**The judge is the reverse case, with a limit.** `JudgeVerdict` (`reasoning: str`,
`ranking: list[int]`) *is* strict-compatible, so on OpenAI its shape is constrained.
But a schema can only guarantee "a list of integers". It cannot guarantee "an exact
permutation of `range(n)`". So §1's repair path still applies under strict decoding.
The Luna judge's omitted candidate (§1) happened with `strict=True` on the judge tool.

**Suggested fix (not applied):** replace the open dict with closed, typed models of
the settings `_apply` can actually apply: the `_CFG_KEYS` as training overrides and
the `_ARCH_KEYS` as architecture overrides. That would make the planner schema
strict-compatible, turn unknown keys into schema errors instead of silent drops, and
remove §2's main failure mode.

---

## 4. Gemini support: added, not verified against a live model

Branch `exp/gemini-seed-reference` adds `GeminiModel` (`src/dlens/agents/_models.py`,
using the native pydantic-ai Google provider) and routes `gemini-*` ids in
`build_llm` (`src/dlens/config.py`). The offline routing tests pass. The same branch
changes `scripts/run_multi_seed.py` in two ways:

- it prints the effective model at the top of every run's own `train_log.txt`;
- it records the model that was actually built in each run's `config.json`.

**Structured output (`PlannerDecision`, `JudgeVerdict`) has not been verified
against a live Gemini model.** The one attempt, on `gemini-3.8-flash` (2026-09-28),
got 0 responses out of 20 calls: 6 × HTTP 503 ("high demand"), then 14 × HTTP 429
(free-tier quota, limit 5).

No transport-level retry is configured. A transient 429 or 503 during a planner call
therefore aborts the run.

Before using Gemini in any recorded experiment, run `scripts/check_structured_output.py`,
which turns every silent path above into a failure:

- the judge ranking must be an exact permutation;
- no planner key may be one that `_apply` would ignore;
- the only allowed actions are `train`, `report` and `stop`.
