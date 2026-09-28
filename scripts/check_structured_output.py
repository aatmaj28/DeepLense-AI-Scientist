"""STRICT gate: does an LLM return the planner's and the judge's typed schemas reliably?

Runs the PRODUCTION code paths, not a toy schema:
  build_llm()  (DLENS_LLM)  ->  ExperimentPlanner.decide(state)  -> PlannerDecision
                            ->  ArchitectureJudge.arun(prompt)   -> JudgeVerdict

Production has three silent paths that would let a broken model look healthy. This
gate turns each one into a FAILURE instead of a pass:

  1. Judge ranking repair (_architecture_search.py:253-256): out-of-range indices are
     dropped and missing ones appended. FAIL unless the ranking is an exact
     permutation of range(n): no duplicates, no out-of-range, none missing.
  2. _apply ignoring keys (_experiment_loop.py:200-223): unknown keys, keys outside
     _ARCH_KEYS/_CFG_KEYS (note: depths/widths are NOT applicable), and non-dict
     "training"/"architecture" groups are dropped without error. FAIL on any.
  3. Non-terminal actions other than `train` (simulate, design_model): the loop has no
     simulation or redesign stage, so it silently treats them as a retrain. FAIL.

Also FAIL: schema/validation errors after retries, and _apply raising on a bad value.
WARN (reported, not failed): a `train` decision that changes nothing, calls that
needed pydantic-ai output-validation retries (the final output is still strictly
validated, but retries measure unreliability), and HTTP 429/5xx transport retries.

Inputs are recorded observations already in the repo:
  planner: run_planner_reliability.build_states() overfit / healthy (VERBATIM RECORDED,
           paper_run resnet34_ref it0 / it1) and underfit (constructed probe)
  judge  : broadened-search round-1 candidates, docs/broader_search/run_partial.log:11-20

Run from the repo root:
  uv run python scripts/check_structured_output.py --list-groq
  uv run python scripts/check_structured_output.py --limits groq:<model id>
  DLENS_LLM=<id> uv run python scripts/check_structured_output.py \
      [--trials N] [--pace SECONDS] [--out PATH]
Exit code 0 only if there are zero FAILs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import time

REPO = os.getcwd()


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _CountRetries:
    """Tee stdout while counting the transport's `[llm retry]` lines for one call."""

    def __enter__(self):
        self._orig, self.n = sys.stdout, 0
        sys.stdout = self
        return self

    def write(self, data):
        self.n += data.count("[llm retry]")
        return self._orig.write(data)

    def flush(self):
        self._orig.flush()

    def __exit__(self, *exc):
        sys.stdout = self._orig


_last_call = [0.0]


async def _pace(seconds: float) -> None:
    """Start consecutive calls at least `seconds` apart (stay under the provider's TPM)."""
    wait = _last_call[0] + seconds - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_call[0] = time.monotonic()


async def groq_info() -> int:
    """--list-groq: models the key can reach. --limits ID: rate-limit headers for one model."""
    import httpx

    from dlens.config import GROQ_BASE_URL

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set in the environment or .env")
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(timeout=60) as c:
        if "--list-groq" in sys.argv:
            r = await c.get(f"{GROQ_BASE_URL}/models", headers=headers)
            r.raise_for_status()
            for m in sorted(r.json()["data"], key=lambda m: m["id"]):
                print(f"{m['id']:46s} owner={m.get('owned_by', '?'):16s} "
                      f"ctx={m.get('context_window', '?')} "
                      f"max_out={m.get('max_completion_tokens', '?')} "
                      f"active={m.get('active', '?')}")
        if "--limits" in sys.argv:
            mid = sys.argv[sys.argv.index("--limits") + 1].removeprefix("groq:")
            r = await c.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json={
                "model": mid, "messages": [{"role": "user", "content": "Reply: ok"}],
                "max_completion_tokens": 16})
            print(f"{mid}: HTTP {r.status_code}")
            for k in sorted(r.headers):
                if k.startswith("x-ratelimit") or k == "retry-after":
                    print(f"  {k}: {r.headers[k]}")
            if r.status_code >= 400:
                print(f"  body: {r.text[:300]}")
    return 0


def _usage(result) -> dict:
    from pydantic_ai.messages import RetryPromptPart

    u = result.usage() if callable(result.usage) else result.usage  # property in 2.x
    retries = sum(isinstance(p, RetryPromptPart)
                  for m in result.all_messages() for p in getattr(m, "parts", []))
    return {"requests": getattr(u, "requests", None), "validation_retries": retries,
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None)}


def ignored_keys(params: dict) -> list[str]:
    """Every key in updated_params that _apply would silently drop (mirrors _apply)."""
    from dlens.agents._experiment_loop import _ARCH_KEYS, _CFG_KEYS

    dropped = []
    for k, v in (params or {}).items():
        if k in ("architecture", "training"):
            if not isinstance(v, dict):
                dropped.append(f"{k} (non-dict group)")
                continue
            allowed = _ARCH_KEYS if k == "architecture" else _CFG_KEYS
            dropped += [f"{k}.{kk}" for kk in v if kk not in allowed]
        elif k not in _ARCH_KEYS and k not in _CFG_KEYS:
            dropped.append(k)
    return dropped


def round1_candidates():
    from dlens.schemas._model_design import ArchitectureSpec

    pat = re.compile(r"\[(\d+)\]\s+(\S+)\s+(\S+)\s+depths=(\[[^\]]*\])\s+widths=(\[[^\]]*\])")
    lines = open(os.path.join(REPO, "docs/broader_search/run_partial.log")).read().splitlines()
    specs = []
    for line in lines[10:20]:  # run_partial.log:11-20
        _, name, fam, depths, widths = pat.search(line).groups()
        specs.append(ArchitectureSpec.model_validate({
            "name": name, "family": fam, "input_shape": [150, 150], "channels": 1,
            "num_classes": 3, "depths": json.loads(depths), "widths": json.loads(widths),
            "physics_informed": fam == "equivariant", "rationale": "",
        }))
    return specs


def production_repair(ranking: list[int], n: int) -> list[int]:
    """What _architecture_search.py:253-256 would silently turn this ranking into."""
    r = [i for i in ranking if 0 <= i < n]
    return r + [i for i in range(n) if i not in r]


async def main() -> int:
    from dlens.config import build_llm  # also loads the repo .env

    if "--list-groq" in sys.argv or "--limits" in sys.argv:
        return await groq_info()

    def _arg(flag, default):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

    trials, pace = int(_arg("--trials", 5)), float(_arg("--pace", 0))
    llm = build_llm()
    base_url = str(getattr(llm, "base_url", "?"))
    print(f"effective LLM: {type(llm).__name__} model_name={llm.model_name} "
          f"base_url={base_url} (DLENS_LLM={os.getenv('DLENS_LLM')}) "
          f"trials={trials} pace={pace}s", flush=True)
    if (os.getenv("DLENS_LLM") or "").startswith("groq") and "api.groq.com" not in base_url:
        raise SystemExit("DLENS_LLM is a groq id but the model is not served by api.groq.com")

    from dlens.agents._architecture_search import ArchitectureJudge, _params_m
    from dlens.agents._experiment_loop import _TERMINAL, _apply
    from dlens.agents._experiment_planner import ExperimentPlanner
    from dlens.schemas._experiment import ExperimentState, PlannerAction, PlannerDecision

    out = {"model": llm.model_name, "base_url": base_url, "trials": trials, "pace_s": pace,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "planner": [], "judge": []}

    # ---- planner ---------------------------------------------------------------
    states = _load("rp", "scripts/run_planner_reliability.py").build_states()
    planner = ExperimentPlanner(model=llm)
    for name in ("overfit", "healthy", "underfit"):
        st = ExperimentState.model_validate(states[name]["state"])
        run = st.runs[-1]
        for t in range(trials):
            t0 = time.monotonic()
            fails, warns = [], []
            rec = {"state": name, "trial": t + 1}
            await _pace(pace)
            # Only the model call is guarded: a bug in the checks below must crash the
            # gate, never be reported as a model failure.
            res = None
            with _CountRetries() as http:
                try:
                    res = await planner.decide(st)
                except Exception as exc:
                    fails.append(f"no valid PlannerDecision: {type(exc).__name__}: {exc}"[:500])
            rec["transport_retries"] = http.n
            if http.n:
                warns.append(f"{http.n} HTTP 429/5xx transport retries")
            if res is not None:
                d = res.output
                if not (isinstance(d, PlannerDecision) and isinstance(d.action, PlannerAction)
                        and isinstance(d.updated_params, dict)):
                    fails.append("schema: output is not a PlannerDecision")
                dropped = ignored_keys(d.updated_params)
                if dropped:
                    fails.append(f"_apply would silently ignore: {dropped}")
                if d.action not in _TERMINAL and d.action is not PlannerAction.TRAIN:
                    fails.append(f"action '{d.action.value}': the loop has no such stage and "
                                 "would silently treat it as a retrain")
                changed = {}
                if d.action not in _TERMINAL:
                    try:  # _apply raising is itself a production failure mode
                        arch2, cfg2 = _apply(d.updated_params, run.architecture,
                                             run.training_config)
                    except Exception as exc:
                        fails.append(f"_apply raised {type(exc).__name__}: {exc}"[:300])
                    else:
                        changed = {k: v for k, v in cfg2.model_dump().items()
                                   if run.training_config.model_dump().get(k) != v}
                        changed |= {k: v for k, v in arch2.model_dump().items()
                                    if run.architecture.model_dump().get(k) != v}
                        if not changed:
                            warns.append("train decision changes no setting (retrains as-is)")
                usage = _usage(res)
                if usage["validation_retries"]:
                    warns.append(f"{usage['validation_retries']} output-validation retries")
                rec |= {"action": d.action.value, "updated_params": d.updated_params,
                        "applied_changes": changed, "rationale": d.rationale[:200], **usage}
            rec |= {"fails": fails, "warns": warns, "seconds": round(time.monotonic() - t0, 1)}
            tag = "FAIL" if fails else ("warn" if warns else "ok  ")
            print(f"planner {tag} {json.dumps(rec, default=str)}", flush=True)
            out["planner"].append(rec)

    # ---- judge -----------------------------------------------------------------
    specs = round1_candidates()
    n = len(specs)
    task = ("Classify 150x150 grayscale strong-lensing images into 3 dark-matter "
            "substructure classes (no_sub, cdm, axion). 7200 training images, 1800 validation.")
    listing = "\n".join(
        f"[{i}] name={c.name} family={c.family.value} depths={c.depths} "
        f"widths={c.widths} params={_params_m(c)}M" for i, c in enumerate(specs))
    prompt = f"{task}\n\nCandidates:\n{listing}\n\nRank all candidate indices."
    judge = ArchitectureJudge(model=llm)
    for t in range(trials):
        t0 = time.monotonic()
        fails, warns = [], []
        rec = {"trial": t + 1}
        await _pace(pace)
        res = None
        with _CountRetries() as http:
            try:
                res = await judge.arun(prompt)
            except Exception as exc:
                fails.append(f"no valid JudgeVerdict: {type(exc).__name__}: {exc}"[:500])
        rec["transport_retries"] = http.n
        if http.n:
            warns.append(f"{http.n} HTTP 429/5xx transport retries")
        if res is not None:
            r = res.output.ranking
            dup = sorted({i for i in r if r.count(i) > 1})
            oob = [i for i in r if not 0 <= i < n]
            missing = [i for i in range(n) if i not in r]
            if dup or oob or missing:
                fails.append(f"malformed ranking production would silently repair: "
                             f"duplicates={dup} out_of_range={oob} missing={missing} "
                             f"-> repaired {production_repair(r, n)}")
            if not (res.output.reasoning or "").strip():
                warns.append("empty reasoning")
            usage = _usage(res)
            if usage["validation_retries"]:
                warns.append(f"{usage['validation_retries']} output-validation retries")
            rec |= {"ranking": r, "reasoning_chars": len(res.output.reasoning or ""), **usage}
        rec |= {"fails": fails, "warns": warns, "seconds": round(time.monotonic() - t0, 1)}
        tag = "FAIL" if fails else ("warn" if warns else "ok  ")
        print(f"judge   {tag} {json.dumps(rec, default=str)}", flush=True)
        out["judge"].append(rec)

    # ---- verdict ---------------------------------------------------------------
    allrec = out["planner"] + out["judge"]
    n_fail = sum(bool(r["fails"]) for r in allrec)
    n_warn = sum(bool(r["warns"]) and not r["fails"] for r in allrec)
    out["verdict"] = {"fail_trials": n_fail, "warn_trials": n_warn, "total": len(allrec)}
    path = _arg("--out", f"structured_output_gate_{re.sub(r'[^A-Za-z0-9.-]+', '_', llm.model_name)}.json")
    json.dump(out, open(path, "w"), indent=2, default=str)
    print("\n" + "=" * 78)
    print(f"GATE {'FAIL' if n_fail else 'PASS'}  model={llm.model_name}  "
          f"fail trials {n_fail}/{len(allrec)}  warn-only trials {n_warn}/{len(allrec)}")
    for r in allrec:
        for f in r["fails"]:
            print(f"  FAIL {r.get('state', 'judge')} #{r['trial']}: {f}")
    print(f"  -> {path}")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
