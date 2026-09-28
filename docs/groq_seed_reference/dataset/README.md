# Seeded DeepLense Model_I dataset — generation record (2026-09-28)

**SEPARATE EXPERIMENT, for reference only.** This dataset was generated fresh for the
Groq-planner seed runs (`docs/GROQ_SEED_REFERENCE.md`). It is **not** the dataset behind
any recorded or published result (that pool was generated unseeded and cannot be
reproduced), and results on it are not comparable to the recorded gpt-5.2 runs.

The data itself lives outside the repository (`~/GSoC/deeplense_data/seeded_2026-09-28/`,
~972 MB). This directory is everything needed to regenerate and verify it.

| File | What it is |
|---|---|
| `dataset.json` | seeds, counts, split rules, per-job results, docker command, image id, generator hash, repo commit |
| `MANIFEST.sha256` | SHA-256 of all 10,800 generated files (`shasum -a 256 -c MANIFEST.sha256` from the data dir) |
| `SPLITS.tsv` | every split file (train / val / train_small / test) and the generated file it hard-links to |
| `pip_freeze.txt`, `image_extra.json` | the image's Python stack, and the pyHalo commit inside the image |
| `generate.log`, `logs/` | driver output and each generator job's log (0 skipped realizations) |

**Seeds:** pool no_sub 1001, cdm 1002, axion 1003 · test no_sub 2001, cdm 2002, axion 2003
(none reuse the original test set's 777/778/779) · split seed 0.

**Regenerate:** build `sandbox/Dockerfile` as `dlens-lenstronomy:latest`, then
`uv run python scripts/gen_seeded_dataset.py --out <dir>` at the recorded repo commit, and
check the result with `shasum -a 256 -c MANIFEST.sha256`. The Dockerfile pins version
*ranges*, so a rebuild can resolve different library versions; compare against
`pip_freeze.txt` and `image_extra.json` first. A byte-identical rebuild was verified on
this machine for a 20-image-per-class sample (two independent runs, 60/60 SHA-256
matches), not for the full set.

**Verified after generation:** 10,800/10,800 manifest hashes OK; 0 shared hashes between
test and pool; 0 duplicates; train_small ⊂ train; train ∩ val = ∅; every split exactly
balanced by class.
