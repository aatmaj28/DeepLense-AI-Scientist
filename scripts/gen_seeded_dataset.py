"""Seeded, manifest-backed DeepLense Model_I dataset generation (reproducible by design).

The original 9,000-image pool was generated unseeded (gen_model1_dataset.py without its
optional 4th argument), so it cannot be regenerated. This driver runs the same
generator with an explicit seed per (split, class), single-threaded BLAS so results do
not depend on thread count, and records everything needed to redo it:

  <out>/raw_pool/<cls>_NNNNN.npy     generated pool   (seeded per class)
  <out>/raw_test/<cls>_NNNNN.npy     held-out test    (seeded per class, disjoint seeds)
  <out>/model1/{train,val}/          seeded 80/20 split (prepare_deeplense_dataset.py)
  <out>/model1/train_small/          first --small-per-class files per class of train/
  <out>/model1_test/                 the test set, as a DatasetRef root
  <out>/MANIFEST.sha256              sha256 of every generated file (sha256sum format)
  <out>/SPLITS.tsv                   split file -> generated file it links to
  <out>/dataset.json                 seeds, counts, commands, image id, pip freeze, ...

Every split file is a hard link to a generated file, so the manifest covers them.

  # time a small run first (throwaway seeds, scratch dir, no manifest)
  uv run python scripts/gen_seeded_dataset.py --bench 20 --out /tmp/bench
  # the real thing
  uv run python scripts/gen_seeded_dataset.py --out ~/GSoC/deeplense_data/<name>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = "dlens-lenstronomy:latest"
CLASSES = ("no_sub", "cdm", "axion")
# Same result regardless of how many cores the container sees.
THREAD_ENV = ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1")


def docker_cmd(cls: str, count: int, outdir: str, seed: int) -> list[str]:
    cmd = ["docker", "run", "--rm", "--network", "none",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    for kv in THREAD_ENV:
        cmd += ["-e", kv]
    cmd += ["-v", f"{REPO}/scripts:/scripts:ro", "-v", f"{outdir}:/out", IMAGE,
            "python", "/scripts/gen_model1_dataset.py", cls, str(count), "/out", str(seed)]
    return cmd


def run_job(job: dict) -> dict:
    os.makedirs(job["outdir"], exist_ok=True)
    t0 = time.monotonic()
    with open(job["log"], "w") as fh:
        rc = subprocess.run(job["cmd"], stdout=fh, stderr=subprocess.STDOUT).returncode
    tail = open(job["log"]).read().strip().splitlines()[-1:] or [""]
    m = re.search(r"DONE (\d+)/(\d+) in (\d+)s \((\d+) attempts\)", tail[0])
    n = len(glob.glob(os.path.join(job["outdir"], f"{job['cls']}_*.npy")))
    res = {k: job[k] for k in ("split", "cls", "seed", "count")} | {
        "returncode": rc, "files": n, "wall_s": round(time.monotonic() - t0, 1),
        "attempts": int(m.group(4)) if m else None, "log": os.path.basename(job["log"])}
    print(f"  [{job['split']}/{job['cls']} seed={job['seed']}] rc={rc} files={n}/"
          f"{job['count']} attempts={res['attempts']} {res['wall_s']}s", flush=True)
    return res


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--pool", type=int, default=3000, help="pool images per class")
    p.add_argument("--test", type=int, default=600, help="test images per class")
    p.add_argument("--pool-seeds", default="1001,1002,1003", help="per class, in CLASSES order")
    p.add_argument("--test-seeds", default="2001,2002,2003", help="per class, in CLASSES order")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--small-per-class", type=int, default=600)
    p.add_argument("--jobs", type=int, default=4, help="concurrent containers (CPU only)")
    p.add_argument("--bench", type=int, default=0,
                   help="timing run: N images per class, pool only, throwaway seeds 9001-9003")
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    pool_seeds = [int(s) for s in args.pool_seeds.split(",")]
    test_seeds = [int(s) for s in args.test_seeds.split(",")]
    reserved = {777, 778, 779}  # the original held-out test set's seeds
    if args.bench:
        pool_seeds, args.pool = [9001, 9002, 9003], args.bench
    if reserved & set(pool_seeds + test_seeds) or len(set(pool_seeds + test_seeds)) < 6:
        raise SystemExit("seeds must be distinct and must not reuse 777/778/779")
    if not args.bench and os.path.exists(os.path.join(out, "raw_pool")):
        raise SystemExit(f"{out}/raw_pool exists; refusing to mix generations")

    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    jobs = []
    splits = [("pool", args.pool, pool_seeds)] + ([] if args.bench else
                                                   [("test", args.test, test_seeds)])
    for split, count, seeds in splits:
        for cls, seed in zip(CLASSES, seeds):
            outdir = os.path.join(out, f"raw_{split}")
            jobs.append({"split": split, "cls": cls, "seed": seed, "count": count,
                         "outdir": outdir, "log": os.path.join(out, "logs", f"{split}_{cls}.log"),
                         "cmd": docker_cmd(cls, count, outdir, seed)})
    jobs.sort(key=lambda j: -j["count"])  # longest first
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"generating {sum(j['count'] for j in jobs)} images in {len(jobs)} jobs, "
          f"{args.jobs} at a time -> {out}", flush=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(run_job, jobs))
    wall = round(time.monotonic() - t0, 1)
    bad = [r for r in results if r["returncode"] != 0 or r["files"] != r["count"]]
    if bad:
        raise SystemExit(f"generation incomplete: {bad}")

    if args.bench:
        print(f"\nBENCH {args.bench}/class in {wall}s wall (jobs ran concurrently)")
        for r in results:
            print(f"  {r['cls']:7s} {r['count'] / r['wall_s']:.3f} img/s  "
                  f"({r['wall_s'] / r['count']:.2f} s/img, attempts {r['attempts']})")
        json.dump({"bench": results, "wall_s": wall},
                  open(os.path.join(out, "bench.json"), "w"), indent=2)
        return 0

    # ---- manifest + overlap -----------------------------------------------------
    files = sorted(glob.glob(os.path.join(out, "raw_*", "*.npy")))
    hashes = {os.path.relpath(f, out): sha256(f) for f in files}
    with open(os.path.join(out, "MANIFEST.sha256"), "w") as fh:
        for rel, h in hashes.items():
            fh.write(f"{h}  {rel}\n")
    pool_h = [h for r, h in hashes.items() if r.startswith("raw_pool/")]
    test_h = [h for r, h in hashes.items() if r.startswith("raw_test/")]
    overlap = sorted(set(pool_h) & set(test_h))
    dup_pool, dup_test = len(pool_h) - len(set(pool_h)), len(test_h) - len(set(test_h))
    print(f"manifest: {len(hashes)} files | test-pool hash overlap {len(overlap)} | "
          f"duplicates within pool {dup_pool}, within test {dup_test}", flush=True)
    if overlap or dup_pool or dup_test:
        raise SystemExit("hash overlap or duplicates found; not splitting")

    # ---- split, train_small, test root --------------------------------------------
    model1 = os.path.join(out, "model1")
    subprocess.run([sys.executable, os.path.join(REPO, "scripts/prepare_deeplense_dataset.py"),
                    "--src", os.path.join(out, "raw_pool"), "--out", model1,
                    "--classes", ",".join(CLASSES), "--seed", str(args.split_seed)], check=True)
    small = os.path.join(model1, "train_small")
    test_root = os.path.join(out, "model1_test")
    os.makedirs(small, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    for cls in CLASSES:
        for i in range(args.small_per_class):
            name = f"{cls}_{i:05d}.npy"
            os.link(os.path.join(model1, "train", name), os.path.join(small, name))
        for f in sorted(glob.glob(os.path.join(out, "raw_test", f"{cls}_*.npy"))):
            os.link(f, os.path.join(test_root, os.path.basename(f)))

    by_inode = {os.stat(f).st_ino: os.path.relpath(f, out) for f in files}
    rows = []
    for d in ("model1/train", "model1/val", "model1/train_small", "model1_test"):
        for f in sorted(glob.glob(os.path.join(out, d, "*.npy"))):
            rows.append(f"{os.path.relpath(f, out)}\t{by_inode[os.stat(f).st_ino]}")
    with open(os.path.join(out, "SPLITS.tsv"), "w") as fh:
        fh.write("split_file\tgenerated_file\n" + "\n".join(rows) + "\n")
    counts = {d: len(glob.glob(os.path.join(out, d, "*.npy")))
              for d in ("model1/train", "model1/val", "model1/train_small", "model1_test")}

    # ---- provenance ----------------------------------------------------------------
    def sh(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    freeze = sh(["docker", "run", "--rm", "--network", "none", IMAGE, "pip", "freeze"])
    with open(os.path.join(out, "pip_freeze.txt"), "w") as fh:
        fh.write(freeze + "\n")
    meta = {
        "started_utc": started, "generation_wall_s": wall,
        "classes": list(CLASSES),
        "seeds": {"pool": dict(zip(CLASSES, pool_seeds)), "test": dict(zip(CLASSES, test_seeds)),
                  "split": args.split_seed},
        "counts": {"pool_per_class": args.pool, "test_per_class": args.test, **counts},
        "rules": {
            "split": f"scripts/prepare_deeplense_dataset.py --seed {args.split_seed} "
                     "--val-frac 0.2 (seeded shuffle of sorted filenames per class)",
            "train_small": f"first {args.small_per_class} files per class of model1/train "
                           "(<cls>_00000 .. by index; the split shuffle makes this a "
                           "seeded random sample)",
        },
        "jobs": results,
        "docker_command_template": " ".join(docker_cmd("<cls>", 0, "<outdir>", 0)),
        "image": {"tag": IMAGE, "id": sh(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE]),
                  "python": sh(["docker", "run", "--rm", "--network", "none", IMAGE,
                                "python", "--version"])},
        "generator_sha256": sha256(os.path.join(REPO, "scripts/gen_model1_dataset.py")),
        "repo_commit": sh(["git", "-C", REPO, "rev-parse", "HEAD"]),
        "host": {"docker": sh(["docker", "version", "-f", "{{.Server.Version}}"]),
                 "colima": sh(["colima", "version"]).splitlines()[:1]},
        "hash_check": {"files": len(hashes), "test_pool_overlap": 0,
                       "duplicates_pool": 0, "duplicates_test": 0},
    }
    json.dump(meta, open(os.path.join(out, "dataset.json"), "w"), indent=2)
    print(json.dumps(counts), f"\ndone in {wall}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
