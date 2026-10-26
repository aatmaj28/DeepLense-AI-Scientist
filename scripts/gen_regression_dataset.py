"""Seeded, manifest-backed SIE regression dataset, with lens parameters as labels.

Same approach as the seeded Model_I driver: fixed seeds, a separately seeded test set,
a SHA-256 manifest and recorded provenance. Before generating anything, it checks the
flip/rotation label maps against lenstronomy (scripts/check_label_transforms.py) and
stops if they disagree. Containers run one at a time; each uses --workers processes.

    <out>/raw_pool/pool_NNNNN.npy + labels.csv   generated pool (seed --pool-seed)
    <out>/raw_test/test_NNNNN.npy + labels.csv   held-out test  (seed --test-seed, disjoint)
    <out>/train/, <out>/val/                     seeded split of the pool (hard links + labels.csv)
    <out>/test/                                  the test set (hard links + labels.csv)
    <out>/MANIFEST.sha256                        every generated file, labels.csv included
    <out>/SPLITS.tsv                             split file -> generated file
    <out>/label_stats.json                       per-split label distributions
    <out>/dataset.json                           seeds, counts, checks, image, versions

Usage:
    uv run python scripts/gen_regression_dataset.py --bench 40 --out /tmp/bench   # timing + determinism
    uv run python scripts/gen_regression_dataset.py --out ~/GSoC/deeplense_data/<name>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from dlens.data._lens_labels import TARGETS
from dlens.data._regression_dataset import (column_stats, ks_uniform, read_labels, sha256, sparkline,
                                            split_rows, write_labels)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = "dlens-lenstronomy:latest"
THREAD_ENV = ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1")
# Seeds already used by the Model_I datasets (original test, seeded pool/test, bench).
RESERVED_SEEDS = {777, 778, 779, 1001, 1002, 1003, 2001, 2002, 2003, 9001, 9002, 9003}
BENCH_SEED = 9101

# The sampling distributions in scripts/lens_regression_sim.py, for the uniformity checks.
UNIFORM_COLUMNS = {"theta_E": (0.5, 2.5), "center_x": (-0.3, 0.3), "center_y": (-0.3, 0.3),
                   "q": (0.6, 1.0), "phi": (0.0, math.pi), "src_R_sersic": (0.1, 0.5),
                   "src_n_sersic": (0.5, 2.5)}
LOG_UNIFORM_COLUMNS = {"src_peak_sb": (90.0, 450.0)}
REPORT_COLUMNS = (*TARGETS, "q", "phi", "src_R_sersic", "src_n_sersic", "src_peak_sb", "src_amp",
                  "exposure_time", "peak_snr", "snr_measured")
SNR_BAND = (14.0, 60.0)  # the validator's default band (tools/_param_validator.py)
N_BINS = ((0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5))


def spearman(a, b) -> float:
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])


def n_bin(n: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (n >= lo) & ((n < hi) if hi < N_BINS[-1][1] else (n <= hi))


def docker(args: list[str], mounts: dict[str, str], cpus: int = 1) -> list[str]:
    cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", str(cpus), "--memory", "6g",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    for kv in THREAD_ENV:
        cmd += ["-e", kv]
    for host, container in mounts.items():
        cmd += ["-v", f"{host}:{container}"]
    return cmd + [IMAGE, *args]


def code_mounts() -> dict[str, str]:
    return {f"{REPO}/scripts": "/scripts:ro", f"{REPO}/src/dlens/data": "/labels:ro"}


def generate(outdir: str, prefix: str, count: int, seed: int, workers: int, log: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    cmd = docker(["python", "/scripts/gen_lens_regression_dataset.py", "/out", "--count", str(count),
                  "--seed", str(seed), "--prefix", prefix, "--workers", str(workers)],
                 {**code_mounts(), outdir: "/out"}, cpus=workers)
    t0 = time.monotonic()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = round(time.monotonic() - t0, 1)
    files = sorted(glob.glob(os.path.join(outdir, f"{prefix}_*.npy")))
    print(f"  [{prefix} seed={seed}] rc={rc} files={len(files)}/{count} in {wall}s "
          f"({count / wall:.2f} img/s)", flush=True)
    return {"prefix": prefix, "seed": seed, "count": count, "returncode": rc, "files": len(files),
            "wall_s": wall, "log": os.path.basename(log), "command": " ".join(cmd)}


def check_transforms() -> dict:
    out = subprocess.run(docker(["python", "/scripts/check_label_transforms.py", "/labels/_lens_labels.py"],
                                code_mounts()), capture_output=True, text=True)
    try:
        result = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"label-transform check did not run:\n{out.stdout}\n{out.stderr}")
    if out.returncode != 0 or not result.get("passed"):
        raise SystemExit("label-transform check FAILED; refusing to generate:\n" + json.dumps(result, indent=1))
    return result


def validate_rows(rows: list[dict]) -> dict:
    """Run every sampled system through the two-stage path's physical validator, SNR check on.

    Also compares its SNR estimate with each image's measured SNR, by Sersic index: the
    estimate scales amp (the brightness at R_sersic) by a constant calibrated on n = 1.
    """
    from dlens.schemas._lens_params import LensParameterSet
    from dlens.tools._param_validator import ValidationRanges, validate_parameters

    ranges = ValidationRanges()
    failed: dict[str, int] = {}
    passed = 0
    estimates = []
    for r in rows:
        params = LensParameterSet(
            z_lens=0.5, z_source=1.0, lens_model_list=["SIE"],
            kwargs_lens=[{t: r[t] for t in TARGETS}],
            source_model_list=["SERSIC_ELLIPSE"],
            kwargs_source=[{"amp": r["src_amp"], "R_sersic": r["src_R_sersic"], "n_sersic": r["src_n_sersic"],
                            "e1": r["src_e1"], "e2": r["src_e2"], "center_x": r["src_x"], "center_y": r["src_y"]}],
            kwargs_data={"numPix": 150, "deltaPix": 0.05, "exposure_time": r["exposure_time"],
                         "background_rms": 0.01},
            kwargs_psf={"psf_type": "GAUSSIAN", "fwhm": 0.087, "pixel_size": 0.05})
        res = validate_parameters(params, ranges)
        passed += res.passed
        estimates.append(res.snr_estimate)
        for name, ok in res.checks.items():
            if not ok:
                failed[name] = failed.get(name, 0) + 1
    est = np.array(estimates, dtype=float)
    meas = np.array([r["snr_measured"] for r in rows])
    n = np.array([r["src_n_sersic"] for r in rows])
    by_n = {}
    for lo, hi in N_BINS:
        sel = n_bin(n, lo, hi)
        by_n[f"{lo}-{hi}"] = {"systems": int(sel.sum()),
                              "snr_check_failed": int(((est[sel] < ranges.snr_min) | (est[sel] > ranges.snr_max)).sum()),
                              "median_estimate": float(np.median(est[sel])),
                              "median_measured": float(np.median(meas[sel]))}
    return {"systems": len(rows), "passed": passed, "failed_checks": failed,
            "snr_band": [ranges.snr_min, ranges.snr_max], "snr_estimate": column_stats(est),
            "snr_check_by_n_sersic": by_n}


def split_stats(rows: list[dict]) -> dict:
    stats = {}
    for col in REPORT_COLUMNS:
        values = np.array([r[col] for r in rows])
        entry = column_stats(np.log10(values) if col == "exposure_time" else values,
                             value_range=UNIFORM_COLUMNS.get(col) or LOG_UNIFORM_COLUMNS.get(col))
        if col == "exposure_time":
            entry["note"] = "log10 seconds"
        if col in UNIFORM_COLUMNS:
            lo, hi = UNIFORM_COLUMNS[col]
            entry["ks_uniform_D"], entry["ks_uniform_p"] = ks_uniform(values, lo, hi)
            entry["in_range"] = bool(values.min() >= lo and values.max() <= hi)
        if col in LOG_UNIFORM_COLUMNS:
            lo, hi = LOG_UNIFORM_COLUMNS[col]
            entry["ks_uniform_D"], entry["ks_uniform_p"] = ks_uniform(np.log(values), math.log(lo), math.log(hi))
            entry["ks_note"] = "uniformity of log(value)"
            entry["in_range"] = bool(values.min() >= lo and values.max() <= hi)
        stats[col] = entry
    for col in ("snr_measured", "peak_snr"):
        v = stats[col]
        values = np.array([r[col] for r in rows])
        v["in_band"] = float(np.mean((values >= SNR_BAND[0]) & (values <= SNR_BAND[1])))
        v["below_band"] = int((values < SNR_BAND[0]).sum())
        v["above_band"] = int((values > SNR_BAND[1]).sum())
        v["spearman"] = {c: spearman(values, [r[c] for r in rows])
                         for c in ("theta_E", "q", "src_n_sersic", "src_R_sersic", "exposure_time")}
        n = np.array([r["src_n_sersic"] for r in rows])
        v["median_by_n_sersic"] = {f"{lo}-{hi}": float(np.median(values[n_bin(n, lo, hi)])) for lo, hi in N_BINS}
    t = np.array([[r[c] for c in TARGETS] for r in rows])
    corr = np.corrcoef(t, rowvar=False)
    stats["target_correlation"] = {"columns": list(TARGETS), "matrix": np.round(corr, 3).tolist(),
                                   "max_abs_offdiag": float(np.max(np.abs(corr - np.eye(len(TARGETS)))))}
    frac = [math.hypot(r["src_x"] - r["center_x"], r["src_y"] - r["center_y"]) / r["theta_E"] for r in rows]
    stats["source_offset_over_theta_E"] = column_stats(np.array(frac), value_range=(0.0, 0.5))
    return stats


def print_stats(name: str, stats: dict) -> None:
    print(f"\n{name}: {stats['theta_E']['n']} images")
    print(f"  {'column':15s} {'min':>8s} {'p5':>8s} {'median':>8s} {'p95':>8s} {'max':>8s} {'std':>7s} "
          f"{'KS p (uniform)':>14s}  histogram")
    for col in (*REPORT_COLUMNS, "source_offset_over_theta_E"):
        s = stats[col]
        ks = f"{s['ks_uniform_p']:.3f}" if "ks_uniform_p" in s else "-"
        label = "log10 exposure" if col == "exposure_time" else col if len(col) <= 15 else "src |β|/θ_E"
        print(f"  {label:15s} {s['min']:8.3f} {s['p5']:8.3f} {s['median']:8.3f} {s['p95']:8.3f} "
              f"{s['max']:8.3f} {s['std']:7.3f} {ks:>14s}  {sparkline(s['hist'])}")
    print(f"  max |correlation| between targets: {stats['target_correlation']['max_abs_offdiag']:.3f}")
    for col in ("snr_measured", "peak_snr"):
        s = stats[col]
        corr = " ".join(f"{c} {v:+.2f}" for c, v in s["spearman"].items())
        by_n = " ".join(f"{k}: {v:.1f}" for k, v in s["median_by_n_sersic"].items())
        print(f"  {col}: {s['in_band']:.2%} in {list(SNR_BAND)} ({s['below_band']} below, {s['above_band']} above) | "
              f"rank corr: {corr}\n    median by n_sersic: {by_n}")


def bench(args: argparse.Namespace, out: str) -> int:
    runs = []
    for tag in ("a", "b"):
        runs.append(generate(os.path.join(out, f"bench_{tag}"), "bench", args.bench, BENCH_SEED,
                             args.workers, os.path.join(out, f"bench_{tag}.log")))
    if any(r["returncode"] or r["files"] != args.bench for r in runs):
        raise SystemExit(f"bench generation failed: {runs}")
    names = sorted(os.listdir(os.path.join(out, "bench_a")))
    same = all(sha256(os.path.join(out, "bench_a", n)) == sha256(os.path.join(out, "bench_b", n)) for n in names)
    rows = read_labels(os.path.join(out, "bench_a", "labels.csv"))
    snr = np.array([r["snr_measured"] for r in rows])
    print(f"\nBENCH {args.bench} images x2 with seed {BENCH_SEED}: byte-identical runs: {same}")
    print(f"  rate {args.bench / runs[0]['wall_s']:.2f} img/s with {args.workers} workers | measured SNR "
          f"p5 {np.percentile(snr, 5):.1f} median {np.median(snr):.1f} p95 {np.percentile(snr, 95):.1f}")
    return 0 if same else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--pool", type=int, default=4200, help="pool images (train + val)")
    p.add_argument("--test", type=int, default=1800, help="held-out test images")
    p.add_argument("--pool-seed", type=int, default=3001)
    p.add_argument("--test-seed", type=int, default=4001)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4, help="processes inside the one container")
    p.add_argument("--bench", type=int, default=0, help="timing + determinism run: N images, twice")
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    print("checking the flip/rotation label maps against lenstronomy ...", flush=True)
    transform_check = check_transforms()
    print(f"  passed: worst relative error {transform_check['worst_rel_err']:.1e} over "
          f"{transform_check['systems']} systems x {transform_check['transforms']} transforms; "
          f"negative control caught {transform_check['negative_control_caught']}", flush=True)
    if args.bench:
        return bench(args, out)

    seeds = (args.pool_seed, args.test_seed)
    if args.pool_seed == args.test_seed or RESERVED_SEEDS & set(seeds):
        raise SystemExit(f"pool and test seeds must differ and avoid {sorted(RESERVED_SEEDS)}")
    if os.path.exists(os.path.join(out, "raw_pool")):
        raise SystemExit(f"{out}/raw_pool exists; refusing to mix generations")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.monotonic()
    jobs = [generate(os.path.join(out, "raw_pool"), "pool", args.pool, args.pool_seed, args.workers,
                     os.path.join(out, "logs", "pool.log")),
            generate(os.path.join(out, "raw_test"), "test", args.test, args.test_seed, args.workers,
                     os.path.join(out, "logs", "test.log"))]
    wall = round(time.monotonic() - t0, 1)
    if any(j["returncode"] or j["files"] != j["count"] for j in jobs):
        raise SystemExit(f"generation incomplete: {jobs}")

    pool_rows = read_labels(os.path.join(out, "raw_pool", "labels.csv"))
    test_rows = read_labels(os.path.join(out, "raw_test", "labels.csv"))
    for name, rows, raw in (("pool", pool_rows, "raw_pool"), ("test", test_rows, "raw_test")):
        on_disk = {os.path.basename(f) for f in glob.glob(os.path.join(out, raw, "*.npy"))}
        if {r["file"] for r in rows} != on_disk or len(rows) != len(on_disk):
            raise SystemExit(f"{name}: labels.csv rows do not match the images on disk")

    # ---- manifest + overlap -----------------------------------------------------
    files = sorted(glob.glob(os.path.join(out, "raw_*", "*.npy")) + glob.glob(os.path.join(out, "raw_*", "labels.csv")))
    hashes = {os.path.relpath(f, out): sha256(f) for f in files}
    with open(os.path.join(out, "MANIFEST.sha256"), "w") as fh:
        for rel, h in hashes.items():
            fh.write(f"{h}  {rel}\n")
    image_h = {s: [h for r, h in hashes.items() if r.startswith(f"raw_{s}/") and r.endswith(".npy")]
               for s in ("pool", "test")}
    label_t = {s: {tuple(round(r[t], 12) for t in TARGETS) for r in rows}
               for s, rows in (("pool", pool_rows), ("test", test_rows))}
    hash_check = {
        "files": len(hashes),
        "test_pool_image_overlap": len(set(image_h["pool"]) & set(image_h["test"])),
        "duplicate_images_pool": len(image_h["pool"]) - len(set(image_h["pool"])),
        "duplicate_images_test": len(image_h["test"]) - len(set(image_h["test"])),
        "test_pool_label_overlap": len(label_t["pool"] & label_t["test"]),
    }
    print(f"manifest: {hash_check}", flush=True)
    if any(v for k, v in hash_check.items() if k != "files"):
        raise SystemExit("overlap or duplicates found; not splitting")

    # ---- split (labels travel with the images) ---------------------------------------
    train_rows, val_rows = split_rows(pool_rows, args.val_frac, args.split_seed)
    splits = {"train": (train_rows, "raw_pool"), "val": (val_rows, "raw_pool"), "test": (test_rows, "raw_test")}
    tsv = ["split_file\tgenerated_file"]
    for split, (rows, raw) in splits.items():
        os.makedirs(os.path.join(out, split), exist_ok=True)
        for r in rows:
            os.link(os.path.join(out, raw, r["file"]), os.path.join(out, split, r["file"]))
            tsv.append(f"{split}/{r['file']}\t{raw}/{r['file']}")
        write_labels(os.path.join(out, split, "labels.csv"), rows)
    with open(os.path.join(out, "SPLITS.tsv"), "w") as fh:
        fh.write("\n".join(tsv) + "\n")
    counts = {s: len(rows) for s, (rows, _) in splits.items()}

    # ---- checks + distributions ------------------------------------------------------
    validation = validate_rows(pool_rows + test_rows)
    print(f"validator (SNR check on): {validation['passed']}/{validation['systems']} pass; failed checks "
          f"{validation['failed_checks'] or 'none'}", flush=True)
    for k, v in validation["snr_check_by_n_sersic"].items():
        print(f"  n_sersic {k}: SNR check failed {v['snr_check_failed']}/{v['systems']}; median estimate "
              f"{v['median_estimate']:.1f} vs measured {v['median_measured']:.1f}", flush=True)
    stats = {s: split_stats(rows) for s, (rows, _) in splits.items()}
    json.dump(stats, open(os.path.join(out, "label_stats.json"), "w"), indent=1)
    for s in splits:
        print_stats(s, stats[s])

    # ---- provenance ----------------------------------------------------------------
    def sh(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    ranges = sh(docker(["python", "-c", "import sys; sys.path.insert(0, '/scripts'); "
                        "import json, lens_regression_sim as s; print(json.dumps(s.RANGES))"], code_mounts()))
    with open(os.path.join(out, "pip_freeze.txt"), "w") as fh:
        fh.write(sh(docker(["pip", "freeze"], {})) + "\n")
    code = {f: sha256(os.path.join(REPO, f)) for f in (
        "scripts/gen_regression_dataset.py", "scripts/gen_lens_regression_dataset.py",
        "scripts/lens_regression_sim.py", "scripts/check_label_transforms.py", "src/dlens/data/_lens_labels.py")}
    meta = {
        "started_utc": started, "generation_wall_s": wall,
        "task": "regression", "targets": list(TARGETS),
        "seeds": {"pool": args.pool_seed, "test": args.test_seed, "split": args.split_seed},
        "counts": {"pool": args.pool, "test": args.test, **counts},
        "sampling_ranges": json.loads(ranges) if ranges else None,
        "rules": {"split": f"seeded shuffle (seed {args.split_seed}) of pool rows sorted by file; "
                           f"first {args.val_frac:.0%} to val",
                  "chunks": "250 images per chunk; chunk c drawn from SeedSequence(seed).spawn()[c]"},
        "snr_definitions": {
            "snr_measured": "on the noisy image: (max - mean of the four 10 px corner boxes) / "
                            "sqrt(max + their variance); reproduces the validator's calibration on 900 real "
                            "Model_I images (p5/median/p95 14.8/18.0/21.8 vs recorded 14.8/18.1/22.0)",
            "peak_snr": "noiseless model peak counts / sqrt(peak counts + (background_rms * exposure)^2); "
                        "canonical Model_I parameters score median 14.5 (p5 12.6, p95 16.6) on it"},
        "label_transform_check": transform_check,
        "validator": validation,
        "hash_check": hash_check,
        "jobs": jobs,
        "image": {"tag": IMAGE, "id": sh(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE]),
                  "python": sh(docker(["python", "--version"], {}))},
        "code_sha256": code,
        "repo_commit": sh(["git", "-C", REPO, "rev-parse", "HEAD"]),
        "repo_dirty": bool(sh(["git", "-C", REPO, "status", "--porcelain"])),
        "host": {"docker": sh(["docker", "version", "-f", "{{.Server.Version}}"]),
                 "colima": sh(["colima", "version"]).splitlines()[:1] if shutil.which("colima") else None},
    }
    json.dump(meta, open(os.path.join(out, "dataset.json"), "w"), indent=2)
    print(f"\n{json.dumps(counts)} done in {wall}s of generation -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
