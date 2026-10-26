"""Generate labelled SIE regression images (runs INSIDE the dlens-lenstronomy container).

Writes <out>/<prefix>_NNNNN.npy (float32, Model_I-style noisy counts) and one labels.csv
with a row per image. The images are split into fixed chunks of --chunk images, and
chunk c is drawn from SeedSequence(seed).spawn(...)[c], so the output depends only on
--seed, --count and --chunk, never on --workers.

Usage (in container):
    python gen_lens_regression_dataset.py OUT --count N --seed S [--prefix pool] [--chunk 250] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import lens_regression_sim as sim


def run_chunk(args: tuple) -> list[dict]:
    out, prefix, start, n, entropy, chunk = args
    child = np.random.SeedSequence(entropy).spawn(chunk + 1)[chunk]
    rng = np.random.default_rng(child)
    np.random.seed(int(child.generate_state(1)[0]))  # lenstronomy's noise uses the global state
    rows = []
    for i in range(start, start + n):
        system = sim.sample(rng)
        model = sim.image_model(system["kwargs_lens"], system["kwargs_source"], system["exposure_time"])
        img = sim.observe(model, system["exposure_time"]).astype(np.float32)
        if img.shape != (sim.NUM_PIX, sim.NUM_PIX) or not np.isfinite(img).all():
            raise RuntimeError(f"bad image at index {i}")
        name = f"{prefix}_{i:05d}.npy"
        np.save(os.path.join(out, name), img)
        rows.append({"file": name, **sim.labels(system),
                     "peak_snr": sim.peak_snr(model, system["exposure_time"]), "chunk": chunk})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--prefix", default="lens")
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tasks = [(args.out, args.prefix, s, min(args.chunk, args.count - s), args.seed, c)
             for c, s in enumerate(range(0, args.count, args.chunk))]
    t0 = time.time()
    rows: list[dict] = []
    with Pool(args.workers) as pool:
        for chunk_rows in pool.imap(run_chunk, tasks):
            rows.extend(chunk_rows)
            print(f"[{args.prefix}] {len(rows)}/{args.count} ({len(rows) / (time.time() - t0):.2f} img/s)",
                  flush=True)
    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{args.prefix}] DONE {len(rows)}/{args.count} in {time.time() - t0:.0f}s", flush=True)
    return 0 if len(rows) == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
