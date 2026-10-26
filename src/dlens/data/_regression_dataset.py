# data/_regression_dataset.py
"""Helpers for the labelled SIE regression dataset: labels.csv I/O, the seeded
train/val split, hashing, and label-distribution statistics.

Each split directory holds ``<prefix>_NNNNN.npy`` images and one ``labels.csv`` with a
row per image (column ``file`` names the image). The regression targets are
``_lens_labels.TARGETS``; the other columns are conveniences (q, phi, HOLISMOKES IV
conventions) and nuisance parameters (source, exposure, peak SNR).
"""

from __future__ import annotations

import csv
import hashlib
import math
import random

import numpy as np

from dlens.data._lens_labels import TARGETS

# Columns kept as text; everything else in labels.csv is a float.
_TEXT_COLUMNS = ("file",)
_INT_COLUMNS = ("chunk",)


def read_labels(path: str) -> list[dict]:
    """Rows of a labels.csv, with numeric columns parsed."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: v if k in _TEXT_COLUMNS else int(v) if k in _INT_COLUMNS else float(v)
                         for k, v in row.items()})
    return rows


def write_labels(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def targets_array(rows: list[dict]) -> np.ndarray:
    """The regression targets as an array [N, 5] in ``TARGETS`` order."""
    return np.array([[row[t] for t in TARGETS] for row in rows], dtype=float)


def split_rows(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded shuffle of the rows sorted by file name; the first ``val_frac`` go to val."""
    ordered = sorted(rows, key=lambda r: r["file"])
    random.Random(seed).shuffle(ordered)
    n_val = int(len(ordered) * val_frac)
    by_name = lambda rs: sorted(rs, key=lambda r: r["file"])  # noqa: E731
    return by_name(ordered[n_val:]), by_name(ordered[:n_val])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ks_uniform(values: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """One-sample Kolmogorov-Smirnov test against U(lo, hi): (statistic D, asymptotic p-value)."""
    x = np.sort((np.asarray(values, dtype=float) - lo) / (hi - lo))
    n = len(x)
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def column_stats(values: np.ndarray, bins: int = 10, value_range: tuple[float, float] | None = None) -> dict:
    """Summary statistics and an equal-width histogram of one label column."""
    v = np.asarray(values, dtype=float)
    lo, hi = value_range if value_range is not None else (float(v.min()), float(v.max()))
    counts, _ = np.histogram(v, bins=bins, range=(lo, hi))
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return {"n": int(v.size), "min": float(v.min()), "p5": float(p5), "median": float(p50),
            "p95": float(p95), "max": float(v.max()), "mean": float(v.mean()), "std": float(v.std()),
            "hist_range": [lo, hi], "hist": counts.tolist()}


def sparkline(counts: list[int]) -> str:
    """A one-line text histogram."""
    marks = " ▁▂▃▄▅▆▇█"
    top = max(counts) or 1
    return "".join(marks[min(len(marks) - 1, round(c / top * (len(marks) - 1)))] for c in counts)
