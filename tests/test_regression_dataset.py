"""labels.csv I/O, the seeded split, and the label-distribution statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlens.data._lens_labels import TARGETS
from dlens.data._regression_dataset import (column_stats, ks_uniform, read_labels, sha256, sparkline,
                                            split_rows, targets_array, write_labels)


def _rows(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return [{"file": f"pool_{i:05d}.npy", "theta_E": rng.uniform(0.5, 2.5), "e1": rng.normal(0, 0.1),
             "e2": rng.normal(0, 0.1), "center_x": rng.uniform(-0.3, 0.3), "center_y": rng.uniform(-0.3, 0.3),
             "peak_snr": rng.uniform(10, 50), "chunk": i // 10} for i in range(n)]


def test_labels_round_trip(tmp_path: Path):
    rows = _rows(7)
    write_labels(str(tmp_path / "labels.csv"), rows)
    back = read_labels(str(tmp_path / "labels.csv"))
    assert [r["file"] for r in back] == [r["file"] for r in rows]
    assert all(isinstance(r["chunk"], int) for r in back)
    assert np.array_equal(targets_array(back), targets_array(rows))  # repr() round-trips floats


def test_targets_array_order():
    arr = targets_array(_rows(3))
    assert arr.shape == (3, len(TARGETS))
    assert arr[1, TARGETS.index("center_y")] == _rows(3)[1]["center_y"]


def test_split_is_seeded_disjoint_and_complete():
    rows = _rows(50)
    train, val = split_rows(rows, 0.2, seed=0)
    assert len(train) == 40 and len(val) == 10
    names = lambda rs: [r["file"] for r in rs]  # noqa: E731
    assert not set(names(train)) & set(names(val))
    assert sorted(names(train) + names(val)) == names(rows)
    again = split_rows(list(reversed(rows)), 0.2, seed=0)  # input order doesn't matter
    assert names(again[1]) == names(val)
    assert names(split_rows(rows, 0.2, seed=1)[1]) != names(val)


def test_ks_uniform_tells_uniform_from_not():
    rng = np.random.default_rng(0)
    d, p = ks_uniform(rng.uniform(0.5, 2.5, 4000), 0.5, 2.5)
    assert p > 0.01 and d < 0.03
    _, p_skewed = ks_uniform(0.5 + 2.0 * rng.uniform(size=4000) ** 1.3, 0.5, 2.5)
    assert p_skewed < 1e-6


def test_ks_uniform_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    x = np.random.default_rng(3).uniform(-0.3, 0.3, 500)
    d, p = ks_uniform(x, -0.3, 0.3)
    ref = stats.kstest(x, "uniform", args=(-0.3, 0.6), method="asymp")
    assert d == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue, abs=0.02)  # Stephens' small-n correction vs plain asymptotic


def test_column_stats_and_sparkline(tmp_path: Path):
    s = column_stats(np.arange(100.0), bins=4, value_range=(0, 100))
    assert s["n"] == 100 and s["min"] == 0 and s["max"] == 99 and s["hist"] == [25, 25, 25, 25]
    assert s["median"] == pytest.approx(49.5)
    assert sparkline([0, 5, 10]) == " ▄█"
    (tmp_path / "f").write_bytes(b"abc")
    assert sha256(str(tmp_path / "f")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
