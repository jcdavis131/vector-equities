"""Leakage-safety regression tests for bench/ (the real-data multi-target harness).

bench/ carries the repo's leakage-critical logic (point-in-time feature audit,
forward-shifted label construction, train-only preprocessing statistics) but
had zero test coverage before this file: pytest's ``testpaths`` never included
``bench/``, and the module has no unit tests of its own. These tests import
the pure functions directly (bench scripts are plain modules, not a package)
and check the invariants the module docstrings promise, so a future edit that
quietly breaks anchor/window arithmetic or leaks test-row statistics into
train-row features fails CI instead of silently corrupting a benchmark run.

No network access, no git blob reads, no model training: everything here is
synthetic and deterministic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build_dataset = _load_module("_bench_build_dataset", "bench/build_dataset.py")
run_benchmark = _load_module("_bench_run_benchmark", "bench/run_benchmark.py")


# ---------------------------------------------------------------------------
# forward_labels: anchor selection + forward-only window arithmetic
# ---------------------------------------------------------------------------


def _bdates(start: str, periods: int) -> np.ndarray:
    return np.array(pd.bdate_range(start=start, periods=periods).astype(str).tolist())


def test_forward_labels_none_when_insufficient_future_history():
    """Anchor + 126 trading days must fit inside the price series, or skip."""
    dates = _bdates("2016-06-01", 100)  # far fewer than anchor(~22) + 126
    adj = np.arange(1.0, 101.0)
    assert build_dataset.forward_labels(dates, adj, fy=2015) is None


def test_forward_labels_none_when_no_trading_day_near_anchor():
    """If the nearest indexed date isn't actually in July fy+1 (e.g. a long
    trading halt straddling the anchor), the row must be skipped rather than
    silently anchored to a wrong date."""
    before = _bdates("2016-01-01", 100)  # ends well before July
    after = _bdates("2016-09-01", 200)  # resumes in September
    dates = np.concatenate([before, after])
    adj = np.arange(1.0, float(len(dates)) + 1.0)
    assert build_dataset.forward_labels(dates, adj, fy=2015) is None


def test_forward_labels_monotonic_increasing_price_has_zero_drawdown():
    """A strictly increasing price path is always at a new high, so the
    forward max drawdown over the window must be exactly 0."""
    dates = _bdates("2016-06-01", 300)
    adj = 100.0 + np.arange(300, dtype=float)  # strictly increasing
    ret, vol, dd = build_dataset.forward_labels(dates, adj, fy=2015)
    assert dd == pytest.approx(0.0, abs=1e-12)
    assert ret > 0.0


def test_forward_labels_monotonic_decreasing_price_drawdown_matches_endpoints():
    """A strictly decreasing price path never sets a new high after t+1, so
    the running peak stays pinned at the first future price and the max
    drawdown is exactly (last - first) / first."""
    dates = _bdates("2016-06-01", 300)
    adj = 500.0 - 0.5 * np.arange(300, dtype=float)  # strictly decreasing, stays positive
    ret, vol, dd = build_dataset.forward_labels(dates, adj, fy=2015)

    anchor = int(np.searchsorted(dates, "2016-07-01", side="left"))
    future = adj[anchor + 1 : anchor + build_dataset.HORIZON_TDAYS + 1]
    expected_dd = (future[-1] - future[0]) / future[0]
    assert dd == pytest.approx(expected_dd, rel=1e-9)
    assert ret < 0.0


def test_forward_labels_constant_daily_return_has_zero_realized_vol():
    """A pure geometric-growth path has a constant daily simple return, so
    the realized-vol label (std of daily returns) must be ~0."""
    dates = _bdates("2016-06-01", 300)
    r = 0.001
    adj = 100.0 * (1.0 + r) ** np.arange(300, dtype=float)
    _ret, vol, _dd = build_dataset.forward_labels(dates, adj, fy=2015)
    assert vol == pytest.approx(0.0, abs=1e-6)


def test_forward_labels_return_uses_only_anchor_and_horizon_endpoint():
    """forward_return must equal adjclose[t+126]/adjclose[t] - 1 exactly,
    not some off-by-one neighbor (a classic leakage/lookback bug)."""
    dates = _bdates("2016-06-01", 300)
    adj = np.arange(1.0, 301.0)
    anchor = int(np.searchsorted(dates, "2016-07-01", side="left"))
    ret, _vol, _dd = build_dataset.forward_labels(dates, adj, fy=2015)
    expected = adj[anchor + build_dataset.HORIZON_TDAYS] / adj[anchor] - 1.0
    assert ret == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# snapshot_audit: drop features that are constant across a ticker's own years
# ---------------------------------------------------------------------------


def test_snapshot_audit_flags_per_ticker_constant_feature():
    """A feature fetched once and copied to every FY row of a ticker (the
    documented 'snapshot' leakage pattern) must get const_frac == 1.0."""
    tickers = np.array(["AAA"] * 5 + ["BBB"] * 5)
    snapshot_col = np.array([1.0] * 5 + [2.0] * 5)  # constant within each ticker
    varying_col = np.arange(10, dtype=float)  # changes every year
    Zr = np.column_stack([snapshot_col, varying_col])

    const_frac = build_dataset.snapshot_audit(Zr, tickers)

    assert const_frac[0] == pytest.approx(1.0)
    assert const_frac[1] == pytest.approx(0.0)


def test_snapshot_audit_ignores_tickers_with_too_few_observations():
    """Tickers with < 3 observed years don't count toward const_frac (too
    little evidence either way), matching the documented rule."""
    tickers = np.array(["AAA", "AAA", "BBB", "BBB"])  # both < 3 obs
    Zr = np.array([[1.0], [1.0], [2.0], [3.0]])
    const_frac = build_dataset.snapshot_audit(Zr, tickers)
    assert const_frac[0] == 0.0  # denominator (n_tick) is 0 -> guarded to 0, not NaN


# ---------------------------------------------------------------------------
# standardize_train_only: harness-train statistics must not see test rows
# ---------------------------------------------------------------------------


def test_standardize_train_only_is_unaffected_by_test_row_values():
    """Perturbing test-only rows (any magnitude) must not change the
    standardized output of train rows: mean/std/impute-median must be
    computed strictly from harness_train_rows."""
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 40, 15, 6
    X = rng.normal(size=(n_train + n_test, n_feat)).astype(np.float64)
    mask = np.ones_like(X, dtype=np.float32)
    harness_train = np.zeros(n_train + n_test, dtype=bool)
    harness_train[:n_train] = True

    out_a = run_benchmark.standardize_train_only(X.copy(), mask.copy(), harness_train)

    X_perturbed = X.copy()
    X_perturbed[n_train:] = X_perturbed[n_train:] * 1e6 + 999.0  # blow up test rows only
    out_b = run_benchmark.standardize_train_only(X_perturbed, mask.copy(), harness_train)

    np.testing.assert_allclose(out_a[:n_train], out_b[:n_train], rtol=1e-6, atol=1e-6)


def test_standardize_train_only_imputes_missing_with_train_median_only():
    """A test row with a fully-missing feature must be imputed with the
    train-split median, never a statistic that includes test rows."""
    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],  # train median = 2.0
            [1000.0],  # test row, would skew a global median to ~250
        ]
    )
    mask = np.array([[1.0], [1.0], [1.0], [0.0]])  # test row's feature is missing
    harness_train = np.array([True, True, True, False])

    out = run_benchmark.standardize_train_only(X, mask, harness_train)
    mu, sd = X[harness_train].mean(), X[harness_train].std()
    expected_imputed_standardized = np.clip((2.0 - mu) / sd, -5.0, 5.0)
    assert out[3, 0] == pytest.approx(expected_imputed_standardized, rel=1e-6)
