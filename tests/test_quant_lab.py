"""Gates for pipeline/quant_lab.py — the 8-step ML-in-the-loop research flow.

The load-bearing properties are:
- walk-forward folds never train on a period at or after the one they score (step 3)
- cross-sectional standardisation is period-local, so it cannot leak across time (step 2)
- the metric stack (Sharpe / Sortino / MaxDD / turnover) is arithmetically correct (step 8)
- a permuted target collapses the measured IC (leakage sentinel)

See docs/SPEC_QUANT_RESEARCH_LAB.md.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))

from quant_lab import (  # noqa: E402
    BENCH_NPZ,
    backtest,
    build_features,
    cross_sectional_z,
    fit_ridge,
    ic_ir,
    load_bench,
    make_signal,
    max_drawdown,
    portfolio_stats,
    predict_ridge,
    run_lab,
    select_universe,
    spearman_ic,
    walk_forward_folds,
)

# ------------------------------------------------------------------ fixtures


def _synthetic_bench(n_per_period: int = 60, periods: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019)) -> dict:
    """Small deterministic bench dict with a real linear signal in MOMENTUM_12_1."""
    rng = np.random.default_rng(7)
    names = ["MOMENTUM_12_1", "RET_1M", "ROE", "PE", "VOLUME_AVG_30D"]
    rows = len(periods) * n_per_period
    X = rng.normal(size=(rows, len(names)))
    y = 0.6 * X[:, 0] + 0.05 * rng.normal(size=rows)
    time_key = np.repeat(np.asarray(periods, dtype=np.int64), n_per_period)
    ticker = np.asarray([f"T{i % n_per_period:03d}" for i in range(rows)])
    return {
        "X": X,
        "X_mask": np.ones_like(X),
        "feature_names": names,
        "y": y,
        "y_mask": np.ones(rows, dtype=bool),
        "time_key": time_key,
        "ticker": ticker,
        "sector": np.asarray(["Tech"] * rows),
        "horizon_tdays": 126,
    }


# ------------------------------------------------- step 3: no future leakage


def test_walk_forward_folds_never_train_on_future():
    """The invariant the whole flow rests on: max(train period) < min(test period)."""
    periods = np.repeat(np.arange(2015, 2025), 5)
    folds = walk_forward_folds(periods)
    assert folds, "expected at least one fold"
    for tr, te in folds:
        assert periods[tr].max() < periods[te].min()


def test_walk_forward_folds_expand_and_cover_each_later_period():
    periods = np.repeat(np.arange(2015, 2025), 5)
    folds = walk_forward_folds(periods, min_train_periods=3)
    tested = [int(np.unique(periods[te])[0]) for _, te in folds]
    assert tested == list(range(2018, 2025))
    sizes = [int(tr.sum()) for tr, _ in folds]
    assert sizes == sorted(sizes), "training window must expand, never shrink"


def test_folds_have_disjoint_train_and_test_rows():
    periods = np.repeat(np.arange(2015, 2021), 4)
    for tr, te in walk_forward_folds(periods):
        assert not (tr & te).any()


# ------------------------------------- step 2: period-local standardisation


def test_cross_sectional_z_is_standardised_within_each_period():
    periods = np.repeat([2020, 2021], 50)
    values = np.concatenate([np.linspace(0, 10, 50), np.linspace(100, 200, 50)])
    observed = np.ones(100, dtype=bool)
    z = cross_sectional_z(values, observed, periods)
    for p in (2020, 2021):
        sel = periods == p
        assert abs(float(z[sel].mean())) < 1e-9
        assert abs(float(z[sel].std()) - 1.0) < 1e-6


def test_cross_sectional_z_does_not_leak_across_periods():
    """Rescaling one period must leave every other period's z-scores untouched."""
    periods = np.repeat([2020, 2021], 30)
    values = np.concatenate([np.linspace(1, 5, 30), np.linspace(1, 5, 30)])
    observed = np.ones(60, dtype=bool)
    base = cross_sectional_z(values, observed, periods)

    bumped = values.copy()
    bumped[periods == 2021] *= 1000.0
    after = cross_sectional_z(bumped, observed, periods)

    np.testing.assert_allclose(base[periods == 2020], after[periods == 2020])


def test_cross_sectional_z_unobserved_entries_are_zero():
    periods = np.zeros(10, dtype=np.int64)
    values = np.arange(10, dtype=float)
    observed = np.ones(10, dtype=bool)
    observed[:3] = False
    z = cross_sectional_z(values, observed, periods)
    assert np.all(z[:3] == 0.0)


def test_winsorization_caps_outliers():
    periods = np.zeros(200, dtype=np.int64)
    values = np.concatenate([np.zeros(199), [1e9]])
    z = cross_sectional_z(values, np.ones(200, dtype=bool), periods)
    assert float(np.abs(z).max()) <= 4.0 + 1e-9


# ---------------------------------------------------- step 4/5: model + IC


def test_ridge_recovers_a_known_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 2] + 0.01 * rng.normal(size=400)
    model = fit_ridge(X, y, lam=1e-6)
    assert model["coef"][0] > 0
    assert model["coef"][2] < 0
    assert abs(model["coef"][1]) < abs(model["coef"][0])
    assert spearman_ic(predict_ridge(model, X), y) > 0.95


def test_spearman_ic_is_plus_one_and_minus_one_at_the_extremes():
    a = np.arange(20, dtype=float)
    assert spearman_ic(a, a) == pytest.approx(1.0)
    assert spearman_ic(a, -a) == pytest.approx(-1.0)


def test_spearman_ic_nan_when_too_few_pairs():
    assert math.isnan(spearman_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_ic_ir_matches_hand_computation():
    ics = [0.10, 0.20, 0.30]
    out = ic_ir(ics)
    assert out["ic_mean"] == pytest.approx(0.2)
    assert out["ic_std"] == pytest.approx(0.1)
    assert out["ic_ir"] == pytest.approx(2.0)
    assert out["n_periods"] == 3


def test_ic_ir_ignores_nan_periods():
    out = ic_ir([0.1, float("nan"), 0.3])
    assert out["n_periods"] == 2


# ------------------------------------------------- step 8: the metric stack


def test_max_drawdown_hand_computed():
    # equity 1.10, 0.88, 0.924 ; peak 1.10 -> trough 0.88 -> dd = -0.20
    assert max_drawdown([0.10, -0.20, 0.05]) == pytest.approx(-0.20)


def test_max_drawdown_is_zero_when_never_negative():
    assert max_drawdown([0.01, 0.02, 0.03]) == pytest.approx(0.0)


def test_portfolio_stats_sharpe_sortino_hit_rate_hand_computed():
    bt = {
        "period_returns": [0.10, -0.20, 0.05],
        "gross_returns": [0.10, -0.20, 0.05],
        "turnovers": [1.0, 0.5, 0.5],
        "name_hit_rate": 0.5,
        "per_period": [],
    }
    s = portfolio_stats(bt, horizon_tdays=126)
    mean = (0.10 - 0.20 + 0.05) / 3.0
    sd = float(np.std([0.10, -0.20, 0.05], ddof=1))
    assert s["mean_period_return"] == pytest.approx(mean)
    assert s["sharpe_per_period"] == pytest.approx(mean / sd)
    # downside deviation uses only the negative period: sqrt(0.20**2) = 0.20
    assert s["sortino_per_period"] == pytest.approx(mean / 0.20)
    assert s["period_hit_rate"] == pytest.approx(2.0 / 3.0)
    assert s["max_drawdown"] == pytest.approx(-0.20)
    assert s["mean_turnover"] == pytest.approx(2.0 / 3.0)


def test_annualization_uses_declared_periods_per_year():
    bt = {
        "period_returns": [0.05, 0.02, 0.03, 0.04],
        "gross_returns": [0.05, 0.02, 0.03, 0.04],
        "turnovers": [1.0, 1.0, 1.0, 1.0],
        "name_hit_rate": 1.0,
        "per_period": [],
    }
    s = portfolio_stats(bt, horizon_tdays=126)
    assert s["periods_per_year"] == pytest.approx(2.0)
    assert s["sharpe_annualized"] == pytest.approx(s["sharpe_per_period"] * math.sqrt(2.0))
    assert "annualization_note" in s


def test_low_sample_warning_fires_under_eight_periods():
    few = {
        "period_returns": [0.01, 0.02, 0.03],
        "gross_returns": [0.01, 0.02, 0.03],
        "turnovers": [1.0, 1.0, 1.0],
        "name_hit_rate": 1.0,
        "per_period": [],
    }
    assert portfolio_stats(few, 126)["low_sample_warning"] is True

    many = dict(few)
    many["period_returns"] = [0.01] * 12
    many["turnovers"] = [1.0] * 12
    assert portfolio_stats(many, 126)["low_sample_warning"] is False


# -------------------------------------------------- step 7: backtest mechanics


def _two_period_book(second_scores):
    tickers = np.asarray(["A", "B", "C", "D"] * 2)
    periods = np.repeat([2020, 2021], 4)
    score = np.asarray([4.0, 3.0, 2.0, 1.0, *list(second_scores)])
    y = np.zeros(8)
    return score, y, periods, tickers


def test_turnover_is_zero_when_the_basket_is_unchanged():
    score, y, periods, tickers = _two_period_book([4.0, 3.0, 2.0, 1.0])
    bt = backtest(score, y, periods, tickers, top_n=2)
    assert bt["turnovers"][1] == pytest.approx(0.0)


def test_turnover_is_one_when_the_basket_is_disjoint():
    score, y, periods, tickers = _two_period_book([1.0, 2.0, 3.0, 4.0])
    bt = backtest(score, y, periods, tickers, top_n=2)
    assert bt["turnovers"][1] == pytest.approx(1.0)


def test_costs_reduce_net_return_in_proportion_to_turnover():
    score, _, periods, tickers = _two_period_book([1.0, 2.0, 3.0, 4.0])
    y = np.full(8, 0.10)
    free = backtest(score, y, periods, tickers, top_n=2, cost_bps=0.0)
    charged = backtest(score, y, periods, tickers, top_n=2, cost_bps=100.0)
    for i, per in enumerate(charged["per_period"]):
        expected = free["per_period"][i]["net_return"] - 0.01 * per["turnover"]
        assert per["net_return"] == pytest.approx(expected)


def test_long_short_nets_the_two_legs():
    tickers = np.asarray(["A", "B", "C", "D"])
    periods = np.asarray([2020] * 4)
    score = np.asarray([4.0, 3.0, 2.0, 1.0])
    y = np.asarray([0.20, 0.10, -0.05, -0.15])
    bt = backtest(score, y, periods, tickers, top_n=2, long_short=True)
    # long A,B = mean(0.20, 0.10) = 0.15 ; short C,D = mean(-0.05, -0.15) = -0.10
    assert bt["per_period"][0]["gross_return"] == pytest.approx(0.15 - (-0.10))


def test_backtest_picks_the_highest_scores():
    tickers = np.asarray(["A", "B", "C", "D"])
    periods = np.asarray([2020] * 4)
    score = np.asarray([1.0, 9.0, 2.0, 8.0])
    y = np.asarray([0.0, 1.0, 0.0, 1.0])
    bt = backtest(score, y, periods, tickers, top_n=2)
    assert bt["per_period"][0]["gross_return"] == pytest.approx(1.0)


# ---------------------------------------------------------- step 1 + step 6


def test_universe_selection_drops_low_coverage_rows():
    data = _synthetic_bench()
    data["X_mask"][:10, :] = 0.0
    keep = select_universe(data, min_coverage=0.5)
    assert not keep[:10].any()
    assert keep[10:].all()


def test_universe_selection_drops_missing_targets():
    data = _synthetic_bench()
    data["y_mask"][:5] = False
    data["y"][5:8] = np.nan
    keep = select_universe(data, min_coverage=0.0)
    assert not keep[:8].any()


def test_signal_is_period_local_and_standardised():
    pred = np.concatenate([np.linspace(0, 1, 20), np.linspace(50, 60, 20)])
    periods = np.repeat([2020, 2021], 20)
    sig = make_signal(pred, periods)
    for p in (2020, 2021):
        assert abs(float(sig[periods == p].mean())) < 1e-9


# ------------------------------------------------ end-to-end + leakage sentinel


def test_run_lab_on_synthetic_data_finds_the_planted_signal():
    data = _synthetic_bench(n_per_period=80)
    rep = run_lab(playbooks=["momentum"], top_n=10, data=data)
    assert rep["step5_validation"]["ic_mean"] > 0.5
    assert rep["step8_portfolio"]["n_periods"] > 0


def test_shuffled_target_destroys_the_planted_signal():
    """Leakage sentinel: permuting y within period must collapse IC toward zero."""
    data = _synthetic_bench(n_per_period=80)
    real = run_lab(playbooks=["momentum"], top_n=10, data=data)
    shuffled = run_lab(playbooks=["momentum"], top_n=10, shuffle_target=True, data=data)
    assert abs(shuffled["step5_validation"]["ic_mean"]) < 0.15
    assert real["step5_validation"]["ic_mean"] > abs(shuffled["step5_validation"]["ic_mean"]) * 3


def test_build_features_rejects_an_unknown_playbook():
    with pytest.raises(KeyError):
        build_features(_synthetic_bench(), ["not_a_playbook"])


@pytest.mark.skipif(not BENCH_NPZ.exists(), reason="bench npz not present")
def test_real_bench_folds_are_leakage_free():
    """Integration: the shipped corpus must produce leakage-free folds."""
    rep = run_lab(playbooks=["momentum"], top_n=40)
    assert rep["step3_folds"], "expected walk-forward folds on the real bench"
    for fold in rep["step3_folds"]:
        assert max(fold["train_periods"]) < fold["test_period"]
    assert rep["step8_portfolio"]["low_sample_warning"] is True


@pytest.mark.skipif(not BENCH_NPZ.exists(), reason="bench npz not present")
def test_real_bench_reports_the_inverted_mean_reversion_block():
    """The corpus shows continuation, not reversal, at the 126-tday horizon.

    Signs are held at their textbook definition, so this block reads as inverted rather
    than being silently re-fit. If this ever flips, the finding changed - update the spec.
    """
    data = load_bench()
    rep = run_lab(playbooks=["momentum", "mean_reversion"], data=data)
    assert rep["step5_playbook_ic"]["mean_reversion"]["inverted"] is True
    assert rep["step5_playbook_ic"]["momentum"]["ic_mean"] > 0
