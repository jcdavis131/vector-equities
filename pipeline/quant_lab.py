"""Quant Research Lab — the 8-step ML-in-the-loop flow over the committed equities bench.

Implements the research architecture from the @quantscience_ 2026-08-24 stack thread:

    universe -> features -> time-series CV (no leakage) -> model -> validation
    (IC, IC-IR, importance) -> signal -> backtest -> portfolio analysis

plus the metric stack (Sharpe, Sortino, MaxDD, hit-rate, turnover) and the
Momentum / Mean-Reversion / Quality / Value playbooks.

Design notes (see docs/SPEC_QUANT_RESEARCH_LAB.md):
- Ridge closed-form substitutes for XGBoost; a JSON report substitutes for MLflow.
  Both are deliberate zero-dependency swaps, not silent ones.
- Seasonality is NOT implemented: the corpus is annual company-FY, so no daily-bar
  playbook can be honestly backtested here.
- The test window is 3 periods. Sharpe over 3 points is not a meaningful estimate and
  the report says so via ``low_sample_warning``.

Usage:
    python3 pipeline/quant_lab.py --report
    python3 pipeline/quant_lab.py --playbook momentum --top-n 40 --long-short
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH_NPZ = ROOT / "bench" / "data" / "equities_bench_v1.npz"
ASSETS = ROOT / "assets"

WINSOR = 4.0
RIDGE_LAMBDA = 10.0
LOW_SAMPLE_PERIODS = 8

# Playbook -> (feature name, sign). sign -1 means "low raw value is the bullish side".
PLAYBOOKS: dict[str, list[tuple[str, float]]] = {
    "momentum": [
        ("RET_3M", 1.0),
        ("RET_6M", 1.0),
        ("RET_12M", 1.0),
        ("MOMENTUM_12_1", 1.0),
        ("PRICE_VS_52W_HIGH", 1.0),
    ],
    "mean_reversion": [
        ("RET_1M", -1.0),
        ("RSI_14_PROXY", -1.0),
    ],
    "quality": [
        ("ROE", 1.0),
        ("ROA", 1.0),
        ("ROIC", 1.0),
        ("FCF_ROIC", 1.0),
        ("ROIC_WACC_SPREAD", 1.0),
        ("ALTMAN_Z", 1.0),
        ("PIOTROSKI_F_SCORE_PROXY", 1.0),
    ],
    "value": [
        ("EARNINGS_YIELD", 1.0),
        ("FCF_YIELD", 1.0),
        ("PE", -1.0),
        ("PB", -1.0),
        ("EV_EBITDA", -1.0),
    ],
}


# ---------------------------------------------------------------- data loading


def load_bench(path: Path | None = None) -> dict:
    """Load the committed bench npz into a plain dict (step 0)."""
    src = Path(path) if path is not None else BENCH_NPZ
    z = np.load(src, allow_pickle=True)
    return {
        "X": np.asarray(z["X"], dtype=np.float64),
        "X_mask": np.asarray(z["X_mask"], dtype=np.float64),
        "feature_names": [str(s) for s in z["feature_names"].tolist()],
        "y": np.asarray(z["y_forward_return"], dtype=np.float64),
        "y_mask": np.asarray(z["mask_forward_return"], dtype=bool),
        "time_key": np.asarray(z["time_key"], dtype=np.int64),
        "ticker": np.asarray([str(s) for s in z["ticker"].tolist()]),
        "sector": np.asarray([str(s) for s in z["sector"].tolist()]),
        "horizon_tdays": int(np.asarray(z["horizon_tdays"]).ravel()[0]),
    }


# ------------------------------------------------------- step 1: universe


def select_universe(data: dict, min_coverage: float = 0.5, liquidity_pct: float = 0.0) -> np.ndarray:
    """Step 1 — universe selection. Returns a boolean keep-mask over rows.

    Drops rows whose forward return is missing/non-finite, or whose feature coverage
    (mean of X_mask across the row) falls below ``min_coverage``. Optionally keeps only
    names above a within-period liquidity percentile.
    """
    y = data["y"]
    keep = data["y_mask"] & np.isfinite(y)
    coverage = data["X_mask"].mean(axis=1)
    keep &= coverage >= min_coverage

    if liquidity_pct > 0.0:
        names = data["feature_names"]
        if "VOLUME_AVG_30D" in names:
            col = names.index("VOLUME_AVG_30D")
            vol = data["X"][:, col]
            observed = data["X_mask"][:, col] > 0
            for period in np.unique(data["time_key"]):
                in_p = (data["time_key"] == period) & keep & observed
                if in_p.sum() < 5:
                    continue
                cutoff = np.percentile(vol[in_p], liquidity_pct * 100.0)
                drop = in_p & (vol < cutoff)
                keep[drop] = False
    return keep


# ------------------------------------------------- step 2: feature engineering


def cross_sectional_z(values: np.ndarray, observed: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Z-score ``values`` within each period, using only observed entries.

    Unobserved entries return 0.0 (the repo's mask + zero-impute convention). Because
    every statistic is computed inside a single period's cross-section, this introduces
    no look-ahead across time.
    """
    out = np.zeros_like(values, dtype=np.float64)
    for period in np.unique(periods):
        sel = (periods == period) & observed
        n = int(sel.sum())
        if n < 2:
            continue
        v = values[sel]
        mu = v.mean()
        sd = v.std()
        if sd <= 0 or not np.isfinite(sd):
            continue
        out[sel] = np.clip((v - mu) / sd, -WINSOR, WINSOR)
    return out


def build_features(data: dict, playbooks: list[str]) -> tuple[np.ndarray, list[str]]:
    """Step 2 — feature engineering. Cross-sectionally z-scored playbook blocks."""
    names = data["feature_names"]
    cols: list[np.ndarray] = []
    used: list[str] = []
    for book in playbooks:
        if book not in PLAYBOOKS:
            raise KeyError(f"unknown playbook {book!r}; have {sorted(PLAYBOOKS)}")
        for feat, sign in PLAYBOOKS[book]:
            if feat not in names:
                continue
            idx = names.index(feat)
            observed = data["X_mask"][:, idx] > 0
            z = cross_sectional_z(data["X"][:, idx], observed, data["time_key"])
            cols.append(sign * z)
            used.append(f"{book}:{feat}")
    if not cols:
        raise ValueError("no playbook features resolved against the bench feature set")
    return np.column_stack(cols), used


# ------------------------------------------------ step 3: time-series CV folds


def walk_forward_folds(periods: np.ndarray, min_train_periods: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Step 3 — expanding walk-forward folds. Fold k trains on years <= Y, tests on Y+1.

    Every returned fold satisfies ``max(train period) < min(test period)``.
    """
    uniq = np.unique(periods)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(min_train_periods, len(uniq)):
        train_periods = uniq[:i]
        test_period = uniq[i]
        tr = np.isin(periods, train_periods)
        te = periods == test_period
        if tr.sum() == 0 or te.sum() == 0:
            continue
        folds.append((tr, te))
    return folds


# --------------------------------------------------- step 4: model training


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> dict:
    """Step 4 — ridge regression, closed form, standardised on TRAIN rows only."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    y_mu = float(y.mean())
    yc = y - y_mu

    n_feat = Xs.shape[1]
    gram = Xs.T @ Xs + lam * np.eye(n_feat)
    coef = np.linalg.solve(gram, Xs.T @ yc)
    return {"coef": coef, "mu": mu, "sd": sd, "intercept": y_mu}


def predict_ridge(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = (X - model["mu"]) / model["sd"]
    return Xs @ model["coef"] + model["intercept"]


# ------------------------------------------------------- step 5: validation


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of ``a`` (ties share the mean rank)."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


def spearman_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation. NaN when fewer than 3 usable pairs."""
    m = np.isfinite(pred) & np.isfinite(actual)
    if int(m.sum()) < 3:
        return float("nan")
    rx = rankdata(pred[m])
    ry = rankdata(actual[m])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def ic_ir(ics: list[float]) -> dict:
    """Step 5 — IC-IR = mean(IC) / std(IC) across periods."""
    vals = np.asarray([v for v in ics if np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_ir": float("nan"), "n_periods": 0}
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return {
        "ic_mean": float(vals.mean()),
        "ic_std": sd,
        "ic_ir": float(vals.mean() / sd) if sd > 0 else float("nan"),
        "n_periods": len(vals),
    }


def feature_importance(model: dict, feat_names: list[str]) -> list[dict]:
    """Step 5 — |coef| x train std, normalised to sum 1."""
    raw = np.abs(model["coef"]) * model["sd"]
    total = float(raw.sum())
    if total <= 0:
        weights = np.zeros_like(raw)
    else:
        weights = raw / total
    ranked = sorted(
        (
            {"feature": n, "weight": float(w), "coef": float(c)}
            for n, w, c in zip(feat_names, weights, model["coef"], strict=True)
        ),
        key=lambda d: -d["weight"],
    )
    return ranked


# ---------------------------------------------------- step 6: signal creation


def make_signal(pred: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Step 6 — cross-sectionally z-score predictions within each period."""
    observed = np.isfinite(pred)
    return cross_sectional_z(pred, observed, periods)


# ---------------------------------------------------------- step 7: backtest


def backtest(
    score: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    tickers: np.ndarray,
    top_n: int = 40,
    long_short: bool = False,
    cost_bps: float = 0.0,
) -> dict:
    """Step 7 — equal-weight top-N long (optionally minus bottom-N short), per period.

    Turnover is name-overlap based against the previous period's long basket, and the
    cost charged is ``cost_bps * turnover`` (in basis points of notional).
    """
    uniq = np.unique(periods)
    period_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    name_wins: list[bool] = []
    per_period: list[dict] = []
    prev_longs: set[str] = set()

    for period in uniq:
        sel = (periods == period) & np.isfinite(score) & np.isfinite(y)
        n = int(sel.sum())
        if n < 2:
            continue
        idx = np.flatnonzero(sel)
        order = idx[np.argsort(-score[idx], kind="mergesort")]
        k = max(1, min(top_n, len(order) // 2 if long_short else len(order)))

        longs = order[:k]
        long_ret = float(y[longs].mean())
        gross = long_ret
        if long_short:
            shorts = order[-k:]
            gross = long_ret - float(y[shorts].mean())

        long_names = {str(t) for t in tickers[longs]}
        if prev_longs:
            overlap = len(long_names & prev_longs)
            turnover = 1.0 - overlap / max(len(long_names), 1)
        else:
            turnover = 1.0
        prev_longs = long_names

        net = gross - (cost_bps / 1e4) * turnover
        gross_returns.append(gross)
        period_returns.append(net)
        turnovers.append(turnover)
        name_wins.extend([bool(v > 0) for v in y[longs]])
        per_period.append(
            {
                "period": int(period),
                "n_universe": n,
                "basket_size": int(k),
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
            }
        )

    return {
        "period_returns": period_returns,
        "gross_returns": gross_returns,
        "turnovers": turnovers,
        "name_hit_rate": float(np.mean(name_wins)) if name_wins else float("nan"),
        "per_period": per_period,
    }


# ------------------------------------------------- step 8: portfolio analysis


def max_drawdown(returns: list[float]) -> float:
    """Largest peak-to-trough decline of the compounded equity curve (negative)."""
    if not returns:
        return float("nan")
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def portfolio_stats(bt: dict, horizon_tdays: int) -> dict:
    """Step 8 — Sharpe, Sortino, MaxDD, hit-rate, turnover over the period returns."""
    r = np.asarray(bt["period_returns"], dtype=np.float64)
    n = len(r)
    if n == 0:
        return {"n_periods": 0, "low_sample_warning": True}

    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    downside = r[r < 0.0]
    dsd = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0

    periods_per_year = 252.0 / float(horizon_tdays)
    sharpe_pp = mean / sd if sd > 0 else float("nan")
    sortino_pp = mean / dsd if dsd > 0 else float("nan")

    return {
        "n_periods": n,
        "mean_period_return": mean,
        "std_period_return": sd,
        "cumulative_return": float(np.prod(1.0 + r) - 1.0),
        "sharpe_per_period": sharpe_pp,
        "sortino_per_period": sortino_pp,
        "sharpe_annualized": sharpe_pp * math.sqrt(periods_per_year) if np.isfinite(sharpe_pp) else float("nan"),
        "sortino_annualized": sortino_pp * math.sqrt(periods_per_year) if np.isfinite(sortino_pp) else float("nan"),
        "max_drawdown": max_drawdown(bt["period_returns"]),
        "period_hit_rate": float((r > 0).mean()),
        "name_hit_rate": bt["name_hit_rate"],
        "mean_turnover": float(np.mean(bt["turnovers"])) if bt["turnovers"] else float("nan"),
        "periods_per_year": periods_per_year,
        "annualization_note": (
            f"periods_per_year = 252/{horizon_tdays} = {periods_per_year:g}; annualized figures assume the "
            "signal is redeployed each holding period. Per-period figures are the measured quantity."
        ),
        "low_sample_warning": n < LOW_SAMPLE_PERIODS,
    }


# ------------------------------------------------------ walk-forward driver


def run_walk_forward(X: np.ndarray, y: np.ndarray, periods: np.ndarray) -> tuple[np.ndarray, list[dict], dict | None]:
    """Fit/predict across every expanding fold. Returns (oos_pred, fold_report, last_model)."""
    oos_pred = np.full(len(y), np.nan)
    fold_report: list[dict] = []
    last_model: dict | None = None

    for tr, te in walk_forward_folds(periods):
        model = fit_ridge(X[tr], y[tr])
        pred = predict_ridge(model, X[te])
        oos_pred[te] = pred
        last_model = model
        fold_report.append(
            {
                "train_periods": [int(v) for v in np.unique(periods[tr])],
                "test_period": int(np.unique(periods[te])[0]),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "ic": spearman_ic(pred, y[te]),
            }
        )
    return oos_pred, fold_report, last_model


def playbook_ic_breakdown(data: dict, keep: np.ndarray, y: np.ndarray, books: list[str]) -> dict:
    """Per-playbook out-of-sample IC, so a block with an inverted sign is visible.

    Signs are held at their textbook definition; they are never re-fit to observed IC,
    because choosing a sign after seeing full-sample IC is look-ahead bias.
    """
    periods = data["time_key"][keep]
    out: dict = {}
    for book in books:
        F, names = build_features(data, [book])
        _, folds, _ = run_walk_forward(F[keep], y, periods)
        stats = ic_ir([f["ic"] for f in folds])
        stats["n_features"] = len(names)
        stats["inverted"] = bool(np.isfinite(stats["ic_ir"]) and stats["ic_ir"] < -1.0)
        out[book] = stats
    return out


# --------------------------------------------------------------- the pipeline


def run_lab(
    playbooks: list[str] | None = None,
    top_n: int = 40,
    long_short: bool = False,
    cost_bps: float = 0.0,
    min_coverage: float = 0.5,
    liquidity_pct: float = 0.0,
    shuffle_target: bool = False,
    seed: int = 13,
    data: dict | None = None,
) -> dict:
    """Run all 8 steps and return a report dict.

    ``shuffle_target`` permutes y within each period — the leakage sentinel. A healthy
    pipeline collapses toward IC 0 under this permutation.
    """
    books = playbooks or ["momentum", "mean_reversion", "quality", "value"]
    d = data if data is not None else load_bench()

    keep = select_universe(d, min_coverage=min_coverage, liquidity_pct=liquidity_pct)
    F, feat_names = build_features(d, books)

    periods = d["time_key"][keep]
    X = F[keep]
    y = d["y"][keep]
    tickers = d["ticker"][keep]

    if shuffle_target:
        rng = np.random.default_rng(seed)
        y = y.copy()
        for period in np.unique(periods):
            sel = np.flatnonzero(periods == period)
            y[sel] = y[rng.permutation(sel)]

    oos_pred, fold_report, last_model = run_walk_forward(X, y, periods)

    scored = np.isfinite(oos_pred)
    ics = [f["ic"] for f in fold_report]
    validation = ic_ir(ics)
    signal = make_signal(oos_pred, periods)
    signal[~scored] = np.nan

    bt = backtest(
        signal[scored],
        y[scored],
        periods[scored],
        tickers[scored],
        top_n=top_n,
        long_short=long_short,
        cost_bps=cost_bps,
    )
    stats = portfolio_stats(bt, d["horizon_tdays"])

    return {
        "spec": "docs/SPEC_QUANT_RESEARCH_LAB.md",
        "source": "@quantscience_ 2026-08-24 quant stack thread (post 2091917484538372518)",
        "status": "candidate — research lane, not a shipped claim",
        "config": {
            "playbooks": books,
            "features": feat_names,
            "top_n": top_n,
            "long_short": long_short,
            "cost_bps": cost_bps,
            "min_coverage": min_coverage,
            "liquidity_pct": liquidity_pct,
            "ridge_lambda": RIDGE_LAMBDA,
            "shuffle_target": shuffle_target,
            "model": "ridge closed-form (numpy) — documented substitute for XGBoost",
        },
        "step1_universe": {
            "rows_total": len(keep),
            "rows_kept": int(keep.sum()),
            "periods": [int(v) for v in np.unique(periods)],
        },
        "step3_folds": fold_report,
        "step5_validation": validation,
        "step5_playbook_ic": playbook_ic_breakdown(d, keep, y, books),
        "step5_importance": feature_importance(last_model, feat_names)[:15] if last_model else [],
        "step7_backtest": bt["per_period"],
        "step8_portfolio": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Quant Research Lab — 8-step ML-in-the-loop")
    ap.add_argument("--playbook", action="append", choices=sorted(PLAYBOOKS), help="repeatable; default all four")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--long-short", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=0.0)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--liquidity-pct", type=float, default=0.0)
    ap.add_argument("--shuffle-target", action="store_true", help="leakage sentinel: permute y within period")
    ap.add_argument("--report", action="store_true", help="write assets/quant_lab_report.json")
    args = ap.parse_args()

    rep = run_lab(
        playbooks=args.playbook,
        top_n=args.top_n,
        long_short=args.long_short,
        cost_bps=args.cost_bps,
        min_coverage=args.min_coverage,
        liquidity_pct=args.liquidity_pct,
        shuffle_target=args.shuffle_target,
    )

    v = rep["step5_validation"]
    s = rep["step8_portfolio"]
    print(f"universe {rep['step1_universe']['rows_kept']}/{rep['step1_universe']['rows_total']} rows")
    print(f"folds {len(rep['step3_folds'])}  IC mean {v['ic_mean']:+.4f}  IC-IR {v['ic_ir']:+.3f}")
    print(
        f"portfolio n={s['n_periods']} cum {s['cumulative_return']:+.3f} "
        f"sharpe/period {s['sharpe_per_period']:+.3f} maxDD {s['max_drawdown']:+.3f} "
        f"hit {s['period_hit_rate']:.2f} turnover {s['mean_turnover']:.2f}"
    )
    if s.get("low_sample_warning"):
        print(f"WARNING low sample: {s['n_periods']} periods — Sharpe is not a meaningful estimate")

    if args.report:
        ASSETS.mkdir(parents=True, exist_ok=True)
        out = ASSETS / "quant_lab_report.json"
        out.write_text(json.dumps(rep, indent=2, default=float))
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
