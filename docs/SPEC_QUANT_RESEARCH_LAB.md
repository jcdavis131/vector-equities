# SPEC — Quant Research Lab (8-step ML-in-the-loop)

Status: **candidate / research lane**. Nothing here promotes a shipped claim.
Source: @quantscience_ thread, 2026-08-24 (post `2091917484538372518`) — a quant
stack blueprint, not a trading strategy.

## What the source specifies

| Part | Source content | Disposition here |
|---|---|---|
| 1) Foundations | Python, Pandas, scikit-learn, SQLAlchemy | tooling; not a code change |
| 2) Data & Storage | yfinance (free), FMP (paid), DuckDB; raw ▶ cleaned ▶ features as separate tables | mapped to existing `pipeline/fetch_*` + `bench/data/*.npz` |
| 3) Research Lab | MLflow; playbooks Momentum / Mean-Reversion / Seasonality; metrics Sharpe, Sortino, MaxDD, hit-rate, turnover | **built** — playbooks + metric stack |
| 4) ML in the Loop | 8-step: universe → features → time-series CV (no leakage) → model → validation (IC, IC-IR, importance) → signal → backtest → portfolio analysis | **built** — `pipeline/quant_lab.py` |
| 5) Execution | Prefect, IBKR, daily fetch→score→allocate→trade→log, guardrails | **out of scope** — see Non-goals |

## Non-goals (deliberate)

- **No live execution.** No IBKR, no broker adapter, no order routing. This repo is a
  static research site; a live trading path is a different risk surface entirely.
- **No new heavy dependencies.** MLflow / Prefect / DuckDB / Polars / XGBoost are not
  added. `pyproject.toml` already carries torch + sklearn; adding an orchestration and
  tracking stack for one research module is pin churn against `dependency-hygiene`.
  Ridge (closed-form, numpy) substitutes for XGBoost; the JSON report substitutes for
  the MLflow run table. Both are documented swaps, not silent ones.
- **No daily-bar strategies.** The committed corpus is annual company-FY
  (`time_key` 2015–2024, `horizon_tdays` 126). Anything requiring daily bars —
  turn-of-month, intraday, stop rules — is unbuildable here and is not faked.
  The Seasonality playbook is therefore **not implemented**; only Momentum,
  Mean-Reversion, Quality and Value are.

## Data contract

Input `bench/data/equities_bench_v1.npz`:

```
X (4831, 85) float32      X_mask (4831, 85) float32
y_forward_return (4831,)  mask_forward_return (4831,) bool   # 4788 usable
time_key (4831,)          # fiscal year 2015..2024
ticker, sector, entity_id
horizon_tdays = 126       # forward holding period, trading days
```

## The 8 steps, as implemented

1. **Universe selection** — drop rows with non-finite target or row coverage
   (`X_mask` row mean) below `min_coverage`. Optionally keep the top liquidity
   percentile by `VOLUME_AVG_30D`. Reported as an audit trail, not a silent filter.
2. **Feature engineering** — playbook blocks over existing columns:
   - `momentum`: RET_3M, RET_6M, RET_12M, MOMENTUM_12_1, PRICE_VS_52W_HIGH
   - `mean_reversion`: RET_1M, RSI_14_PROXY (sign-flipped — short-term reversal)
   - `quality`: ROE, ROA, ROIC, FCF_ROIC, ROIC_WACC_SPREAD, ALTMAN_Z, PIOTROSKI_F_SCORE_PROXY
   - `value`: EARNINGS_YIELD, FCF_YIELD, PE, PB, EV_EBITDA (PE/PB/EV_EBITDA sign-flipped — cheap is high)
   Each feature is **cross-sectionally** z-scored within its own `time_key` and
   winsorized to ±4. Cross-sectional standardisation uses only contemporaneous data,
   so it introduces no look-ahead.
3. **Time-series CV (no leakage)** — expanding walk-forward over fiscal years. Fold *k*
   trains on years `<= Y_k` and validates on `Y_k + 1`. Invariant asserted in test:
   `max(train_year) < min(val_year)` for every fold. No shuffling, no purge gap needed
   (annual, non-overlapping).
4. **Model training** — ridge regression, closed form
   `w = (XᵀX + λI)⁻¹ Xᵀy`, fit on train folds only. Feature means/stds for the model
   matrix come from **train rows only**. Baseline: predict train mean.
5. **Validation** — Spearman rank IC per period; `IC-IR = mean(IC) / std(IC)`;
   feature importance = `|coef| × train std`, normalised to sum 1.
6. **Signal creation** — predictions cross-sectionally z-scored within period → score.
7. **Backtest** — per period, equal-weight long top-N by score; optional short bottom-N.
   Period return = mean realised `y_forward_return` of the basket, minus
   `cost_bps × turnover`. Turnover = `1 − |A ∩ B| / N` vs previous period's names.
8. **Portfolio analysis** — Sharpe, Sortino (MAR = 0), MaxDD on the compounded equity
   curve, period hit-rate, name-level hit-rate, mean turnover, cumulative return.

## Honesty gates

- `n_periods` is reported on every stat block. The test window is **3 periods**
  (2022, 2023, 2024). A Sharpe over 3 points is not a statistically meaningful
  estimate, and the report carries `low_sample_warning: true` whenever `n_periods < 8`.
- Annualisation uses `periods_per_year = 252 / horizon_tdays = 2.0` and is emitted as
  `sharpe_annualized` **alongside** `sharpe_per_period`, never instead of it, with
  `annualization_note` stating the assumption.
- A shuffled-target sentinel test asserts |IC| stays small when `y` is permuted within
  period — if that ever fails, the pipeline is leaking.

## Acceptance criteria

- [ ] `python3 pipeline/quant_lab.py --report` writes `assets/quant_lab_report.json`
- [ ] every walk-forward fold satisfies `max(train_year) < min(val_year)`
- [ ] shuffled-target IC-IR collapses toward 0 relative to the real run
- [ ] metric arithmetic (Sharpe / Sortino / MaxDD / turnover) matches hand-computed fixtures
- [ ] `pytest tests/test_quant_lab.py` green
- [ ] no new entries in `pyproject.toml` dependencies
