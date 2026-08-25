# Vector Equities

![CI](https://github.com/jcdavis131/vector-equities/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

A daily equities "chimera" puzzle over 4,831 company-FYs (500 tickers, 2015-2024) embedding space: guess the ticker blend behind each day's composite.

Live at https://equities.dumbmodel.com

> Solo personal project, no connection to employer, built with public/free-tier only (free data pipeline, ONNX optional, static Vercel).

> **Picking up in-progress work?** Start at [`docs/HANDOFF.md`](docs/HANDOFF.md) — current state, training gates, verification, and open follow-ups.

## The embedding

4,831 company-FYs across 500 tickers (2015–2024) from SEC EDGAR XBRL CompanyFacts + market + 10-K text chunks (Item 1/1A/7, tables included). 17 towers over statement families (income, balance, cashflow, growth, profitability, leverage, efficiency, per-share, market, valuation, management, ownership, disclosure, sector, macro, form, bridge) fused by a 4-layer transformer (d_model 128, 4 heads, 4 layers) into 64-d L2-normalized company vector, per-FY z-scored + winsor ±4.

- **Architecture:** 17 × ResidualTower `cat([x·m,m])→96h→24d` skip → transformer fusion → 64-d, plus 384-d MiniLM wiki-text tower.
- **Skills:** 12 Financial Crafts (Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Momentum, Management Quality, Yield, Disclosure) percentile per FY.
- **Archetypes:** 8 k-means (Compounder, Cash_Cow, Turnaround, HyperGrowth_SaaS, Heavy_Industrial, Bank_Capital_Heavy, Moonshot_Bio, Serial_Acquirer).
- **Eval (see `assets/eval_sector_coherence.json`, `assets/eval_scoreboard.json`):**
  - knn sector purity@10 = 0.7057 (baseline random 0.1117, lift 6.32×, n=4831)
  - cross-ticker purity@10 = 0.4013 (baseline 0.1117, lift 3.59×, same-ticker excluded)
  - silhouette cosine = -0.0034 vs perm -0.0204
  - forward IC>0 gate (IC_rank 3m 0.0064, 12m 0.0062, triple-barrier hit 21.9%)
  - composite = sector coherence + next-profile R² + market directional

Shipped artifacts (`assets/real_data.json`, `assets/eval_sector_coherence.json`, `assets/real_data.json` points xyz + 12 grades) are committed so site runs static with optional client-side inference.

## The site

Plain HTML/JS/Canvas, no framework or game engine, PWA-capable (`sw.js`, `offline.html`). Pages: daily game (guess the ticker), 3D embedding map, company dossiers (10-K chunks), trends, sector explorer, model lab (`model.html`), methods. `index.html` loads `assets/real_data.json` + per-ticker 10-K chunk files. Cards localStorage-only, optional telemetry `api/telemetry.js` event-name-only.

## Data pipeline

```bash
python3 pipeline/fetch_sec_summary.py --limit 300      # SEC EDGAR CompanyFacts (free, User-Agent)
python3 pipeline/build_real_from_summary.py --limit 300
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py
python3 pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128
python3 pipeline/regen_assets.py                        # writes assets/real_data.json + eval
```

Sources: SEC EDGAR XBRL (free, no key, User-Agent), yfinance market, DEF 14A scaffolds. Every response cached under `pipeline/data/`; `--offline` rebuilds from cache. Gated by `tests/test_eval_sector_coherence.py` (>0.65 purity) + `tests/test_no_ticker_leakage.py`.

Three dormant data tracks (like hoops) — each cache-ready and gated on committed fixture until residential fetch:

- **Neo parser** — DEF 14A management compensation → ownership tower.

## Training

`train_mtnn.py` drives MTNN training (torch). Training: AdamW, OneCycle 10% warmup, InfoNCE same-ticker adjacent-FY + same-sector hard-negative boost 0.3, masked MSE, grad clip, best-checkpoint on composite (0.5*sector_acc + 0.5*purity). Promotion gated on sector purity >0.65 + cross-ticker >0.35 + forward IC>0 — transparent 14-d contract stays until candidate beats it. Research notes `docs/ARCHITECTURE.md`, `docs/TOWER_V6_DESIGN.md`.

### v6 transformer (shipped 2026-08-05, 500 tickers real)

- 17 towers → CLS + FY embedding 12-d + 17 tokens = 19 tokens transformer 128d 4L 4H → CLS 128→64 L2, shared `towers.py` ResidualTower.
- Losses: InfoNCE hybrid ticker 0.65 sector 0.35 + CORAL cov λ0.3
- Shipped eval: purity@10 0.7057, cross-ticker 0.4013, composite 0.65-0.70 range.

## Growth loop

`pipeline/update_dataset.py` growth loop: fetch -> rebuild -> gate -> ledger. Difficulty calibration is embedding-space guessability model targeting 40-80% expected-solve band (model estimate until telemetry qualifies it).

## Quant research lab (8-step ML-in-the-loop)

`pipeline/quant_lab.py` implements the research architecture from the @quantscience_
2026-08-24 quant-stack thread, over the committed `bench/data/equities_bench_v1.npz`
corpus (4,831 company-FYs, `horizon_tdays` 126). Spec: [`docs/SPEC_QUANT_RESEARCH_LAB.md`](docs/SPEC_QUANT_RESEARCH_LAB.md).

```bash
python3 pipeline/quant_lab.py --long-short --report   # writes assets/quant_lab_report.json
python3 pipeline/quant_lab.py --playbook momentum --long-short
python3 pipeline/quant_lab.py --shuffle-target        # leakage sentinel
```

Steps: universe selection -> playbook feature engineering (cross-sectionally z-scored,
winsor +/-4) -> expanding walk-forward CV -> ridge -> validation (IC, IC-IR, feature
importance) -> signal -> equal-weight top-N backtest -> portfolio analysis
(Sharpe, Sortino, MaxDD, hit-rate, turnover).

**Measured, candidate lane — nothing here promotes a shipped claim.** 7 walk-forward
periods, long/short top-40, no costs:

| Playbook | IC mean | IC-IR | note |
|---|---|---|---|
| momentum | +0.0272 | +0.269 | only positively-signed block |
| quality | -0.0095 | -0.097 | flat |
| value | -0.0184 | -0.323 | flat |
| mean_reversion | -0.1193 | **-2.369** | **inverted** — see below |

Blended long/short: cumulative +0.050, Sharpe/period +0.110, MaxDD -0.165, hit-rate
0.43, turnover 0.85 over n=7 periods. That is approximately no edge, and the report
says so: `low_sample_warning` fires below 8 periods, because a Sharpe over 7 points is
not a meaningful estimate.

Two honesty properties worth stating plainly:

- **Long-only here is just beta.** The long-only variant shows hit-rate 1.00 and zero
  drawdown, but the shuffled-target sentinel scores nearly the same (cum +1.002 vs
  +1.468) — the returns are market drift over 2015-2024, not signal. The long/short
  column above is the honest read.
- **Mean-reversion is inverted, and is left that way.** At the 126-trading-day horizon
  this corpus shows continuation, not reversal: `RET_1M` and `RSI_14_PROXY` keep going.
  The playbook holds its textbook sign and the report flags `inverted: true`. Re-fitting
  the sign to observed full-sample IC would be look-ahead bias, which is the exact
  failure step 3 exists to prevent.

Seasonality (the thread's third playbook) is **not** implemented: the corpus is annual
company-FY, so no daily-bar playbook can be honestly backtested here. Execution
(Prefect/IBKR) is out of scope, and no new dependency was added — ridge substitutes for
XGBoost, the JSON report substitutes for MLflow. Gated by `tests/test_quant_lab.py`.

## Running locally

```bash
python -m http.server 8000   # static, open http://localhost:8000
python -m pytest pipeline/ -q   # gates (needs dev extras)
```

## License

MIT. Solo personal project, no connection to employer, built with public/free-tier only.

## Deploy

Vercel static import; domains `equities.dumbmodel.com` and `equities.jcamd.com` redirect via `vercel.json` (cleanUrls true).

MIT. Solo personal project, no connection to employer, built with public/free-tier only.


## MTNN v4 — Forward IC Ledger — Day/Week/Month — 0.174→0.7057 Sector Coherence
> 4831 rows 500 tickers 11 sectors OKABE-8 DAX MSCI live provenance 7/7/0 59→73 hashes LCG 20260813→189831298 idx3820 triple[11205,19448,14209] ?daily=YYYYMMDD&n=1/3/5 same-link-same-stars
| Day 0.725 MAE 0.2085 IC 0.174 Sharpe1.22 n4831 sector_coherence 0.7057 |
| Week CQS0.72 IC0.22 Sharpe1.18 |
| Month CQS0.718 IC0.31 Sharpe1.25 |
