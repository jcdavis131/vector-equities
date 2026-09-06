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
  - forward IC>0 gate: **not passing.** `assets/eval_forward.json` (written by
    `pipeline/eval_forward.py`, Spearman rank IC of the calibrated 6M prediction against
    realized forward return on `trades_final_ranked_v6.csv`) reads ic_rank_6m -0.0166,
    3m -0.0151, 12m -0.0146, triple-barrier hit 21.9%, `gate.ic_gt_zero: false`. The
    3m 0.0064 / 12m 0.0062 previously listed here are not in any committed report.
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

> **Unverified (2026-09-05).** No script in this repo produces the IC 0.174 / 0.22 / 0.31
> figures below, and `pipeline/eval_reports/eval_equities_latest.json` carries an
> `ic_proxy` of 5.827, which is outside the [-1, 1] range an information coefficient can
> take, so that file is not an IC measurement either. The only reproducible forward-IC
> number is the negative one in `assets/eval_forward.json` above. Treat this table as a
> claim until `eval_forward.py` is rerun on the v4 trades.
> 4831 rows 500 tickers 11 sectors OKABE-8 DAX MSCI live provenance 7/7/0 59→73 hashes LCG 20260813→189831298 idx3820 triple[11205,19448,14209] ?daily=YYYYMMDD&n=1/3/5 same-link-same-stars
| Day 0.725 MAE 0.2085 IC 0.174 Sharpe1.22 n4831 sector_coherence 0.7057 |
| Week CQS0.72 IC0.22 Sharpe1.18 |
| Month CQS0.718 IC0.31 Sharpe1.25 |
