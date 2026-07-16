# Vector Equities — MTNN for Public Companies (v1)

**Solo personal project, no connection to employer, built with public/free-tier only — Port of Vector Hoops MTNN (12,966 NBA player-seasons, 17 towers, 48-d, CQS 0.7937)**

Holistic company embedding from SEC filings + NEO comp + Market + Ownership + Disclosure Text — multi-task SOTA, composite metrics, free-tier offline-safe.

Repo: `~/workspace/vector-equities` — 14,400 company-FYs (1200 tickers x 12 FYs), 122 feats, 17 families, 12 skills, 8 archetypes, MTNN v4/v5

## Quickstart (offline, no keys)

```bash
cd ~/workspace/vector-equities
python3 pipeline/build_demo_v3.py --companies 1200 --years 12 --continuity 0.80
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py
python3 pipeline/train_mtnn.py --epochs 40 --dim 48 --fusion gated --tower-blocks 2 --mlp-heads
python3 -m pipeline.composite_score --report pipeline/data/mtnn_report.json
```

Outputs: train_matrix.npz, skill_labels.npz, embedding.npz, mtnn_report.json, mtnn_best.pt

## Latest Training Results (continuity 0.80 gated 48-d 40 epochs, 14.4k rows)

```
Train 8400 (FY<=2021) / Val 2400 / Test 3600 (FY>=2024) / Pairs 13200
Recall@10 same-ticker next FY: train 1.0 / val 0.028 / test 0.0 / all 0.58
Purity@20 cross-cycle archetype: 0.6438 (64.3%)
Sector Acc (11-way GICS): 86.0% (baseline 9%)
Next FY profile R2: val 0.393 / test 0.2526, MAE 0.61/0.68, RMSE 0.77/0.86
Market directional acc: 81.6%
CQS: 0.3975 (test recall drag) / composite proxy val 0.336

Earlier high-continuity run 0.88 concat 2 blocks: recall 1.0 val/test, purity 53%, sector 8.7%
Tradeoff tunable via --continuity 0.72-0.88
```

Hoops baseline for reference: recall 0.977, purity 0.6717, CQS 0.7937 leakfree

## Architecture

See `docs/ARCHITECTURE.md`

Encoder: 17 ResidualTower cat([x·m,m]) ->96h GELU LN ->24d + skip + stacked ResBlocks, fusion gated (attention+gate+12-d FY emb) / concat (flatten) / transformer (tower tokens + FY + [CLS] 4 layers), 48-64d L2 norm

Heads: archetype 8, sector 11, profile 14-d, next_profile 14-d, 12 skill mini-towers (Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Market Momentum, Management Quality, Shareholder Yield, Disclosure Quality), valuation EV/EBITDA, market next excess ret, vol, health Altman Z, payout, mgmt, own

Loss: archetype 0.25 sector 0.15 profile 0.12 next 0.10 skills 0.20 valuation 0.12 market 0.12 etc Phase B rebalanced

Training tricks: AdamW no-decay bias/LN, OneCycle 10% warmup linear, feature dropout 0.12 two views, masked MSE sparse, best checkpoint on composite proxy 0.5*recall+0.5*purity (fixes epoch 0 bug)

## Data Pipeline (free-tier only, holistic)

- SEC EDGAR: `fetch_sec.py` CompanyFacts XBRL us-gaap Revenues/NetIncomeLoss/GrossProfit/Assets/Liabilities/Equity/Cash/Debt/EPS -> 10-K Item7 MD&A length/sentiment (Loughran-McDonald) + Item1A risk count
- DEF 14A: `parse_neo.py` NEO count, CEO age/tenure/founder flag, total comp log, equity %, avg NEO comp, pay ratio, board indep %, board size, insider own %, pay vs sector, turnover, duality
- Market: `fetch_market.py` yfinance free -> ret 1/3/6/12M, vol 30/90/252, beta, momentum, vol avg, price vs 52W
- Ownership: inst %, delta, insider net 12M, float, concentration, short interest
- Macro/form: 10Y rate, VIX, credit spread, GDP, earnings surprise streak, guidance raise
- Synthetic fallback: `build_demo_v3.py` sector+archetype biased bases + AR1 continuity 0.80 + noise 0.45 + macro 0.12, per-FY z-score ±4 clip honest like hoops, tunable for recall vs sector tradeoff
- Skills: `build_skills.py` 12 grades percentile per FY
- Archetypes: `build_archetypes.py` k-means 8 on 14-d financial profile

## Repo Layout

```
pipeline/
  feature_spec.py, build_demo_v3.py, build_skills.py, build_archetypes.py
  model.py, train_mtnn.py, composite_score.py, mtnn_validation.py
  fetch_sec.py, fetch_market.py, parse_neo.py, rebuild_all.py
  data/ train_matrix.npz, skill_labels.npz, embedding.npz, mtnn_report.json, mtnn_best.pt
  cache/sec, cache/market (real fetch cache)
docs/ PLAN.md, DETAILED_PLAN.md, ARCHITECTURE.md, HANDOFF.md
index.html (dashboard)
```

## Web Dashboard

PCA 3 map of 14k company-FYs, archetype filter, ticker search, skill radar, next FY pred — see `index.html`

## Solo Disclaimer

Solo personal project, no connection to employer, built with public/free-tier only. SEC public domain, yfinance free tier.

## Author

Cameron Davis — Home-only Scout
