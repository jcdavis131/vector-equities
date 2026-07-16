# Vector Equities — MTNN for Public Companies — REAL SEC LIVE

**Solo personal project, no connection to employer, built with public/free-tier only — Port of Vector Hoops MTNN**

**LIVE PROD: 964 company-FYs • 100 tickers • 17 towers • 64-d transformer MTNN • val 69.8% test 89% purity 64.2% CQS 0.611**

Hoops 12,966 player-seasons → Equities REAL: SEC EDGAR CompanyFacts XBRL (curl fallback, UA Cameron Davis, <10 req/s) 2015-2024 summary cache 20 tags <5KB each, per-FY z ±4, 122 feats, 17 families, 12 skills, 8 archetypes, 11 GICS

## Quickstart — REAL Pipeline

```bash
cd ~/workspace/vector-equities
# 1. Fetch SEC summaries (free, <60MB RSS, <10 req/s)
python3 pipeline/fetch_sec_summary.py --limit 100
# → cache/sec/sec_summary/summary_*.json (99 files for first 100 tickers)

# 2. Build real matrix
python3 pipeline/build_real_from_summary.py
# → train_matrix.npz (964,122) 100 tickers avg 9.6 FY

# 3. Skills + Archetypes
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py

# 4. Train transformer (PROD)
python3 pipeline/train_mtnn.py --epochs 50 --dim 64 --fusion transformer --d-model 128 --batch 256
# → epoch 40 val 0.698 test 0.89 all 0.918 purity 0.642 sector 0.619 CQS 0.6108
# gated baseline was val 0.402 test 0.03 CQS 0.274 (rejected, +57% params win)

# 5. Export true 64-d embedding (FIXED — not Z-fallback)
python3 -c "
import json, torch, numpy as np
from collections import Counter
# see docs/REAL_PROD_REPORT.md for full export
"
# → pipeline/data/embedding.npz (964,64) mean -0.0059
# → assets/real_data.json 964 points, manifest.json, real_data_latest.json 100

# 6. Dashboard
open index.html  # shows 964 PCA map, archetype filter, skill radar, similarity
```

## Latest Results — REAL SEC 100 tickers

```
Matrix: 964 rows x 122 feats, 100 tickers, 12 sectors, FYs 2015-2024 avg 9.6/ticker
Pairs: 864 adjacent FY same ticker (565 train /199 val /100 test)

Gated d48 b256 40ep (rejected):
  val R@10 0.402 test 0.03 all 0.72 purity 0.616 CQS 0.2737

Transformer d64 d_model128 4L4H 50ep (PROMOTED PROD):
  val 0.698 (69.8%) test 0.89 (89%) all 0.918 purity 0.642 sector 0.619 CQS 0.6108
  checkpoint mtnn_best.pt 2.6M
  Improvement: +29.6pts val +86pts test CQS +123% vs gated

Next profile: MAE z 0.216 val /0.27 test R2 0.150 val 0.134 test
```

Hoops reference: recall 0.977 purity 0.6717 CQS 0.7937 leakfree

## Architecture — 17 towers

- Families: income 15, balance 10, cashflow 7, growth 9, profitability 5, leverage_liquidity 7, efficiency 5, per_share 5, market_price 10, valuation 8, management_neo 14, ownership 6, disclosure_text 6, sector_context 3, macro_regime 4, form 6, bbref_bridge 2
- Tower: cat([x·m, m]) → d_in*2 → 96h GELU LN → 24d + skip, 2 blocks
- Fusion: transformer token list T=17 towers + FY token 12-d + CLS → 4 layers self-attention d_model 128 → CLS → 64-d L2
- Season emb: n_seasons=10 (uniq fiscal_year 2015-2024) shape [10,12]
- Heads: archetype 8, sector 11, profile/next 14-d, skills 12 mini-towers 0-100, valuation, market, health, payout, mgmt, own
- Loss: InfoNCE temp 0.08 same-ticker adj FY + feat dropout 0.12 + same-sector hard-neg 0.3, archetype 0.25 sector 0.15 profile 0.12 next 0.10 skills 0.20 valuation 0.12 market 0.12

## Data Pipeline — free-tier

- SEC: `fetch_sec_summary.py` CompanyFacts 20 tags via curl fallback, 10 FYs, <5KB each, cache
- Build: `build_real_from_summary.py` per-FY median impute z ±4 Hoops DNA
- Market demo: `fetch_market.py` single-ticker yfinance ok (MMM hist 502,7) → cache/market/{ticker}.json last_close, avg_vol_30d, ret_12m, vol_252d
- Synthetic fallback: build_demo_v3.py 14,400 rows 122 feats continuity 0.80 (unused in prod)

## Repo Layout

```
pipeline/
  feature_spec.py, build_real_from_summary.py, build_demo_v3.py, build_skills.py, build_archetypes.py
  model.py (EquitiesMTNN gated/concat/transformer), train_mtnn.py, composite_score.py
  fetch_sec_summary.py, fetch_market.py, fetch_sec.py (old), parse_neo.py
  data/ train_matrix.npz (964,122), embedding.npz (964,64 TRUE), mtnn_best.pt 2.6M, mtnn_report.json, feature_manifest.json
  cache/sec/sec_summary/ 102 files (100 tickers + extras), cache/market/ 47 files in-progress
docs/ ARCHITECTURE.md, REAL_PROD_REPORT.md, PLAN.md
assets/ real_data.json 1.3M 964 points, real_data_latest.json 100, manifest.json, real_pca.json
index.html 964 FYs • 100 tickers • 64-d transformer • CQS 0.61 dashboard
```

## Web Dashboard

`index.html` — PCA-3 964 company-FY map, color archetype 8, size momentum, filter sector/arch/ticker, inspect embedding 64-d, skill radar 12, top similar cosine 64-d, next FY pred. Stats: 964 rows 864 pairs val 69.8% test 89% purity 64.2% sector 61.9% CQS 0.611 towers 17 dim 64.

## Scale to 300

- Running: `fetch_sec_summary.py --limit 100 --start 100` (pid 9378) → then --start 200 timeout 600 → verify 300 files
- Build: `build_real_from_summary.py --limit 300` → ~3000 rows
- Retrain: `train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128 --batch 256`
- Market: finish 100 market jsons then merge into market_price family

## Solo Disclaimer

Solo personal project, no connection to employer, built with public/free-tier only. SEC public domain, yfinance free tier.

## Author

Cameron Davis — Home-only Scout
