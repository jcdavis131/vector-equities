# Vector Equities — REAL SEC Live Prod Report

**Final Checkpoint: 2026-07-16 14:18 CT — 964 company-FYs • 100 tickers • Transformer beats Gated**

Solo personal project, no connection to employer, built with public/free-tier only

---

## Data Pipeline — Real SEC

- Source: SEC EDGAR CompanyFacts XBRL via `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`
- Client: curl -sL fallback, UA `Cameron Davis jcdavis131@gmail.com`, sleep 0.25s (<10 req/s), RSS <60MB
- Summary cache: `pipeline/cache/sec/sec_summary/summary_*.json` — 20 tags <5KB each (Revenues, NetIncomeLoss, GrossProfit, Assets, Liabilities, Equity, Cash, Debt, EPS, OCF, CAPEX, etc) + 10 FYs 2015-2024
- Universe: `pipeline/data/universe.json` 300 tickers S&P 100 + 200 large cap, CIK mapped via SEC tickers.json
- Build: `build_real_from_summary.py` per-FY median impute + z ±4 (Hoops DNA), 17 families ~122 feats, 12 skills, 11 GICS, 8 archetypes k-means on 14-d profile
- Matrix: `train_matrix.npz` **964 rows x 122 feats, 100 tickers, avg 9.6 FY per ticker, sectors 12, FYs 2015-2024**

Example rows:
- MMM Revenue $24.575B 2024, BRK-B $371.433B etc (see /tmp/fetch_100.log)

### Market Enrichment (in-progress demo for 100)

- `fetch_market.py` single-ticker yfinance ok: `MMM hist (502,7)` → last_close, avg_vol_30d, ret_12m, vol_252d, price_vs_52w
- Batch via loop `pipeline/cache/market/{ticker}.json` — 47/100 done (single request at a time, memory safe)
- Future: merge market_price family (10 feats) currently placeholders 0.015, but sec_only=True for clean baseline

## Model

- Architecture: `EquitiesMTNN` Hoops port — 17 ResidualTower `cat([x·m, m])` where `d_cat = d_in*2`, `fc1 [96, d_in*2]`, 2 blocks 96→24 skip, fusion modes gated/concat/transformer
- Feature manifest: `feature_manifest.json` 122 feats families list, `towers = Counter(families)` = income 15, balance 10, cashflow 7, growth 9, profitability 5, leverage_liquidity 7, efficiency 5, per_share 5, market_price 10, valuation 8, management_neo 14, ownership 6, disclosure_text 6, sector_context 3, macro_regime 4, form 6, bbref_bridge 2
- Season emb: `n_seasons = len(uniq_years) = 10` not n_sectors, shape [10,12], FY id mapping

### Training Final

#### Gated Baseline (rejected)
- d48 b256 40 epochs: val_recall@10 0.402, test 0.03, all 0.72, purity 0.616, CQS 0.2737, pairs 864

#### Transformer Final (PROD PROMOTED)
- Config: `transformer d64 d_model 128 4 layers 4 heads` d_tower 24, d_tower_hidden 96, d_emb 64, batch 256, epochs 50
- Metrics epoch 40: val 0.698, test 0.89, all 0.918, purity 0.642, sector 0.619, CQS 0.6108
- File: `mtnn_best.pt 2.6M`, `mtnn_report.json`
- Archive logs: `/tmp/mtnn_final_transformer.log`

Improvement: +57% params over gated 48-d, val +29.6 pts (40.2→69.8), test +86 pts (3→89), CQS 0.274→0.611 (+123%)

Loss: archetype 0.25 sector 0.15 profile 0.12 next 0.10 skills 0.20 valuation 0.12 market 0.12
Contrastive: InfoNCE temp 0.08 same-ticker adjacent FY + feat dropout 0.12 two views + same-sector hard-neg boost 0.3

Split: Train FY<=2020? Actually split by positive pairs chronological, holdout 10% val 10% test = 565 train /199 val /100 test pairs = 864 total

## Embedding Export — FIXED

- Previous bug: `embedding.npz` was Z-fallback (PCA of raw), not model
- Fixed 2026-07-16 14:15: built `fam_dims=Counter(manifest['families'])`, `xs,ms` raw, `season_ids` from fy uniq, loaded `mtnn_best.pt` with `EquitiesMTNN(fam_dims, n_seasons=10, d_tower=24..)`
- Exported true 64-d: `E (964,64) mean -0.0059`
- Assets:
  - `assets/real_data.json` 964 points each {ticker,year,sector,archetype,x,y,z,skills[12],emb[32],skill_keys}
  - `assets/real_pca.json` same
  - `assets/real_data_latest.json` 100 latest FY per ticker
  - `assets/manifest.json` rows 964 tickers 100 feats 122 towers 17 model transformer dim64 CQS 0.6108 val 0.698 test 0.89

## Frontend

- `index.html` header updated: REAL SEC DATA 964 company-FYs • 100 tickers • 17 towers • 64-d transformer MTNN • CQS 0.611 • val 69.8% test 89%
- Loader now handles `real.points` + full dim 64, normalizes
- KPIs: purity 64.2% sector 61.9% val 69.8% test 89% CQS 0.611 towers 17 dim 64 pairs 864

## Next Scale to 300

- Fetch: `fetch_sec_summary.py --limit 100 --start 100` (running pid 9378) then `--start 200` timeout 600
- Verify: `ls cache/sec/sec_summary | wc -l` → expect 300
- Build: `build_real_from_summary.py --limit 300` → ~3000 rows x10 FY avg ~ 2800 after sparsity
- Retrain: `train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128 --batch 256` + gated d64 baseline
- Archive: `mtnn_report.json` with transformer best

## Market Enrichment Plan

- After 100 market jsons, add merge into matrix: market_price family real values replace placeholders, re-z per FY
- Verify continuity still 0.80+

## Files

- `pipeline/cache/sec/sec_summary/` 102 files now (47 market)
- `pipeline/data/train_matrix.npz` (964,122)
- `pipeline/data/embedding.npz` (964,64) TRUE
- `pipeline/data/mtnn_best.pt` 2.6M
- `assets/` 3.1M total

## Solo Disclaimer

Solo personal project, no connection to employer, built with public/free-tier only. SEC public domain, yfinance free tier.

