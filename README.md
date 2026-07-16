# Vector Equities — MTNN for Public Companies — REAL SEC LIVE

**Solo personal project, no connection to employer, built with public/free-tier only — Port of Vector Hoops MTNN**

**LIVE PROD v2: 2741 company-FYs • 283 tickers • 17 towers • 64-d transformer MTNN • val 88.2% test 86.2% purity 65.9% CQS 0.635**

Hoops 12,966 player-seasons → Equities REAL: SEC EDGAR CompanyFacts XBRL (curl fallback, UA Cameron Davis jcdavis131@gmail.com, <10 req/s) 2015-2024 summary cache 20 tags <5KB each, per-FY z ±4, 122 feats, 17 families, 12 skills, 8 archetypes, 11 GICS

## Live
- **GitHub:** https://github.com/jcdavis131/vector-equities
- **Prod:** https://equities.dumbmodel.com (Vercel import pending, assets ready) + equities.jcamd.com → equities.dumbmodel.com via vercel.json
- **Hub:** https://jcamd.com — now lists Equities alongside Hoops/Pitch/Gridiron
- **Demo:** `index.html` loads `assets/real_data.json` 2741 points (x,y,z PCA of 64-d, skills 0-100, emb[32])

## Quickstart — REAL Pipeline

```bash
cd ~/workspace/vector-equities
# 1. Fetch SEC summaries (free, <60MB RSS, <10 req/s) — 282 files for 300 tickers
python3 pipeline/fetch_sec_summary.py --limit 100 --start 0
python3 pipeline/fetch_sec_summary.py --limit 100 --start 100
python3 pipeline/fetch_sec_summary.py --limit 100 --start 200
# → cache/sec/sec_summary/summary_*.json (282 files)

# 2. Build real matrix
python3 pipeline/build_real_from_summary.py --limit 300
# → train_matrix.npz (2741,122) 283 tickers avg 9.7 FY

# 3. Skills + archetypes
python3 pipeline/build_skills.py
python3 pipeline/build_archetypes.py

# 4. Train MTNN
python3 -u pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128 --batch 256 --tower-blocks 2
# → mtnn_best.pt 3M, embedding.npz (2741,64), mtnn_report.json val 88.2% test 86.2% CQS 0.635

# 5. Regen assets (true 64-d)
python3 pipeline/regen_assets.py
# → assets/real_data.json 2741 points, manifest.json, real_data_latest 283, real_pca.json
```

## Results v2 vs v1
- **v1 (100 tickers):** 964 rows, 864 pairs, transformer d64 50ep val 69.8% test 89% all 91.8% purity 64.2% sector 61.9% CQS 0.611
- **v2 (300 tickers):** **2741 rows, 283 tickers, 2458 pairs (1611 train/564 val/283 test), 60ep best_epoch55 val 88.2% test 86.2% all 95.6% purity 65.9% sector 55.4% next R2 0.314 CQS 0.6347** — **+18.4pts val, +123% over gated baseline**

Scale: 2.84x rows, 2.83x tickers, same 17 towers (income 15, balance 10, cashflow 7, growth 9, profitability 5, leverage_liquidity 7, efficiency 5, per_share 5, market_price 10, valuation 8, management_neo 14, ownership 6, disclosure_text 6, sector_context 3, macro_regime 4, form 6, bbref_bridge 2)

## Data Pipeline — Real SEC

- Source: SEC EDGAR CompanyFacts XBRL via `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`
- Client: curl -sL fallback, UA `Cameron Davis jcdavis131@gmail.com`, sleep 0.25s (<10 req/s), RSS <60MB
- Universe: `pipeline/data/universe.json` 300 tickers S&P 100 + 200 large cap, CIK mapped via SEC tickers.json
- Build: `build_real_from_summary.py` per-FY median impute + z ±4 (Hoops DNA)
- Market: `fetch_market_300.py` single-ticker yfinance loop, 601 market jsons (300 universe done MAR etc)

## Architecture

- **EquitiesMTNN**: 17 ResidualTower `cat([x·m, m])` where `d_cat = d_in*2`, `fc1 [96, d_in*2]`, 2 blocks 96→24 skip
- Fusion: transformer (4 layers, 4 heads, d_model 128) vs gated/concat
- Season emb: `n_seasons = len(uniq_years) = 10` shape [10,12]
- Loss: archetype 0.25 sector 0.15 profile 0.12 next 0.10 skills 0.20 valuation 0.12 market 0.12
- Contrastive: InfoNCE temp 0.08 same-ticker adjacent FY + feat dropout 0.12 two views + same-sector hard-neg boost 0.3

## Embedding Export Fixed (critical)

- Previous bug: embedding.npz was Z-fallback (PCA of raw), not model
- Fixed: built `fam_dims=Counter(families)`, `xs,ms` raw, `season_ids` from fy uniq, loaded `mtnn_best.pt` with `EquitiesMTNN(fam_dims, n_seasons=10, d_tower=24..)`
- Export true 64-d: `E (2741,64) mean -0.0059`

## Vercel / Dumbmodel

- `vercel.json` cleanUrls + redirect equities.jcamd.com → equities.dumbmodel.com
- Import repo `jcdavis131/vector-equities` in Vercel dashboard, add domains `equities.dumbmodel.com` + `equities.jcamd.com`
- Hub `jcamd.com` already updated to include Equities card

## Solo Disclaimer

Solo personal project, no connection to employer, built with public/free-tier only. SEC public domain, yfinance free tier.

