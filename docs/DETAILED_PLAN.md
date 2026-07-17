# Vector Equities — Detailed Execution Plan (Step-by-Step)

This doc mirrors user request: "Make a detailed plan for each step in the process, then create a dynamic todo list for each step and execute step by step constantly updating your todos"

## Phase 0: Review Hoops MTNN

- Read `vector-hoops/pipeline/train_mtnn.py` 1900 lines, `mtnn_validation.py`, `composite_score.py`, `build_vectors.py`
- Extract patterns: 17 families, ResidualTower cat([x·m,m]) -> 96h -> 24d, Gated/Concat/Transformer fusion, 12-d season emb, 48-d L2, InfoNCE same-player adjacent, hard-neg boost, AdamW no-decay, OneCycle 10% warmup, masked MSE, CQS = recall*0.35 + purity*0.25 + ...

Done — scaffolded `vector-equities/README.md` + `docs/PLAN.md`

## Phase 1: Data Spec & Pipelines (Holistic)

Goal: all data you can find for public companies like hoops holistic.

Families designed (122 feats):
- Financial statements: income (15), balance (10), cashflow (7) -> from SEC XBRL us-gaap Revenues, GrossProfit, NetIncomeLoss, Assets, Liabilities, Equity, Cash, Debt, etc
- Growth (9): YoY + 3Y CAGR
- Profitability (5): ROE, ROA, ROIC, margins
- Leverage/Liquidity (7): Current, Debt/Eq, Debt/EBITDA, Interest Coverage
- Efficiency (5): Asset Turnover, Inventory, Receivables
- Per-share (5): EPS, BVPS, FCFPS, dilution
- Market (10): ret 1/3/6/12M, vol 30/90/252, beta, momentum (yfinance free)
- Valuation (8): PE, PB, PS, EV/EBITDA, EV/Sales, EY, FCF yield
- Management NEO (14): from DEF 14A — neo count, CEO age/tenure/founder flag, total comp log, equity %, avg NEO comp, pay ratio, board indep %, board size, insider own %, CEO pay vs sector, turnover, duality
- Ownership (6): inst %, delta, insider net 12M, float, concentration, short interest
- Disclosure text (6): MDA length, sentiment (Loughran-McDonald), risk factor count, change YoY, Fog proxy, uncertainty
- Sector context (3), Macro regime (4): rates 10Y, VIX, credit spread, GDP
- Form (6): earnings surprise streak, guidance raise, EPS revision, price vs 52W, RSI proxy
- Bridge (2): Altman Z, Piotroski F

Fetchers:
- `fetch_sec.py`: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json with User-Agent VectorEquities, caches to pipeline/cache/sec — public domain
- `fetch_market.py`: yfinance Ticker history free tier -> returns, vol, beta, cache
- `parse_neo.py`: DEF 14A regex parser scaffold, founder detection, age extract, compensation fields
- Synthetic fallback: `build_demo_v3.py` generates 1200x12 with sector+archetype biased bases + AR1 continuity 0.80 (tunable) for offline CI, honest per-FY z-score

Todo tracker: `docs/PLAN.md` updated live

## Phase 2: Skills & Archetypes

- 12 skills composite linear like hoops skill lens: grades 0-100 percentile per FY, weights defined in `build_skills.py`
- 8 archetypes via k-means on 14-d game profile (financial fingerprint) — auto-named list but centroids stored, labeling via inspection

## Phase 3: MTNN Model

Ported architectures verbatim:
- `_ResBlock`, `ResidualTower`, `GatedFusion`, `ConcatFusion`, `TransformerFusion`, `SkillTowers`, `EquitiesMTNN`
- Fusion modes same knobs as hoops: tower_width, tower_hidden, skill_hidden, d_model, n_fusion_layers, fusion_hidden (previously unswept param that is 57% params)
- Heads: archetype 8, sector 11, profile 14, next_profile 14, skills 12 mini-towers, valuation, market, vol, health, payout, mgmt, own
- Loss weights rebalanced: archetype 0.25 sector 0.15 profile 0.12 next 0.10 skills 0.20 valuation 0.12 market 0.12 etc

## Phase 4: Training & Eval

- `train_mtnn.py`: InfoNCE temp 0.08 same-ticker adjacent, same-sector hard-neg boost 0.2-0.3, feature dropout 0.12 two views, AdamW wd 1e-4, OneCycle
- Split: train FY <=2021, val 2022-2023, test >=2024 (like hoops y<=2021 train etc)
- Metrics: recall@10, purity@20, sector acc, next R2, market directional, CQS composite (0.35 recall +0.25 purity +0.20 R2 clip +0.10 sector +0.10 market bonus)
- Checkpoint: best composite proxy (0.5 recall +0.5 purity) not recall-only (fixes bug where epoch 0 recall 1.0 picked before learning)
- Training matrix: train_matrix.npz Z [N,D] mask, fiscal_year, ticker, sector, cluster
- Runs: 8000 rows 30 epochs gated ~40 sec CPU, 14400 rows 40 epochs gated ~2 min, transformer ~5-7 min

## Phase 5: Web Artifact + GitHub

- Dashboard: `index.html` will show PCA 3 company map (like hoops), archetype explorer, ticker search, skill lens radar, next FY prediction vs actual
- Git repo: init with MIT + disclaimer footer, pushable structure, docs, pipeline, assets
- Final review: user reviews github repo (local repo path ~/workspace/vector-equities)

## Live Progress Log (updated as we go)

- 11:44 start, review hoops MTNN 1900 lines
- 11:46 scaffold repo + README + PLAN
- 11:47 feature_spec 122 feats 17 families + synthetic v1 8000 rows
- 11:48 build_skills + build_archetypes done
- 11:49 first train: recall 0.002 (continuity too low)
- 11:52 build_demo_v2 continuity 0.88 -> recall 1.0 purity 0.53 (too high)
- 11:53 build_demo_v3 continuity 0.72 + sector bias -> sector 86.8% market 80.6% R2 0.40 but recall 2.6%
- 11:59 retry continuity 0.80 balanced run (in progress) — targeting recall 0.4-0.7 + sector 0.7+ + R2 0.3+
- Next: finalize repo, web artifact, git init, push docs
