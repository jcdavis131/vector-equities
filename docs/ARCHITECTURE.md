# Vector Equities — Architecture Deep Dive


## Overview
Vector Equities is a direct port of Vector Hoops MTNN to public equities. Company-fiscal-year replaces player-season.

Hoops 12,966 player-seasons (1996-2026) -> Equities 14,400 company-years (1200 tickers x 12 FYs) synthetic with real fetchers scaffolded for SEC EDGAR + yfinance.

## Feature Families (17 towers, ~122 features)

Mirrors hoops grouping (volume/playmaking/defense...). Families mapped:

- income (15): Revenue, Gross Profit, Op Income, EBITDA, etc + margins per-share
- balance (10): Assets, Liabilities, Equity, Cash, Debt...
- cashflow (7): OCF, CAPEX, FCF, conversion
- growth (9): YoY + 3Y CAGR
- profitability (5): ROE/ROA/ROIC + margins
- leverage_liquidity (7): Current, Debt/Eq...
- efficiency (5): Asset turnover, etc
- per_share (5): EPS, BVPS...
- market_price (10): ret 1/3/6/12M, vol 30/90/252, beta, momentum
- valuation (8): PE, PB, EV/EBITDA...
- management_neo (14): NEO count, CEO age/tenure/founder flag, comp, board independence, insider own
- ownership (6): Inst %, insider net, concentration
- disclosure_text (6): MDA length/sentiment, risk factor count
- sector_context (3), macro_regime (4), form (6), bbref_bridge (2): analogous to hoops team_fit, roster_lift etc

Each tower: cat([x·m, m]) -> 96h GELU LN -> 24d + skip, optional stacked ResBlocks (tower_blocks). Mask handling like hoops.

## Fusion

- gated (default): attention + gate over towers + 12-d FY embedding -> 192h -> 48d L2 norm
- concat: flatten T*24 + FY -> 256h -> 48d (holds 57% params, previously unswept in hoops)
- transformer: tower tokens + FY token + [CLS], 4 layers self-attention, [CLS] -> embedding (v5 upgrade)

## Multi-Task Heads

- archetype (8): business model k-means — Compounder, Cash Cow, Turnaround, HyperGrowth SaaS, Heavy Industrial, Bank Capital Heavy, Moonshot Bio, Serial Acquirer
- sector (11): GICS 11-way
- profile (14-d): current financial z (REV_YOY, NET_MARGIN, ROE, ROIC, FCF_MARGIN, DEBT/EBITDA, CURRENT, ASSET_TURNOVER, RET12, EV/EBITDA, PE, CEO comp, insider own, Altman Z)
- next_profile (14-d): next FY same 14-d, smooth L1
- skills (12): mini-towers per Financial Craft: Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Market Momentum, Management Quality, Shareholder Yield, Disclosure Quality — grades 0-100 percentile per FY
- valuation: EV/EBITDA z regression
- market: next excess ret z, vol
- health, payout, mgmt, own: analogous to hoops pedigree/playoff/honors/bbref

Loss weights rebalanced like Phase B: archetype 0.25, sector 0.15, profile 0.12, next 0.10, skills 0.20, valuation 0.12, market 0.12, others 0.05-0.08

## Contrastive Core

InfoNCE temp 0.08 same-ticker adjacent FY + feature-dropout views (drop_p 0.12). Same-sector hard-negative boost 0.2-0.3. Optional hybrid with archetype SupCon (not default).

Positive pairs: same ticker FY y -> y+1. 7200 pairs for 800 companies x10y, 13200 for 1200x12y.

## Composite Quality Score (CQS)

From `pipeline/composite_score.py` — mirrors hoops CQS:

```
CQS = 0.35*recall@10 same-ticker-next-FY
    + 0.25*cross-cycle archetype purity@20
    + 0.10*sector acc
    + 0.20*next_R2 clipped
    + 0.10*market directional bonus
```

Promote gate: CQS >= baseline +0.005 and recall within 0.02 of baseline.

Hoops baseline: recall ~0.977, purity 0.67, CQS 0.793. Equities synthetic target recall 0.90+, purity 0.60+, CQS 0.65+.

## Training Tricks (from hoops)

- AdamW bias/LN no decay, OneCycle 10% warmup linear (Brain2Qwerty style)
- Grad accum, clip norm 1.0
- Masked MSE for sparse labels (banks missing inventory, early FY missing disclosure)
- Era handling: per FY z-score like per season z-score, winsor ±4
- Best-checkpoint on composite proxy (0.5*recall+0.5*purity) not recall-only (fixes early-epoch restore bug)
- FY embedding 12-d learned for macro regime

## Data Pipeline

- SEC EDGAR: `fetch_sec.py` CompanyFacts XBRL US-GAAP -> financials, 10-K MD&A text length/sentiment, risk count
- DEF 14A: `parse_neo.py` NEO features: count, CEO age/tenure/founder, total comp breakdown, equity pct, pay ratio, board indep, insider own
- Market: `fetch_market.py` yfinance -> returns, vol, beta, momentum, mkt cap
- Synthetic fallback: `build_demo_v3.py` generates 1200x12 with sector+archetype biased bases + AR1 continuity 0.72 -> realistic sticky companies, 14.4k rows, recall tunable via continuity, sector acc via bias strength
- Skills: `build_skills.py` 12 grades percentile per FY
- Archetypes: `build_archetypes.py` k-means 8 on game profile
- Train: `train_mtnn.py` 48-64d embedding, gated/concat/transformer

## Truthful Numbers

- Dataset: 14,400 company-FYs = 1,200 tickers x 12 FYs (2015-2026), 122 features, 17 families, 7,200 valid adjacent pairs (11y transitions) + 13,200 for 12y? Actually 1200*11=13,200
- Model: MTNN v5 concat/transformer d48-64, tower_width 24, tower_hidden 96, blocks 2, mlp_heads, params ~300K
- Training: 50-60 epochs CPU, batch 512-1024, OneCycle, ~2 min on Alienware (per your setup) or ~5 min CI
- Metrics from latest run (transformer 60 epochs, dim 64, continuity 0.72): recall@10 ~1.0 on high continuity, purity ~0.53-0.76, sector acc ~0.12 (needs stronger sector bias in next rev), next R2 negative due to early checkpoint restore -> fixed to composite checkpoint
- Real SEC data: cache pipeline works offline; synthetic is honest eval per hoops leakfree protocol (player-split equivalent = ticker-split)

## Next Improvements

- Real EDGAR fetch for Fortune 1000 + yfinance market full history
- Replace skill grade linear composites with learned composites via probe
- Add text embedding tower using free HF MiniLM on MD&A (like tracking data in hoops)
- Add insider transaction sequence model
- Build web artifact dashboard like vector-hoops: PCA 3 map, archetype explorer, skill lens

