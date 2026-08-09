# DATA_MODEL_equities — Vector Equities

### Source
- SEC EDGAR CompanyFacts XBRL (2015-2024) — income, balance, cashflow, 10-K Item 1/1A/7 text chunks
- yfinance market (price, volume, market cap)
- DEF 14A Neo scaffolding (offline fallback)

### Normalization
- Per-FY z-score within fiscal year, winsor ±4, FY context-honest.
- Coverage scalar mean mask to prevent zero-impute bias.

### Features
- 122 features across 17 families: income, balance, cashflow, growth, profitability, leverage, efficiency, per-share, market, valuation, management_neo, ownership, disclosure_text, sector_context, macro_regime, form, bbref_bridge (cross-sport legacy)

### Towers
- ResidualTower `cat([x·m,m])→96h→24d` LayerNorm skip L2 per family, m = coverage mask.

### Fusion
- Transformer fusion d_model 128, 4 layers, 4 heads, CLS token + FY embedding 12-d + 17 tower tokens = 19 tokens → CLS 128→64 L2.

### Heads
- 8 archetypes, 11 GICS sectors, 14-d profile, next-year profile 14-d, 12 skill grades, valuation, market, health, payout.

### Skills
- 12 Financial Crafts: Profitability, Growth, Moat, Cash_conversion, Capital_alloc, Balance_health, Efficiency, Valuation_discipline, Momentum, Mgmt_quality, Yield, Disclosure.

### Eval
- sector purity@10 0.7057 lift 6.32x (n=4831, baseline random 0.1117), cross-ticker 0.4013 lift 3.59x, silhouette -0.0034 vs -0.0204, forward IC gate IC>0 (triple-barrier 21.9% hit rate), composite 0.4*recall_no_wiki+0.25*purity+0.2*next_R2+0.15*sector.

### Assets
- `assets/real_data.json` points xyz + 12 grades + embedding, `assets/eval_sector_coherence.json`, `assets/eval_scoreboard.json`.

### Provenance
- 2026-08-05 500 tickers v6 real matrix, 4,831 FYs, measured purity 0.7057. Older 0.174 was placeholder pre-expansion.
