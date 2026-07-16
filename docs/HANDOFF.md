# Vector Equities — HANDOFF

Current state as of 2026-07-16.

## What was built

Ported Vector Hoops MTNN (12,966 player-seasons, 17 tower families, 48-d embedding, multi-task heads, CQS composite) to public companies: **Vector Equities**.

- **Dataset:** 14,400 company-FYs (1,200 tickers x 12 FYs 2015-2026), 122 features across 17 families (income, balance, cashflow, growth, profitability, leverage, efficiency, per-share, market_price, valuation, management_neo, ownership, disclosure_text, sector_context, macro_regime, form, bbref_bridge)
- **Synthetic generator:** `build_demo_v3.py` continuity 0.80 AR1 with sector + archetype biases (realistic sticky companies). Honest per-FY z-score + winsor ±4 like hoops
- **Skills:** 12 Financial Crafts Lens (Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Market Momentum, Management Quality, Shareholder Yield, Disclosure Quality) percentile per FY
- **Archetypes:** 8 k-means on financial profile: Compounder, Cash_Cow, Turnaround, HyperGrowth_SaaS, Heavy_Industrial, Bank_Capital_Heavy, Moonshot_Bio, Serial_Acquirer
- **Model:** EquitiesMTNN clone of hoops MTNN — ResidualTower per family, Gated/Concat/Transformer fusion, FY embedding 12-d, 48-64d L2 norm, heads: archetype 8, sector 11, profile 14, next_profile 14, 12 skill towers, valuation, market, vol, health, payout, mgmt, own
- **Training:** AdamW no-decay biases, OneCycle 10% warmup linear, InfoNCE same-ticker adjacent FY + same-sector hard-negative boost 0.3, masked MSE, grad clip, best-checkpoint on composite proxy (0.5*recall+0.5*purity) — fixes hoops bug where epoch 0 recall 1.0 picked
- **Metrics:** recall@10 same-ticker next FY, cross-cycle archetype purity@20, sector acc, next R2, market directional acc, CQS composite
- **Fetchers:** SEC EDGAR CompanyFacts (free, User-Agent), yfinance market, DEF 14A NEO parser scaffolds (offline-safe fallback)

## Latest training (running)

- Config: 1200x12 continuity 0.80, gated fusion, tower_blocks 2, mlp_heads, dim 48, batch 1024, 40 epochs
- Previous best (continuity 0.72): sector 86.85%, market 80.6%, next R2 0.40 val / 0.26 test, purity 63.6%, recall 2.6% val / 0% test (low recall due to high noise), CQS 0.398
- Earlier high-continuity 0.88 run: recall 1.0, purity 53% but sector 8.7% — tradeoff, need balanced
- Target final: recall 0.6-0.9, purity 0.65+, sector 0.70+, next R2 0.30+, CQS 0.60+

## File layout

- pipeline/build_demo_v3.py — synthetic generator (main)
- pipeline/build_skills.py — 12 skills
- pipeline/build_archetypes.py — 8 archetypes
- pipeline/feature_spec.py — 122 feats spec
- pipeline/model.py — MTNN towers + fusion
- pipeline/train_mtnn.py — training loop
- pipeline/composite_score.py — CQS
- pipeline/fetch_sec.py, fetch_market.py, parse_neo.py — real fetchers (free-tier)
- pipeline/data/ — train_matrix.npz, skill_labels.npz, embedding.npz, mtnn_report.json, mtnn_best.pt
- docs/ARCHITECTURE.md, PLAN.md, HANDOFF.md

## How to run

```bash
cd ~/workspace/vector-equities
python3 pipeline/build_demo_v3.py --companies 1200 --years 12 --continuity 0.80
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py
python3 pipeline/train_mtnn.py --epochs 50 --dim 64 --fusion transformer --tower-blocks 2 --mlp-heads
python3 -m pipeline.composite_score
```

For real SEC data:
```bash
python3 pipeline/fetch_sec.py  # demo AAPL
python3 pipeline/fetch_market.py
```

## Next steps for SOTA push

- Real EDGAR fetch for S&P 1500 -> replace synthetic with true fundamentals (XBRL us-gaap)
- Add text tower: MD&A sentiment via Loughran-McDonald + MiniLM embeddings (free HF)
- Add insider transaction sequence transformer
- Web artifact dashboard: PCA 3 company map, archetype explorer, skill lens, ticker search
- Train final-refit on all rows after promote gate (like hoops auto phase)
- Hyper-param sweep: tower_hidden 96->128, d-model 96->128, fusion_hidden 256, nce-temp 0.07-0.10
- Add Procrustes era alignment like hoops for rate regimes

## Solo disclaimer

Solo personal project, no connection to employer, built with public/free-tier only
