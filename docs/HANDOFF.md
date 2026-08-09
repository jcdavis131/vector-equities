# Vector Equities — HANDOFF

Current state as of 2026-08-09 — coherence pass (4,831 FYs 500 tickers 64-d purity 0.7057 lift 6.32x) / previous baseline 2026-07-16.

> Solo personal project, no connection to employer, built with public/free-tier only (free data pipeline, static Vercel).

> Picking up in-progress work? This file is the entry point. Parity target: vector-hoops README style.

## What was built

Ported Vector Hoops MTNN (12,966 player-seasons, 17 tower families, 48-d embedding, multi-task heads, CQS composite) to public companies: **Vector Equities** — now at 4,831 company-FYs, 500 tickers, 2015-2024 SEC EDGAR XBRL real pipeline (v6 real towers).

- **Dataset (served 2026-08-05):** 4,831 company-FYs, 500 tickers, 11 GICS sectors, 138 features across 17 families (income 15, balance 10, cashflow 7, growth 9, profitability 5, leverage_liquidity 7, efficiency 5, per_share 5, market_price 10, valuation 8, management_neo 14, ownership 6, disclosure_text 6, sector_context 3, macro_regime 4, form 6, bbref_bridge 2) + wiki 16-d optional = see `assets/feature_manifest_v6_real.json`. Per-FY z-score winsor ±4 honest like hoops. Proof files: `assets/real_data.json`, `assets/eval_sector_coherence.json` `n_rows_total=4831 n_companies=500`, `assets/eval_scoreboard.json`.
- **Synthetic generator (dormant, for offline tests):** `build_demo_v3.py` continuity 0.80 AR1 with sector + archetype biases (realistic sticky companies). Honest per-FY z-score + winsor ±4 like hoops. Still used for `tests/test_no_fabricated_embedding.py` coverage.
- **Skills:** 12 Financial Crafts Lens (Profitability, Growth, Moat, Cash Conversion, Capital Allocation, Balance Health, Efficiency, Valuation Discipline, Market Momentum, Management Quality, Shareholder Yield, Disclosure Quality) percentile per FY — `pipeline/build_skills.py`
- **Archetypes:** 8 k-means on financial profile: Compounder, Cash_Cow, Turnaround, HyperGrowth_SaaS, Heavy_Industrial, Bank_Capital_Heavy, Moonshot_Bio, Serial_Acquirer — `pipeline/build_archetypes.py`
- **Model:** EquitiesMTNN clone of hoops MTNN — 17× ResidualTower `cat([x·m,m])→96h→24d` LayerNorm skip L2, tokens 17×40→proj128, fusion CLS + FY 12-d→128 + 17 tokens = 19 tokens transformer d_model128 n_layers4 n_heads4 ff512 pre-LN dropout0.15 → CLS 128→512→64 L2 (`assets/eval_scoreboard.json` `embedding_model=equities_mtnn_v_rebuild_d64_transformer` dim 64). Heads: archetype 8, sector 11, profile 14, next_profile 14, 12 skill towers, valuation, market, vol, health, payout, mgmt, own.
- **Training:** AdamW no-decay biases, OneCycle 10% warmup linear, InfoNCE same-ticker adjacent FY + same-sector hard-negative boost 0.3, masked MSE, grad clip, best-checkpoint on composite proxy (0.5*recall+0.5*purity) — fixes hoops bug where epoch 0 recall 1.0 picked. Forward heads IC>0 gate proves future signal, not just label.
- **Metrics served:** k-NN sector purity@10 0.7057 (baseline random 0.1117 lift 6.32×, permutation 0.1106), cross-ticker purity@10 0.4013 (baseline 0.1117 lift 3.59×), silhouette -0.0034 vs -0.0204 permutation, forward IC_rank 1m 0.0051 /3m 0.0064 /6m 0.007 /12m 0.0062 triple-barrier hit 0.2189 n_trades 233 — see `assets/eval_sector_coherence.json` computed 2026-08-05T03:12:21Z + `assets/eval_scoreboard.json` 2026-08-05T03:13:45Z.
- **Fetchers:** SEC EDGAR CompanyFacts (free, User-Agent), yfinance market, DEF 14A NEO parser scaffolds (offline-safe fallback), `pipeline/fetch_sec_summary.py` chunk cache in `pipeline/data/`

## Latest training (2026-08-05)

- Config: real v6 towers, gated transformer fusion, tower_blocks 2, mlp_heads, dim 64, d_model 128 L4 H4, 500 tickers 4831 FYs, batch 1024, epochs 60
- Shipped best (re-build d64 transformer): sector 0.7057 purity gate >0.65 PASS, cross 0.4013 gate >0.35 PASS, forward IC>0 PASS, composite tracked in `eval_scoreboard.json`. Superseded placeholder 0.174 (cross 0.167 lift 1.5×) from older matrix with S&P 500 expansion placeholder rows — kept as provenance note in scoreboard `superseded` field, not shipped metric. Diagnostic runs show model-derived vs placeholder subsets score similarly, but served 2026-08-01 matrix is 500 tickers v6 real.

## File layout

- `index.html` / `play.html` / `companies.html` / `model.html` / `methods.html` — plain HTML/JS/Canvas no framework, PWA `sw.js`
- `assets/real_data.json` — 4831 points xyz, 12 skill grades, embeddings 64-d L2
- `assets/real_pca.json` / `real_pca_full.json` — PCA sky
- `assets/manifest.json` — PWA manifest, duplicate at `/manifest.json`
- `assets/eval_sector_coherence.json` + `assets/eval_scoreboard.json` — shipped eval, gate source
- `assets/feature_manifest_v6_real.json` — 138 feat spec
- `pipeline/fetch_sec_summary.py` — SEC summary fetcher (chunked cache)
- `pipeline/build_real_from_summary.py` — real matrix builder
- `pipeline/build_skills.py` — 12 skills
- `pipeline/build_archetypes.py` — 8 archetypes
- `pipeline/feature_spec.py` — feat spec (no ticker)
- `pipeline/model.py` — MTNN towers + fusion
- `pipeline/train_mtnn.py` — training loop
- `pipeline/composite_score.py` + `eval_sector_coherence.py` + `eval_forward.py` — gates
- `pipeline/towers_v6/` — external towers (industry_gdelt etc, optional)
- `docs/ARCHITECTURE.md`, `DATA_MODEL_equities.md`, `HANDOFF.md`, `PLAN.md`
- `tests/test_eval_sector_coherence.py`, `test_no_ticker_leakage.py`, `test_no_fabricated_embedding.py`, `test_verifier_never_fabricates.py`, `test_calibration_provenance.py`

## How to run (2026-08-09)

```bash
cd ~/workspace/vector-equities
python3 pipeline/fetch_sec_summary.py --limit 300          # SEC EDGAR XBRL + market chunks -> pipeline/data/
python3 pipeline/build_real_from_summary.py --limit 300    # 138 feats 17 families z-score winsor ±4
python3 pipeline/build_skills.py && python3 pipeline/build_archetypes.py
python3 pipeline/train_mtnn.py --epochs 60 --dim 64 --fusion transformer --d-model 128
python3 pipeline/regen_assets.py            # -> assets/real_data.json + pca
python pipeline/eval_sector_coherence.py    # -> 0.7057 / 0.4013
python pipeline/eval_forward.py             # IC>0 gate
python -m http.server 8000                  # open http://localhost:8000
python -m pytest -q
```

For demo compare (synthetic, offline-only):

```bash
python3 pipeline/build_demo_v3.py --companies 1200 --years 12 --continuity 0.80
```

## Next steps / SOTA push (open)

- Real EDGAR full historical backfill 2010-2024 for 500 tickers — verify per-FY impute / coverage scalar hygiene keeps 0.65 gate as n grows
- Add text tower full MDA sentiment Loughran-McDonald + MiniLM 384-d embeddings fusion weight ablation (currently 16-d reduction optional)
- Add insider transaction sequence transformer (Form4 chronograph `docs/DEF14A_FORM4_CHRONOGRAPH_SPEC.md`)
- Web artifact polish: verify 3D map perf 60fps Canvas, dossier deep-link, methods glass-box SHAP
- Hyper-param sweep: tower_hidden 96→128, d-model 96→128→160, fusion_hidden 256→512, nce-temp 0.07-0.10, hard-neg boost 0.3→0.4
- Procrustes era alignment like hoops for rate regimes — FY embedding 12-d currently pos proj gated `year_norm_excluded_from_X` so safe
- Cross-sport unified (equities ↔ hoops MTV) — nano controller ready, measure transfer

## Verification commands (parity with hoops)

```bash
python3 -m json.tool manifest.json > /dev/null && echo "manifest ok"
python3 -m json.tool vercel.json > /dev/null && echo "vercel ok"
python3 -m json.tool assets/eval_scoreboard.json | grep -A2 knn_sector_purity_at_10
python3 -m pytest -q
python3 -m http.server 8011 --bind 127.0.0.1 & curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8011/manifest.json
```

## Provenance note

- Solo personal project, no connection to employer, public/free-tier only, static Vercel.
- Scrubbed internal IDs — docs clean per bundle policy.
- Bundles harness: `bundles/manifest.json` mirrors root harness manifest v3.3-OODA-Agentic-MoMA-Graph-Checkpoint + scout-cli 0.8.0 13 agents / 12 packs / 6 ultra modules (conceptually 11 packs).

